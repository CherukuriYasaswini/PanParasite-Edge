// PanParasite-Edge Phase 6 (Final) implementation: pipeline orchestrator.
// OUR code below performs no malloc/calloc/new/vector/cv::Mat::create.
// Stage order is load-bearing (see pipeline.hpp); do not reorder.

#include "pipeline.hpp"

#include <cstdio>
#include <cstring>

namespace panparasite {

bool PanParasitePipeline::boot(StaticMemoryArena& arena,
                               const PipelineConfig& cfg) noexcept {
  shutdown();
  for (int i = 0; i < 64; ++i) cfg_.nodeId[i] = cfg.nodeId[i];
  for (int i = 0; i < 256; ++i) {
    cfg_.ringPath[i] = cfg.ringPath[i];
    cfg_.dbPath[i] = cfg.dbPath[i];
  }
  cfg_.stage2 = cfg.stage2;
  gate_.reset();
  for (int i = 0; i < kMaxTopoCandidates; ++i) topo_.candidates[i] = TopoCandidate{};
  topo_.numCandidates = 0;
  topo_.globalBeta0 = topo_.globalBeta1 = topo_.globalChi = 0;
  topo_.overflow = false;
  gated_.n = 0;
  gated_.capExceeded = false;
  audit_.int8_runs = 0;
  audit_.fp32_fallback_count = 0;
  for (int i = 0; i < kMaxActiveTracks; ++i) tracks_.tracks[i] = Track{};
  tracks_.nextId = 0;
  for (int i = 0; i < kMaxSlideTracks; ++i) slide_.slots[i] = SlideTrack{};
  for (int i = 0; i < kNumTaxa; ++i) slide_.tally[i] = 0;
  for (int i = 0; i < 37; ++i) slideUuid_[i] = '\0';
  snap_ = UiSnapshot{};
  fieldIndex_ = 0;

  if (!microNetInitBoot(arena)) return false;
  if (!makeSlideUuid(slideUuid_)) return false;
  // Ring: open existing, else format-then-open (first boot / field wipe).
  if (!ringOpen(ring_, cfg_.ringPath, kRingSectors)) {
    if (!ringFormat(cfg_.ringPath, kRingSectors) ||
        !ringOpen(ring_, cfg_.ringPath, kRingSectors)) {
      return false;
    }
  }
  if (!auditOpen(adb_, cfg_.dbPath)) {
    ringClose(ring_);
    return false;
  }
  std::snprintf(snap_.slideUuid, sizeof(snap_.slideUuid), "%s", slideUuid_);
  std::snprintf(snap_.nodeId, sizeof(snap_.nodeId), "%s", cfg_.nodeId);
  snap_.ringDirectIo = ring_.directIo;
  char bootPayload[512];
  std::snprintf(bootPayload, sizeof(bootPayload), "boot node=%s slide=%s",
                cfg_.nodeId, slideUuid_);
  auditAppend(adb_, kEvBoot, bootPayload, nowUs());
  booted_ = true;
  return true;
}

void PanParasitePipeline::shutdown() noexcept {
  ringClose(ring_);
  auditClose(adb_);
  booted_ = false;
}

bool PanParasitePipeline::processField(StaticMemoryArena& arena,
                                       FieldReport& rep) noexcept {
  rep = FieldReport{};
  if (!booted_) return false;
  rep.fieldIndex = fieldIndex_;

  // ---- Stage 1: acquisition gate ----
  const AcquisitionMetrics gm = checkAcquisitionGate(arena, gate_);
  rep.deltaIt = gm.delta_it;
  rep.phiFocus = gm.phi_focus;
  rep.gateLocked = gate_.locked;
  if (!gate_.locked) {
    ++fieldIndex_;  // still counts as an observed (unlocked) field
    return true;
  }

  // ---- Stage 2: integral adaptive binarization + closing ----
  runStage2(arena, cfg_.stage2);

  // ---- Stage 3: dual-connectivity homology ----
  const int nCand = analyzeTopology(arena, topo_);
  rep.numCandidates = (nCand < 0) ? 0 : nCand;

  // ---- Stage 4A: micro-logistic gate (Top K <= 8) ----
  runLogisticGate(topo_, gated_);
  rep.numGated = gated_.n;
  rep.capExceeded = gated_.capExceeded;

  // Precompute bboxes (equivMap compact ids; classify must not run yet:
  // scratch is shared, and integral borrow happens inside classify).
  BBox boxes[kMaxDetections];
  for (int i = 0; i < kMaxDetections; ++i) boxes[i] = BBox{};
  const int nG = (gated_.n < kMaxDetections) ? gated_.n : kMaxDetections;
  for (int i = 0; i < nG; ++i) {
    const int ci = gated_.idx[i];
    if (ci < 0 || ci >= topo_.numCandidates) continue;
    candidateBBox(static_cast<uint32_t>(ci) + 1u, arena.equivMap(),
                  kFrameWidth, kFrameHeight, boxes[i]);
  }

  // ---- Stages 4.1/4.2/5 per candidate: occlusion -> ROI -> diffuse? ----
  // ---- -> classify. Scratch is re-extracted per candidate, so sharing ----
  // ---- is safe and heap-free. ----
  Detection dets[kMaxDetections];
  bool detTrig[kMaxDetections] = {};
  int nDet = 0;
  int nTrig = 0;
  for (int i = 0; i < nG; ++i) {
    const int ci = gated_.idx[i];
    if (ci < 0 || ci >= topo_.numCandidates || !boxes[i].valid) continue;
    const double o = computeOcclusion(topo_.candidates[ci], ci, arena);
    extractCandidateROI(arena, boxes[i]);
    const bool trig = needsActivePerception(o);
    if (trig) {
      diffuseScratchROI(arena, kPMIters);
      ++nTrig;
    }
    const MicroNetProbs pr =
        microNetClassify(arena, topo_.candidates[ci].fbsc, audit_);
    if (pr.argmax < 0) continue;
    Detection& d = dets[nDet];
    detTrig[nDet] = trig;
    ++nDet;
    d.x0 = static_cast<float>(boxes[i].x0);
    d.y0 = static_cast<float>(boxes[i].y0);
    d.x1 = static_cast<float>(boxes[i].x1);
    d.y1 = static_cast<float>(boxes[i].y1);
    d.cls = pr.argmax;
    double best = 0.0;
    for (int k = 0; k < kNumTaxa; ++k) {
      if (pr.p[k] > best) best = pr.p[k];
    }
    d.conf = best;
  }
  rep.numDetections = nDet;
  rep.numTriggered = nTrig;

  // ---- Stage 6: Hungarian track + log-odds + slide commit ----
  trackerStep(tracks_, nDet > 0 ? dets : nullptr, nDet);
  slideCommit(slide_, tracks_);
  rep.slideVerifiedTotal = verifiedTotal();
  rep.fp32Fallbacks = audit_.fp32_fallback_count;

  // ---- Stage 7: persist field summary (ring sector + ledger row) ----
  char payload[512];
  std::snprintf(payload, sizeof(payload),
                "f=%d det=%d trig=%d ver=%d cap=%d fb=%d",
                rep.fieldIndex, rep.numDetections, rep.numTriggered,
                rep.slideVerifiedTotal, (int)rep.capExceeded,
                rep.fp32Fallbacks);
  const uint64_t ts = nowUs();
  ringAppend(arena, ring_, kEvFieldSummary,
             reinterpret_cast<const uint8_t*>(payload),
             std::strlen(payload), ts);
  if (rep.capExceeded) {
    auditAppend(adb_, kEvCapExceeded, "CANDIDATE_CAP_EXCEEDED", ts);
  }
  auditAppend(adb_, kEvFieldSummary, payload, ts);

  // ---- Stage 8: snapshot publish (server, if attached, renders it) ----
  snap_.fields = fieldIndex_ + 1;
  snap_.gateLocked = true;
  snap_.focusPhi = rep.phiFocus;
  snap_.deltaIt = rep.deltaIt;
  snap_.capExceeded = rep.capExceeded;
  snap_.fp32Fallbacks = rep.fp32Fallbacks;
  int nVer = 0, nRej = 0, nAct = 0;
  for (int i = 0; i < kMaxActiveTracks; ++i) {
    const Track& t = tracks_.tracks[i];
    // REJECTED tracks are terminal/inactive but still counted; free slots
    // (inactive + kTrackActive + id -1) match no branch.
    if (t.status == kTrackVerified && t.active) ++nVer;
    else if (t.status == kTrackRejected) ++nRej;
    else if (t.status == kTrackActive && t.active) ++nAct;
  }
  snap_.verified = nVer;
  snap_.rejected = nRej;
  snap_.activeTracks = nAct;
  for (int k = 0; k < kNumTaxa; ++k) snap_.tally[k] = slide_.tally[k];
  // Live candidate overlay for the web minimap / findings panel.
  snap_.numCands = (nDet < 8) ? nDet : 8;
  for (int i = 0; i < 8; ++i) snap_.cands[i] = UiCandidate{};
  for (int i = 0; i < snap_.numCands; ++i) {
    UiCandidate& c = snap_.cands[i];
    c.x0 = dets[i].x0;
    c.y0 = dets[i].y0;
    c.x1 = dets[i].x1;
    c.y1 = dets[i].y1;
    c.cls = dets[i].cls;
    c.conf = dets[i].conf;
    c.trig = detTrig[i];
  }

  ++fieldIndex_;
  return true;
}

int PanParasitePipeline::verifiedTotal() const noexcept {
  int s = 0;
  for (int k = 0; k < kNumTaxa; ++k) s += slide_.tally[k];
  return s;
}

void PanParasitePipeline::slideTally(int out[7]) const noexcept {
  if (out == nullptr) return;
  for (int k = 0; k < kNumTaxa; ++k) out[k] = slide_.tally[k];
}

}  // namespace panparasite
