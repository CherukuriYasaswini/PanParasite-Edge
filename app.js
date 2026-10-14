'use strict';
/* MICROPARA frontend (Revision 8.2 blueprint).
   No dependencies. Works served by the C++ server (live /api bridge) or
   opened directly (rich local demo state). */

const $ = id => document.getElementById(id);
const AMBER = '#D97706';

const TAXA = ['Ascaris lumbricoides', 'Trichuris trichiura', 'Hookworm',
  'Schistosoma mansoni', 'Schistosoma haematobium',
  'Non-Biological Matrix Debris', 'Unknown / Other Helminth'];
const TAXA_SHORT = ['Ascaris', 'Trichuris', 'Hookworm', 'S. mansoni',
  'S. haematobium', 'Debris', 'Unknown'];
// Manual-override dropdown per blueprint (6 entries; Other Helminth -> class 6).
const OVERRIDE_OPTS = ['Ascaris lumbricoides', 'Trichuris trichiura', 'Hookworm',
  'Schistosoma mansoni', 'Schistosoma haematobium', 'Other Helminth'];

const I18N = {
  en: {nav_analyze:'ANALYZE',nav_results:'RESULTS',nav_history:'HISTORY',nav_device:'DEVICE',nav_settings:'SETTINGS',
       start_title:'Screening Workspace',start_btn:'START NEW SCREENING',resume_title:'Resumable sessions'},
  te: {nav_analyze:'విశ్లేషణ',nav_results:'ఫలితాలు',nav_history:'చరిత్ర',nav_device:'పరికరం',nav_settings:'అమరికలు',
       start_title:'స్క్రీనింగ్ వర్క్‌స్పేస్',start_btn:'కొత్త స్క్రీనింగ్ ప్రారంభించు',resume_title:'పునఃప్రారంభ సెషన్లు'},
  hi: {nav_analyze:'विश्लेषण',nav_results:'परिणाम',nav_history:'इतिहास',nav_device:'डिवाइस',nav_settings:'सेटिंग्स',
       start_title:'स्क्रीनिंग कार्यक्षेत्र',start_btn:'नई स्क्रीनिंग शुरू करें',resume_title:'फिर से शुरू करने योग्य सत्र'}
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------- state ---------------- */
const S = {
  view: 'analyze', ws: 1,
  caseId: 'CASE #24061', sampleId: 'Sample 01',
  sampleType: 'Fecal Kato-Katz', notes: '',
  deviceId: 'MICROPARA-PI5-01',
  candidates: [], selected: -1,
  dismissed: {}, decisions: {}, overrides: {},
  density: 'normal', zoom: 1, pan: {x: 0, y: 0},
  live: false, liveCap: false, finalized: false,
  fieldName: 'Field A1', lang: 'en', seq: 24061
};

function store(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }
function load(k) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null; } catch (e) { return null; } }

/* ---------------- procedural specimen field (1920x1080 space) ---------------- */
function genField(density, seed) {
  const rnd = mulberry32(seed);
  const out = [];
  let shapes;
  if (density === 'high') {
    shapes = [];
    for (let i = 0; i < 12; i++) shapes.push({t: i % 6 === 5 ? 'ring' : 'egg', cls: i % 6});
  } else if (density === 'occluded') {
    shapes = [{t:'egg',cls:0},{t:'lat',cls:3},{t:'term',cls:4},{t:'egg',cls:2},{t:'ring',cls:5}];
  } else {
    shapes = [{t:'egg',cls:0},{t:'lat',cls:3},{t:'ring',cls:5}];
  }
  shapes.forEach((sh, i) => {
    const w = 120 + Math.floor(rnd() * 60), h = 80 + Math.floor(rnd() * 40);
    const x = 150 + Math.floor(rnd() * (1920 - 300 - w));
    const y = 120 + Math.floor(rnd() * (1080 - 240 - h));
    const conf = sh.t === 'ring' ? 0.31 + rnd() * 0.2
      : (density === 'occluded' ? 0.55 + rnd() * 0.2 : 0.68 + rnd() * 0.27);
    const trig = density === 'occluded' && i < 2;
    const b1 = sh.t === 'ring' ? 1 : 0;
    const rh = sh.t === 'ring' ? 0.55 + rnd() * 0.1 : rnd() * 0.04;
    const th = sh.t === 'lat' ? 68 + rnd() * 18 : (sh.t === 'term' ? 6 + rnd() * 10 : rnd() * 8);
    out.push({id: i, x0: x, y0: y, x1: x + w, y1: y + h, cls: sh.cls, conf: +conf.toFixed(2),
      trig: trig, shape: sh.t,
      fbsc: {b0: 1, b1: b1, rhole: +rh.toFixed(3), theta: +th.toFixed(1),
             o: trig ? +(0.66 + rnd() * 0.1).toFixed(2) : +(0.05 + rnd() * 0.2).toFixed(2),
             eng: 'INT8'}});
  });
  return out;
}

/* ---------------- canvas rendering ---------------- */
const FW = 1920, FH = 1080;
function viewScale() {
  const c = $('specCanvas');
  return (c.width / FW) * S.zoom;
}
function clampPan() {
  const mx = S.zoom > 1 ? (FW - FW / S.zoom) / 2 : 0;
  const my = S.zoom > 1 ? (FH - FH / S.zoom) / 2 : 0;
  S.pan.x = Math.max(-mx, Math.min(mx, S.pan.x));
  S.pan.y = Math.max(-my, Math.min(my, S.pan.y));
}
function drawField() {
  const c = $('specCanvas'), g = c.getContext('2d');
  const W = c.width, H = c.height;
  const rnd = mulberry32(1234);
  const bg = g.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#25313D'); bg.addColorStop(1, '#131A22');
  g.fillStyle = bg; g.fillRect(0, 0, W, H);
  g.fillStyle = 'rgba(255,255,255,0.05)';
  for (let i = 0; i < 420; i++) {
    g.fillRect(rnd() * W, rnd() * H, 1.6, 1.6);
  }
  g.strokeStyle = 'rgba(255,255,255,0.06)';
  for (let i = 0; i < 5; i++) {
    g.beginPath();
    const y = rnd() * H;
    g.moveTo(0, y); g.lineTo(W, y + (rnd() - 0.5) * 60); g.stroke();
  }
  const s = viewScale();
  const ox = W / 2 - (FW / 2 + S.pan.x) * s;
  const oy = H / 2 - (FH / 2 + S.pan.y) * s;
  const X = x => ox + x * s, Y = y => oy + y * s;
  S.candidates.forEach(cd => {
    if (S.dismissed[cd.id]) return;
    const cx = (cd.x0 + cd.x1) / 2, cy = (cd.y0 + cd.y1) / 2;
    const rx = (cd.x1 - cd.x0) / 2, ry = (cd.y1 - cd.y0) / 2;
    g.fillStyle = cd.shape === 'ring' ? 'rgba(160,150,135,0.5)' : 'rgba(205,182,132,0.85)';
    g.beginPath(); g.ellipse(X(cx), Y(cy), rx * s, ry * s, 0, 0, 7); g.fill();
    g.strokeStyle = '#6B5436'; g.lineWidth = 2; g.stroke();
    if (cd.shape === 'ring') {
      g.fillStyle = 'rgba(37,49,61,0.9)';
      g.beginPath(); g.ellipse(X(cx), Y(cy), rx * s * 0.6, ry * s * 0.6, 0, 0, 7); g.fill();
    }
    if (cd.shape === 'lat' || cd.shape === 'term') {
      g.fillStyle = 'rgba(205,182,132,0.9)';
      g.beginPath();
      if (cd.shape === 'lat') {
        g.moveTo(X(cx - 10), Y(cy - ry)); g.lineTo(X(cx + 10), Y(cy - ry));
        g.lineTo(X(cx), Y(cy - ry - 34));
      } else {
        g.moveTo(X(cx + rx), Y(cy - 12)); g.lineTo(X(cx + rx), Y(cy + 12));
        g.lineTo(X(cx + rx + 40), Y(cy));
      }
      g.closePath(); g.fill(); g.stroke();
    }
  });
  // vignette + scale bar (100 µm = 161 px at 0.62 µm/px)
  const vg = g.createRadialGradient(W/2, H/2, H*0.35, W/2, H/2, H*0.85);
  vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,0.45)');
  g.fillStyle = vg; g.fillRect(0, 0, W, H);
  g.fillStyle = '#E2E8F0'; g.font = '12px sans-serif';
  const sb = 161 * s;
  g.fillRect(W - 24 - sb, H - 22, sb, 4);
  g.fillText('100 µm', W - 24 - sb, H - 28);
  // candidate boxes
  S.candidates.forEach(cd => {
    if (S.dismissed[cd.id]) return;
    const hi = cd.conf >= 0.70;
    const sel = cd.id === S.selected;
    g.strokeStyle = sel ? '#1D4ED8' : (hi ? '#15803D' : AMBER);
    g.lineWidth = sel ? 3 : 2;
    g.strokeRect(X(cd.x0), Y(cd.y0), (cd.x1 - cd.x0) * s, (cd.y1 - cd.y0) * s);
    g.fillStyle = sel ? '#1D4ED8' : (hi ? '#15803D' : AMBER);
    const lbl = TAXA_SHORT[cd.cls] + ' ' + cd.conf.toFixed(2);
    g.font = 'bold 12px sans-serif';
    const tw = g.measureText(lbl).width + 12;
    g.fillRect(X(cd.x0), Y(cd.y0) - 20, tw, 18);
    g.fillStyle = '#fff';
    g.fillText(lbl, X(cd.x0) + 6, Y(cd.y0) - 6);
  });
}

function drawMinimap() {
  const m = $('minimap'), g = m.getContext('2d');
  g.clearRect(0, 0, m.width, m.height);
  g.fillStyle = '#F8FAFC'; g.fillRect(0, 0, m.width, m.height);
  g.strokeStyle = '#CBD5E1';
  for (let i = 0; i <= 6; i++) { g.beginPath(); g.moveTo(i * 30, 0); g.lineTo(i * 30, 120); g.stroke(); }
  for (let j = 0; j <= 4; j++) { g.beginPath(); g.moveTo(0, j * 30); g.lineTo(180, j * 30); g.stroke(); }
  g.strokeStyle = '#1D4ED8'; g.lineWidth = 2;
  g.strokeRect(4, 4, 172 / 3, 112 / 2);
  S.candidates.forEach(cd => {
    if (S.dismissed[cd.id] || cd.conf >= 0.70) return;
    const ax = ((cd.x0 + cd.x1) / 2 / FW) * 172 / 3 + 4;
    const ay = ((cd.y0 + cd.y1) / 2 / FH) * 112 / 2 + 4;
    g.fillStyle = AMBER;
    g.beginPath(); g.arc(ax, ay, 4.5, 0, 7); g.fill();
    cd._mx = ax; cd._my = ay;
  });
}

/* ---------------- findings panel + 4-layer evidence ---------------- */
function taxonOf(cd) {
  const ov = S.overrides[cd.id];
  if (ov !== undefined) return ov;
  return TAXA[cd.cls];
}
function whyText(cd) {
  if (cd.shape === 'ring') return 'Why am I seeing this? A closed rim with a large hollow interior and ragged hull ratios resembles matrix debris more than any egg taxon.';
  if (cd.shape === 'lat') return 'Why am I seeing this? A lateral spine-like protrusion with a high folded angle matches the S. mansoni morphotype.';
  if (cd.shape === 'term') return 'Why am I seeing this? A terminal spine-like protrusion aligned with the major axis matches the S. haematobium morphotype.';
  if (cd.fbsc && cd.fbsc.b1 === 2) return 'Why am I seeing this? Two fully enclosed vacuoles (Beta_1 = 2) with a solid hull match the Trichuris morphotype.';
  return 'Why am I seeing this? A smooth convex shell with no enclosed holes and stable moments match a plain egg morphotype.';
}
function renderFindings() {
  const list = $('findingsList');
  list.innerHTML = '';
  const vis = S.candidates.filter(cd => !S.dismissed[cd.id]);
  $('findingsCount').textContent = S.live && S.liveCap
    ? '(' + vis.length + ', device cap: 8)' : '(' + vis.length + ')';
  $('densityBanner').classList.toggle('hidden', !(S.live ? S.liveCap : S.density === 'high'));
  vis.forEach(cd => {
    const hi = cd.conf >= 0.70;
    const card = document.createElement('div');
    card.className = 'cand-card' + (cd.id === S.selected ? ' selected' : '');
    const ov = S.overrides[cd.id];
    const title = (ov !== undefined ? ov : TAXA[cd.cls]);
    const head = hi ? 'High confidence' : 'Review recommended';
    card.innerHTML =
      '<h4>' + (hi ? '' : 'Possible ') + title + '</h4>' +
      '<span class="pill ' + (hi ? 'high' : 'review') + '">' +
        (hi ? 'High confidence' : 'Review recommended') + ' · ' + cd.conf.toFixed(2) + '</span>' +
      '<div class="micro-actions">' +
        '<button type="button" class="confirm-btn' + (S.decisions[cd.id] === 'confirmed' ? ' on' : '') + '">Confirm Finding</button>' +
        '<button type="button" class="review-btn' + (S.decisions[cd.id] === 'review' ? ' on' : '') + '">Needs Review</button>' +
        '<button type="button" class="dismiss-btn" title="Dismiss as Artifact">🗑 Dismiss as Artifact</button>' +
      '</div><div class="evslot"></div>';
    card.querySelector('.confirm-btn').addEventListener('click', ev => {
      ev.stopPropagation();
      if (S.decisions[cd.id] === 'confirmed') delete S.decisions[cd.id];
      else S.decisions[cd.id] = 'confirmed';
      postReview('CONFIRM', cd.id); renderAll();
    });
    card.querySelector('.review-btn').addEventListener('click', ev => {
      ev.stopPropagation();
      if (S.decisions[cd.id] === 'review') delete S.decisions[cd.id];
      else S.decisions[cd.id] = 'review';
      renderAll();
    });
    card.querySelector('.dismiss-btn').addEventListener('click', ev => {
      ev.stopPropagation();
      S.dismissed[cd.id] = true;
      if (S.selected === cd.id) S.selected = -1;
      postReview('REJECT', cd.id); renderAll();
    });
    card.addEventListener('click', () => {
      S.selected = cd.id;
      if (S.ws === 3) S.ws = 4;
      renderAll();
    });
    if (cd.id === S.selected) renderEvidence(card.querySelector('.evslot'), cd);
    list.appendChild(card);
  });
  const sel = S.candidates.find(c => c.id === S.selected);
  $('contrastBadge').classList.toggle('hidden', !(sel && sel.trig));
}

function renderEvidence(slot, cd) {
  const w = Math.round((cd.x1 - cd.x0) * 0.62), h = Math.round((cd.y1 - cd.y0) * 0.62);
  const div = document.createElement('div');
  div.className = 'evidence';
  div.innerHTML =
    '<p><b>Layer 1 — ' + (cd.conf >= 0.70 ? 'High confidence' : 'Review recommended') + '</b><br>' + whyText(cd) + '</p>' +
    '<p><b>Layer 2 — Visual evidence</b><br><canvas class="roi" width="220" height="140"></canvas><br>' +
    '<label><input type="checkbox" class="contourT"> Contour / spine highlight</label></p>' +
    '<p><b>Layer 3 — Supporting info</b></p>' +
    '<div class="kv"><span>Calibrated confidence</span><b>' + cd.conf.toFixed(2) + '</b></div>' +
    '<div class="kv"><span>Dimensions</span><b>' + w + ' × ' + h + ' µm</b></div>' +
    '<div class="kv"><span>Image quality</span><b>' + (cd.trig ? 'Enhanced' : 'Good') + '</b></div>' +
    '<select class="taxon-override">' + OVERRIDE_OPTS.map(o =>
      '<option' + ((S.overrides[cd.id] || TAXA[cd.cls]) === o ? ' selected' : '') + '>' + o + '</option>').join('') + '</select>' +
    '<p><b>Layer 4 — </b><button type="button" class="techlink">Technical Details</button></p>';
  slot.appendChild(div);
  drawRoi(div.querySelector('.roi'), cd, false);
  div.querySelector('.contourT').addEventListener('change', ev => {
    drawRoi(div.querySelector('.roi'), cd, ev.target.checked);
  });
  div.querySelector('.taxon-override').addEventListener('change', ev => {
    S.overrides[cd.id] = ev.target.value;
    renderAll();
  });
  div.querySelector('.techlink').addEventListener('click', ev => {
    ev.stopPropagation();
    openTech(cd);
  });
}

function drawRoi(cv, cd, contour) {
  const g = cv.getContext('2d');
  const src = $('specCanvas');
  const s = viewScale();
  const ox = src.width / 2 - (FW / 2 + S.pan.x) * s;
  const oy = src.height / 2 - (FH / 2 + S.pan.y) * s;
  try {
    g.drawImage(src, ox + cd.x0 * s, oy + cd.y0 * s,
      (cd.x1 - cd.x0) * s, (cd.y1 - cd.y0) * s, 0, 0, cv.width, cv.height);
  } catch (e) {
    g.fillStyle = '#334155'; g.fillRect(0, 0, cv.width, cv.height);
  }
  if (contour) {
    g.strokeStyle = '#FDE725'; g.lineWidth = 2;
    g.beginPath();
    g.ellipse(cv.width / 2, cv.height / 2, cv.width * 0.38, cv.height * 0.32, 0, 0, 7);
    g.stroke();
    if (cd.shape === 'lat' || cd.shape === 'term') {
      g.fillStyle = '#FDE725';
      g.beginPath(); g.arc(cv.width / 2, 14, 5, 0, 7); g.fill();
    }
  }
}

function openTech(cd) {
  const f = cd.fbsc || {b0: 1, b1: 0, rhole: 0, theta: 0, o: 0, eng: 'INT8'};
  const rows = [
    ['Beta_0', 1], ['Beta_1', f.b1], ['R_hole', f.rhole],
    ['theta_folded', (f.theta || 0) + '°'], ['O_score', f.o],
    ['engine', f.eng + ' / FP32 fallback ready'],
    ['bbox px', cd.x0 + ',' + cd.y0 + ' – ' + cd.x1 + ',' + cd.y1]
  ];
  let h = '<table>';
  rows.forEach(r => { h += '<tr><td>' + r[0] + '</td><td><b>' + r[1] + '</b></td></tr>'; });
  h += '<tr><td>f_BSC[16]</td><td>[1, ' + f.b1 + ', ' + f.rhole + ', …]</td></tr></table>' +
    '<p class="muted">Full 16-D Betti signature vector retained in the device audit ledger.</p>';
  $('techBody').innerHTML = h;
  $('techSheet').classList.remove('hidden');
}

/* ---------------- workflow / views ---------------- */
const STEPS = ['SAMPLE', 'IMAGE', 'ANALYSIS', 'FINDINGS', 'REVIEW', 'RESULT'];
function renderWorkflow() {
  let lit = [];
  if (S.ws === 2) lit = [0];
  else if (S.ws === 3) lit = S.selected >= 0 ? [1, 2, 3] : [1, 2];
  else if (S.ws === 4) lit = [3, 4];
  else if (S.ws >= 5) lit = [5];
  const decided = Object.keys(S.decisions).length > 0 ||
    Object.keys(S.dismissed).length > 0 || Object.keys(S.overrides).length > 0;
  if ((S.ws === 3 || S.ws === 4) && decided && lit.indexOf(4) < 0) lit.push(4);
  document.querySelectorAll('.wf-step').forEach(el => {
    const i = +el.dataset.s;
    el.classList.toggle('lit', lit.indexOf(i) >= 0);
    el.classList.toggle('done', lit.length > 0 && i < Math.max.apply(null, lit));
  });
}
function showView(v) {
  S.view = v;
  ['analyze', 'results', 'history', 'device', 'settings'].forEach(k => {
    $('view-' + k).classList.toggle('hidden', k !== v);
  });
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.view === v);
  });
  $('hdrSection').textContent = v.toUpperCase();
  if (v === 'results') renderResults();
  if (v === 'history') renderHistory();
}
function showState(n) {
  S.ws = n;
  ['st1', 'st2', 'st3', 'st5', 'st6'].forEach(id => $(id).classList.add('hidden'));
  $('st' + (n === 4 ? 3 : n)).classList.remove('hidden');
  renderWorkflow();
}

/* ---------------- state flows ---------------- */
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 2200);
}

function startScreening(resume) {
  if (!resume) {
    if (S.finalized) {
      // A finished case stays in history; a fresh screening gets a new ID.
      S.seq += 1;
      S.caseId = 'CASE #' + S.seq;
    }
    S.sampleId = 'Sample 01'; S.sampleType = $('setSampleType').value;
    S.notes = ''; S.candidates = []; S.selected = -1;
    S.dismissed = {}; S.decisions = {}; S.overrides = {};
    S.finalized = false;
    $('verifyCheck').checked = false;
    $('btnFinalize').disabled = true;
    $('completeView').classList.add('hidden');
  }
  showView('analyze');
  showState(2);
  refreshSetup();
}
function refreshSetup() {
  $('caseIdAuto').value = S.caseId.replace('CASE #', '');
  $('caseDateAuto').value = new Date().toLocaleString();
  $('deviceIdAuto').value = S.deviceId;
  $('sampleId').value = S.sampleId;
  $('sampleType').value = S.sampleType;
  $('sampleNotes').value = S.notes;
  $('caseLabel').textContent = S.caseId + ' | ' + S.sampleId;
  $('verifyCaseId').textContent = 'Case ID #' + S.caseId.replace('CASE #', '');
}
function beginAnalysis() {
  S.sampleId = $('sampleId').value || 'Sample 01';
  S.sampleType = $('sampleType').value;
  S.notes = $('sampleNotes').value;
  S.candidates = genField(S.density, 4242);
  S.selected = -1; S.dismissed = {}; S.decisions = {}; S.overrides = {};
  S.zoom = 1; S.pan = {x: 0, y: 0};
  $('caseLabel').textContent = S.caseId + ' | ' + S.sampleId;
  showState(3);
  renderAll();
  toast('Field A1 analyzed: ' + S.candidates.length + ' candidates');
}

function resultStats() {
  const bySpecies = {};
  let confirmed = 0, dismissed = 0;
  S.candidates.forEach(cd => {
    if (S.dismissed[cd.id]) { dismissed++; return; }
    if (S.decisions[cd.id] === 'confirmed') {
      confirmed++;
      const t = S.overrides[cd.id] || TAXA[cd.cls];
      bySpecies[t] = (bySpecies[t] || 0) + 1;
    }
  });
  return {bySpecies, confirmed, dismissed};
}
function showResult() {
  showState(5);
  const st = resultStats();
  const keys = Object.keys(st.bySpecies);
  $('resultState').textContent = keys.length
    ? 'Review complete: ' + st.confirmed + ' confirmed finding(s).'
    : 'No confirmed findings. ' + (S.candidates.length ? 'All targets need review or were dismissed.' : 'No targets detected in this field.');
  $('resultSpecies').innerHTML = keys.length
    ? keys.map(k => '<p>' + k + ': <b>' + st.bySpecies[k] + '</b></p>').join('')
    : '<p class="muted">None confirmed.</p>';
  $('dismissedCount').textContent = st.dismissed;
  $('imageQuality').textContent = S.density === 'occluded' ? 'Enhanced (contrast interpolation)' : 'Good';
}
function buildReport() {
  const st = resultStats();
  const lines = Object.keys(st.bySpecies).map(k => k + ': ' + st.bySpecies[k]);
  $('reportPreview').innerHTML =
    '<p><b>' + S.caseId + ' | ' + S.sampleId + ' · ' + S.sampleType + '</b></p>' +
    '<p>Confirmed: ' + (lines.join('; ') || 'none') +
    ' · Dismissed artifacts: ' + st.dismissed + '</p>';
  $('sysInfo').innerHTML =
    '<p>Device ' + S.deviceId + ' · gate τ 0.38 · Platt T 1.38 · INT8/FP32 · O threshold 0.65</p>';
  showState(6);
}
function finalizeReport() {
  const st = resultStats();
  const flagged = S.candidates.some(cd => !S.dismissed[cd.id] && cd.conf < 0.70);
  const rec = {caseId: S.caseId, sample: S.sampleId, type: S.sampleType,
    ts: new Date().toISOString(), status: 'Complete',
    review: flagged ? 'Flagged' : 'Clean',
    findings: st.confirmed + ' confirmed / ' + st.dismissed + ' dismissed',
    notes: $('techNotes').value};
  const reps = load('mp_reports') || [];
  reps.push(rec);
  store('mp_reports', reps);
  const hist = load('mp_history') || [];
  hist.unshift(rec);
  store('mp_history', hist);
  S.finalized = true;
  $('completeView').classList.remove('hidden');
  toast('Report saved for ' + S.caseId);
}

function suspendSession() {
  const total = S.candidates.length;
  const rev = Object.keys(S.decisions).length;
  const list = load('mp_suspended') || [];
  list.unshift({caseId: S.caseId, sampleId: S.sampleId, ts: new Date().toISOString(),
    progress: rev + ' of ' + total + ' findings reviewed',
    snap: {ws: S.ws, density: S.density, candidates: S.candidates,
           decisions: S.decisions, dismissed: S.dismissed, overrides: S.overrides,
           sampleType: S.sampleType, notes: S.notes}});
  store('mp_suspended', list.slice(0, 12));
  renderResume();
  toast('Session suspended — resume anytime from quick-launch cards.');
}
function renderResume() {
  const list = load('mp_suspended') || [];
  const box = $('resumeCards');
  box.innerHTML = '';
  if (!list.length) {
    box.innerHTML = '<p class="muted">No suspended sessions.</p>';
    return;
  }
  list.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'resume-card';
    d.innerHTML = '<div class="rc-row"><div><b>' + r.caseId + '</b> · ' + r.sampleId +
      '<br><span class="muted">' + r.ts + ' · ' + r.progress + '</span></div>' +
      '<button type="button">Resume Session -&gt;</button></div>';
    d.querySelector('button').addEventListener('click', () => {
      const sn = r.snap || {};
      S.ws = sn.ws || 3; S.density = sn.density || 'normal';
      S.candidates = sn.candidates || []; S.decisions = sn.decisions || {};
      S.dismissed = sn.dismissed || {}; S.overrides = sn.overrides || {};
      S.sampleType = sn.sampleType || S.sampleType; S.notes = sn.notes || '';
      S.selected = -1; S.caseId = r.caseId; S.sampleId = r.sampleId;
      const rest = (load('mp_suspended') || []).filter((_, k) => k !== i);
      store('mp_suspended', rest);
      refreshSetup(); showView('analyze');
      // Terminal result/report screens are not resumable targets: land back
      // in the workspace with the restored candidates.
      showState(S.ws >= 5 ? 3 : (S.ws === 1 ? 2 : S.ws));
      renderAll(); renderResume();
      toast('Resumed ' + r.caseId);
    });
    box.appendChild(d);
  });
}

/* ---------------- results / history / device / settings ---------------- */
function renderResults() {
  const reps = load('mp_reports') || [];
  $('resultsList').innerHTML = reps.length ? reps.map(r =>
    '<div class="calm-box"><b>' + r.caseId + '</b> · ' + r.sample + ' · ' + r.ts +
    '<br>' + r.findings + ' · ' + r.status + ' / ' + r.review + '</div>').join('')
    : '<p class="muted">No completed reports yet.</p>';
}
function seedHistory() {
  if (load('mp_history')) return;
  store('mp_history', [
    {caseId: 'CASE #24058', sample: 'Sample 01', ts: '2026-10-11T09:14:00', status: 'Complete', review: 'Clean', findings: '2 confirmed / 0 dismissed'},
    {caseId: 'CASE #24059', sample: 'Urine 02', ts: '2026-10-12T11:02:00', status: 'Complete', review: 'Flagged', findings: '1 confirmed / 1 dismissed'},
    {caseId: 'CASE #24060', sample: 'Sample 04', ts: '2026-10-13T08:47:00', status: 'In progress', review: 'Flagged', findings: '0 confirmed / 0 dismissed'}
  ]);
}
function renderHistory() {
  const q = ($('historySearch').value || '').toLowerCase();
  const d = $('historyDate').value, st = $('historyStatus').value, rv = $('historyReview').value;
  const rows = (load('mp_history') || []).filter(r =>
    (!q || (r.caseId + ' ' + r.sample).toLowerCase().indexOf(q) >= 0) &&
    (!d || (r.ts || '').indexOf(d) === 0) &&
    (!st || r.status === st) && (!rv || r.review === rv));
  $('historyBody').innerHTML = rows.map(r =>
    '<tr><td>' + r.caseId + '</td><td>' + r.sample + '</td><td>' + (r.ts || '').slice(0, 10) +
    '</td><td>' + r.status + '</td><td>' + r.review + '</td><td>' + r.findings + '</td></tr>').join('') ||
    '<tr><td colspan="6" class="muted">No matching records.</td></tr>';
}
function applyLang(code) {
  S.lang = code;
  const dict = I18N[code] || I18N.en;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const v = dict[el.dataset.i18n];
    if (v) el.textContent = v;
  });
  store('mp_settings', {lang: code});
}

/* ---------------- backend bridge ---------------- */
function liveMode() {
  try { return location.protocol.indexOf('http') === 0; } catch (e) { return false; }
}
function postReview(action, id) {
  if (!S.live) return;
  try {
    fetch('/api/review', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: action, id: id})}).catch(() => {});
  } catch (e) {}
}
function refreshLive() {
  if (!S.live) return;
  fetch('/api/status').then(r => r.json()).then(d => {
    $('liveFlag').textContent = 'LIVE DEVICE LINK';
    S.liveCap = !!d.cap;
    if (d.ringDirectIo !== undefined) {
      $('ringStatus').textContent = '256 sectors × 4096 B · O_DIRECT ' +
        (d.ringDirectIo ? 'active' : 'fallback (buffered)');
    }
    if (Array.isArray(d.cands) && d.cands.length) {
      // Preserve an open evidence panel: live refresh swaps candidates only
      // when nothing is selected.
      if (S.selected < 0) {
        S.candidates = d.cands.map((c, i) => ({id: i,
        x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1, cls: c.cls < 0 ? 6 : c.cls,
        conf: c.conf || 0, trig: !!c.trig, shape: 'egg',
        fbsc: {b0: 1, b1: 0, rhole: 0, theta: 0, o: c.trig ? 0.7 : 0.1, eng: 'INT8'}}));
      }
      if (S.ws === 3 || S.ws === 4) renderAll();
    }
  }).catch(() => {});
}

/* ---------------- master render ---------------- */
function renderAll() {
  drawField();
  drawMinimap();
  renderFindings();
  renderWorkflow();
  $('zoomLabel').textContent = Math.round(S.zoom * 100) + '%';
  document.querySelectorAll('.density-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.density === S.density);
  });
  $('fieldLabel').textContent = S.fieldName + ' · ' + S.density;
}

/* ---------------- init ---------------- */
function bind() {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', () => showView(el.dataset.view));
  });
  document.querySelectorAll('.wf-step').forEach(el => {
    el.addEventListener('click', () => {
      // SAMPLE->setup, IMAGE/ANALYSIS/FINDINGS->workspace, REVIEW->workspace
      // (evidence), RESULT->result screen.
      const target = [2, 3, 3, 3, 4, 5][+el.dataset.s] || 2;
      showView('analyze');
      showState(target);
    });
  });
  $('btnStart').addEventListener('click', () => startScreening(false));
  $('btnSuspend').addEventListener('click', suspendSession);
  $('btnBackSetup').addEventListener('click', () => showState(1));
  $('btnBegin').addEventListener('click', beginAnalysis);
  $('btnToReview').addEventListener('click', showResult);
  $('btnGenerateReport').addEventListener('click', buildReport);
  $('verifyCheck').addEventListener('change', ev => {
    $('btnFinalize').disabled = !ev.target.checked;
  });
  $('btnFinalize').addEventListener('click', finalizeReport);
  $('btnViewReport').addEventListener('click', () => {
    $('reportPreview').scrollIntoView();
    toast('Report ready for print / export.');
  });
  $('btnNewScreening2').addEventListener('click', () => {
    startScreening(false);
    renderResume();
  });
  $('techClose').addEventListener('click', () => $('techSheet').classList.add('hidden'));
  const zoomStep = f => {
    S.zoom = Math.max(0.5, Math.min(4, S.zoom * f));
    clampPan(); renderAll();
  };
  $('zoomIn').addEventListener('click', () => zoomStep(1.25));
  $('zoomOut').addEventListener('click', () => zoomStep(0.8));
  $('zoomLabel').addEventListener('click', () => { S.zoom = 1; S.pan = {x: 0, y: 0}; renderAll(); });
  $('zoomFit').addEventListener('click', () => { S.zoom = 1; S.pan = {x: 0, y: 0}; renderAll(); });
  $('zoomReset').addEventListener('click', () => { S.zoom = 1; S.pan = {x: 0, y: 0}; renderAll(); });
  document.querySelectorAll('.density-btn').forEach(b => {
    b.addEventListener('click', () => {
      S.density = b.dataset.density;
      S.candidates = genField(S.density, S.density === 'high' ? 777 : (S.density === 'occluded' ? 999 : 4242));
      S.selected = -1; S.dismissed = {}; S.decisions = {}; S.overrides = {};
      renderAll();
    });
  });
  $('specCanvas').addEventListener('click', ev => {
    const r = $('specCanvas').getBoundingClientRect();
    const px = (ev.clientX - r.left) * ($('specCanvas').width / r.width);
    const py = (ev.clientY - r.top) * ($('specCanvas').height / r.height);
    const s = viewScale();
    const ox = $('specCanvas').width / 2 - (FW / 2 + S.pan.x) * s;
    const oy = $('specCanvas').height / 2 - (FH / 2 + S.pan.y) * s;
    const fx = ((px - ox) / s), fy = ((py - oy) / s);
    let best = -1, bd = 1e12;
    S.candidates.forEach(cd => {
      if (S.dismissed[cd.id]) return;
      if (fx < cd.x0 - 8 || fx > cd.x1 + 8 || fy < cd.y0 - 8 || fy > cd.y1 + 8) return;
      const cx = (cd.x0 + cd.x1) / 2 - fx, cy = (cd.y0 + cd.y1) / 2 - fy;
      const d = cx * cx + cy * cy;
      if (d < bd) { bd = d; best = cd.id; }
    });
    if (best >= 0) {
      S.selected = best;
      if (S.ws === 3) S.ws = 4;
      renderAll();
    }
  });
  $('minimap').addEventListener('click', ev => {
    const r = $('minimap').getBoundingClientRect();
    const mx = (ev.clientX - r.left) * ($('minimap').width / r.width);
    const my = (ev.clientY - r.top) * ($('minimap').height / r.height);
    let best = null, bd = 1e12;
    S.candidates.forEach(cd => {
      if (S.dismissed[cd.id] || cd.conf >= 0.70 || cd._mx === undefined) return;
      const d = (cd._mx - mx) * (cd._mx - mx) + (cd._my - my) * (cd._my - my);
      if (d < bd) { bd = d; best = cd; }
    });
    if (best && bd < 30 * 30) {
      // Recenter the viewport on the anchor (full-frame coords).
      S.pan.x = (best.x0 + best.x1) / 2 - FW / 2;
      S.pan.y = (best.y0 + best.y1) / 2 - FH / 2;
      S.selected = best.id;
      clampPan(); renderAll();
    }
  });
  ['historySearch', 'historyDate', 'historyStatus', 'historyReview'].forEach(id => {
    $(id).addEventListener('input', renderHistory);
    $(id).addEventListener('change', renderHistory);
  });
  $('diagToggle').addEventListener('click', () => $('diagBody').classList.toggle('hidden'));
  $('setLang').addEventListener('change', ev => applyLang(ev.target.value));
  $('setSampleType').addEventListener('change', ev => { S.sampleType = ev.target.value; });
}

document.addEventListener('DOMContentLoaded', () => {
  bind();
  seedHistory();
  const st = load('mp_settings');
  if (st && st.lang) { $('setLang').value = st.lang; applyLang(st.lang); }
  S.live = liveMode();
  $('liveFlag').textContent = S.live ? 'LIVE DEVICE LINK' : 'LOCAL DEMO STATE';
  refreshSetup();
  renderResume();
  showView('analyze');
  showState(1);
  renderAll();
  if (S.live) {
    refreshLive();
    setInterval(refreshLive, 2500);
  }
});
