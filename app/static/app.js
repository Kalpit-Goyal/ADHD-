/* NeuroScan AI — frontend behavior */

const $ = (id) => document.getElementById(id);

const DEFAULT_STATUS =
  'Demos were excluded from training. Uploads are processed temporarily and are not retained.';

let selectedDemo = null;
let selectedFile = null;
let lastResult = null;
let allDoctors = [];
let currentCity = 'Pune';
let demoAvailability = { adhd: true, control: true };
let busy = false;

/* ── helpers ─────────────────────────────────────────────────────── */

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

function safeUrl(value) {
  try {
    const url = new URL(String(value), window.location.origin);
    if (url.protocol === 'http:' || url.protocol === 'https:') return url.href;
  } catch (_) { /* ignore */ }
  return '#';
}

function telHref(phone) {
  const cleaned = String(phone || '').replace(/[^\d+]/g, '');
  return cleaned ? `tel:${cleaned}` : '';
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : '—';
}

async function fetchJson(url, options = {}, timeoutMs = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (_) { payload = null; }

    if (!response.ok) {
      let detail = payload && payload.detail ? payload.detail : text;
      if (Array.isArray(detail)) {
        detail = detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
      }
      if (detail && typeof detail === 'object') detail = JSON.stringify(detail);
      throw new Error(detail || `Request failed (${response.status})`);
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('The request timed out. Try a shorter recording.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/* ── selection / upload ──────────────────────────────────────────── */

function setSelectionUI(name, meta, hasSelection) {
  $('fileName').textContent = name;
  $('fileMeta').textContent = meta;
  $('filePreview').classList.toggle('has-file', hasSelection);
}

function refreshAnalyzeButton() {
  const btn = $('analyzeBtn');
  btn.disabled = busy || (!selectedDemo && !selectedFile);
}

function selectDemo(key) {
  if (busy || !demoAvailability[key]) return;
  selectedDemo = key;
  selectedFile = null;
  $('fileInput').value = '';
  $('sourceFs').value = '128';
  document.querySelectorAll('.scan-btn').forEach((el) => el.classList.remove('active'));
  $(key === 'adhd' ? 'demoAdhd' : 'demoControl').classList.add('active');

  const name = key === 'adhd'
    ? 'ADHD demo · v10p.npy'
    : 'Control demo · v107.npy';
  const meta = key === 'adhd'
    ? 'Held-out ADHD subject · 19 channels · 128 Hz'
    : 'Held-out control subject · 19 channels · 128 Hz';
  setSelectionUI(name, meta, true);
  $('analyzeStatus').classList.remove('error');
  $('analyzeStatus').textContent = DEFAULT_STATUS;
  $('resultError').hidden = true;
  refreshAnalyzeButton();
}

function processFile(file) {
  if (!file) return;
  const suffix = (file.name.includes('.') ? file.name.split('.').pop() : '').toLowerCase();
  const allowed = ['npy', 'csv', 'txt', 'mat'];

  if (!allowed.includes(suffix)) {
    $('analyzeStatus').classList.add('error');
    $('analyzeStatus').textContent = 'Unsupported file type. Use .npy, .csv, .txt, or .mat.';
    return;
  }
  if (file.size > 80 * 1024 * 1024) {
    $('analyzeStatus').classList.add('error');
    $('analyzeStatus').textContent = 'File is larger than 80 MB.';
    return;
  }

  selectedFile = file;
  selectedDemo = null;
  document.querySelectorAll('.scan-btn').forEach((el) => el.classList.remove('active'));
  setSelectionUI(file.name, `${suffix.toUpperCase()} · ${formatBytes(file.size)}`, true);
  $('analyzeStatus').classList.remove('error');
  $('analyzeStatus').textContent = DEFAULT_STATUS;
  $('resultError').hidden = true;
  refreshAnalyzeButton();
}

function handleFile(event) {
  processFile(event.target.files && event.target.files[0]);
}

function clearSelection() {
  if (busy) return;
  selectedDemo = null;
  selectedFile = null;
  $('fileInput').value = '';
  document.querySelectorAll('.scan-btn').forEach((el) => el.classList.remove('active'));
  setSelectionUI('No recording selected', 'Choose a demo or upload a file', false);
  $('analyzeStatus').classList.remove('error');
  $('analyzeStatus').textContent = DEFAULT_STATUS;
  refreshAnalyzeButton();
}

function setBusy(value) {
  busy = value;
  const btn = $('analyzeBtn');
  btn.classList.toggle('busy', value);
  $('analyzeBtnText').textContent = value ? 'Screening…' : 'Run Screening';
  $('fileInput').disabled = value;
  $('demoAdhd').disabled = value || !demoAvailability.adhd;
  $('demoControl').disabled = value || !demoAvailability.control;
  refreshAnalyzeButton();
}

/* ── analysis ────────────────────────────────────────────────────── */

async function runAnalysis() {
  if (busy || (!selectedDemo && !selectedFile)) return;

  const isDemo = Boolean(selectedDemo);
  setBusy(true);
  $('resultError').hidden = true;
  $('analyzeStatus').classList.remove('error');
  $('analyzeStatus').textContent = isDemo
    ? 'Loading the held-out EEG recording…'
    : 'Uploading your EEG recording…';

  const slowHint = setTimeout(() => {
    if (busy) $('analyzeStatus').textContent =
      'Extracting spectral + connectivity features and scoring epochs… this can take up to a minute.';
  }, 700);

  try {
    let data;
    if (isDemo) {
      data = await fetchJson(`/api/demo/${selectedDemo}`, {}, 180000);
    } else {
      const form = new FormData();
      form.append('file', selectedFile);
      form.append('source_fs', $('sourceFs').value || '128');
      data = await fetchJson('/api/analyze', { method: 'POST', body: form }, 240000);
    }

    if (!data) throw new Error('The server returned an empty response.');

    lastResult = data;
    renderResult(data);
    $('analyzeStatus').textContent =
      `Screened ${data.n_epochs} of ${data.n_epochs_total} epochs from ${data.duration_seconds}s of EEG.`;

    if (window.innerWidth < 1020) {
      $('resultsContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    const city = (currentCity || 'Pune').trim();
    await loadDoctors({ city });
    renderReferralPreview(data, allDoctors);
  } catch (error) {
    showResultError(error.message || 'Analysis failed.');
    $('analyzeStatus').classList.add('error');
    $('analyzeStatus').textContent = error.message || 'Analysis failed.';
  } finally {
    clearTimeout(slowHint);
    setBusy(false);
  }
}

function showResultError(message) {
  const box = $('resultError');
  box.textContent = `⚠ ${message}`;
  box.hidden = false;
}

function renderResult(data) {
  const positive = Boolean(data.positive);
  const score = Math.max(0, Math.min(100, Number(data.screening_score) || 0));
  const probability = Math.max(0, Math.min(1, Number(data.probability) || 0));
  const control = Math.max(0, Math.min(1, Number(data.control_probability) || 0));

  $('resultsEmpty').style.display = 'none';
  $('resultsContent').classList.add('visible');
  $('resultError').hidden = true;

  $('resultType').textContent = 'EEG · ADHD SCREENING';
  $('resultLabel').textContent = data.label;
  $('resultLabel').className = `verdict-label ${positive ? 'positive' : 'negative'}`;
  $('resultScore').textContent = String(score);

  const ring = $('scoreRing');
  ring.style.setProperty('--score', String(score));
  ring.style.setProperty('--ring-color', positive ? 'var(--amber)' : 'var(--green)');

  $('resultText').textContent = positive
    ? 'This EEG shows a pattern that resembles the ADHD group more than the control group. It is a screening signal, not a diagnosis — the next step is an in-person clinical assessment.'
    : 'This EEG did not show a strong ADHD-like pattern. EEG screening cannot rule ADHD out, so if attention or behaviour concerns persist, a clinician should still assess them.';

  const source = data.source === 'demo'
    ? `Held-out demo · ${data.demo_label || data.filename}`
    : `Upload · ${data.filename}`;
  $('resultMeta').textContent =
    `${source} · threshold ${data.threshold} · ${data.model}`;

  $('adhdPct').textContent = pct(probability);
  $('controlPct').textContent = pct(control);
  setTimeout(() => {
    $('adhdBar').style.width = `${probability * 100}%`;
    $('controlBar').style.width = `${control * 100}%`;
  }, 80);

  const stats = [
    { value: `${data.n_epochs}`, label: `Epochs kept / ${data.n_epochs_total}` },
    { value: `${data.duration_seconds}s`, label: 'Recording length' },
    { value: `${data.channels}`, label: 'EEG channels' },
    { value: Number(data.theta_beta_ratio).toFixed(2), label: 'Mean θ/β log ratio' },
  ];
  $('resultStats').innerHTML = stats.map((item) => `
    <div class="stat-card">
      <span class="stat-card-value">${esc(item.value)}</span>
      <span class="stat-card-label">${esc(item.label)}</span>
    </div>
  `).join('');

  const bands = data.band_profile || [];
  const maxBand = Math.max(...bands.map((band) => Number(band.pct) || 0), 1);
  $('bandList').innerHTML = bands.map((band) => {
    const value = Number(band.pct) || 0;
    const width = (value / maxBand) * 100;
    const theta = String(band.band).toLowerCase() === 'theta';
    return `
      <div class="band-item">
        <div class="probability-row">
          <span class="band-name">${esc(band.band)}</span>
          <span class="band-pct">${value.toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill band-fill ${theta ? 'theta' : ''}" style="width:${width}%"></div>
        </div>
      </div>
    `;
  }).join('');

  const edges = data.top_edges || [];
  $('edgeList').innerHTML = edges.length
    ? edges.map((edge) => `
        <span class="edge-chip">${esc(edge.pair)} <b>${esc(edge.strength)}</b></span>
      `).join('')
    : '<span class="edge-chip">No connectivity pairs available</span>';

  renderReferralPreview(data, allDoctors);
}

function renderReferralPreview(data, doctors) {
  const box = $('resultReferral');
  if (!data) { box.hidden = true; return; }

  const positive = Boolean(data.positive);
  const top = (doctors || []).slice(0, 3);

  const intro = positive
    ? 'A positive screen should be confirmed by a clinician. These specialists assess ADHD in children and adolescents — call before visiting.'
    : 'A negative screen is not a clean bill of health. If attention, school, or behaviour concerns persist, these clinicians can assess them properly.';

  const docs = top.length ? top.map((doctor) => {
    const phone = telHref(doctor.phone);
    const phoneLabel = doctor.phone_display || doctor.phone || '';
    return `
      <div class="referral-doc">
        <div class="referral-doc-main">
          <div class="referral-doc-name">${esc(doctor.name)}</div>
          <div class="referral-doc-meta">${esc(doctor.clinic || doctor.specialty || 'ADHD assessment')} · ${esc(doctor.address || '')}</div>
        </div>
        ${phone
          ? `<a class="referral-doc-phone" href="${esc(phone)}">${esc(phoneLabel)}</a>`
          : '<span class="referral-doc-meta">Phone not listed</span>'}
      </div>
    `;
  }).join('') : '<div class="referral-doc"><div class="referral-doc-main"><div class="referral-doc-name">Specialist directory unavailable</div><div class="referral-doc-meta">Open the specialists section for the map search.</div></div></div>';

  box.innerHTML = `
    <div class="referral-head">
      <span class="referral-title">Confirm with a specialist</span>
      <span class="referral-title">${positive ? 'RECOMMENDED' : 'IF CONCERNS PERSIST'}</span>
    </div>
    <p class="referral-text">${esc(intro)}</p>
    <div class="referral-docs">${docs}</div>
    <button class="referral-cta" type="button" onclick="scrollToSpecialists()">See all specialists &amp; map →</button>
  `;
  box.hidden = false;
}

function scrollToSpecialists() {
  $('specialists').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── specialists ─────────────────────────────────────────────────── */

function doctorCardHTML(doctor) {
  const phone = telHref(doctor.phone);
  const phoneLabel = doctor.phone_display || doctor.phone || '';
  const distance = doctor.distance_km != null
    ? `<span class="doctor-distance">${Number(doctor.distance_km).toFixed(1)} km away</span>`
    : '';
  const tags = (doctor.tags || []).slice(0, 4)
    .map((tag) => `<span class="tag">${esc(tag)}</span>`).join('');

  const actions = [];
  if (phone) {
    actions.push(`<a class="primary" href="${esc(phone)}">Call ${esc(phoneLabel)}</a>`);
  } else {
    actions.push('<span class="doctor-actions-muted">Phone not listed</span>');
  }
  const mapQuery = doctor.maps_query ||
    [doctor.name, doctor.clinic, doctor.address].filter(Boolean).join(' ');
  actions.push(
    `<button class="show-map-btn" type="button" ` +
    `data-query="${esc(mapQuery)}" data-name="${esc(doctor.name || 'Specialist')}">` +
    `Show on map</button>`,
  );
  if (doctor.website) {
    actions.push(`<a href="${safeUrl(doctor.website)}" target="_blank" rel="noopener">Website ↗</a>`);
  }

  return `
    <article class="doctor-card">
      <div class="doctor-top">
        <div>
          <div class="doctor-name">${esc(doctor.name || 'Specialist')}</div>
          <div class="doctor-specialty">${esc(doctor.specialty || 'ADHD assessment')}</div>
        </div>
        ${distance}
      </div>
      <div class="doctor-clinic">
        ${doctor.clinic ? `<b>${esc(doctor.clinic)}</b> · ` : ''}${esc(doctor.address || doctor.city || '')}
      </div>
      ${tags ? `<div class="doctor-tags">${tags}</div>` : ''}
      ${doctor.note ? `<div class="doctor-note">${esc(doctor.note)}</div>` : ''}
      <div class="doctor-actions">${actions.join('')}</div>
    </article>
  `;
}

function showDoctorOnMap(query, name) {
  const target = String(query || '').trim() ||
    `ADHD child psychiatrist in ${currentCity || 'Pune'}`;
  const encoded = encodeURIComponent(target);

  $('mapEmbed').src =
    `https://maps.google.com/maps?q=${encoded}&z=15&output=embed`;
  $('mapLink').href =
    `https://www.google.com/maps/search/?api=1&query=${encoded}`;
  $('mapCaption').textContent = `Location · ${name || 'Specialist'}`;

  if (window.innerWidth < 1020) {
    $('mapEmbed').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function renderDoctors(data) {
  allDoctors = data.doctors || [];

  const source = data.source || 'maps_search';
  const badge = $('doctorSource');
  badge.textContent = source === 'google_places'
    ? 'LIVE · GOOGLE PLACES'
    : source === 'curated' ? 'CURATED · PUNE' : 'MAP SEARCH';
  badge.classList.toggle('live', source === 'google_places');

  $('doctorNotice').textContent = data.notice || '';

  const mapEmbed = safeUrl(data.map_embed);
  if (mapEmbed !== '#') $('mapEmbed').src = mapEmbed;

  const mapLink = safeUrl(data.map_link);
  if (mapLink !== '#') $('mapLink').href = mapLink;

  const city = data.city || currentCity || 'Pune';
  $('mapCaption').textContent = `ADHD specialists · ${city}`;

  $('doctorList').innerHTML = allDoctors.length
    ? allDoctors.map(doctorCardHTML).join('')
    : `<div class="doctor-empty">
         No curated listings for this area. Use the Google Map to search
         “ADHD child psychiatrist in ${esc(city)}”, and confirm each clinician's
         specialty by phone.<br/><br/>
         To enable live listings anywhere, set a Google Places API key on the server.
       </div>`;

  renderReferralPreview(lastResult, allDoctors);
}

async function loadDoctors({ city = 'Pune', lat = null, lng = null } = {}) {
  $('doctorList').innerHTML = Array.from({ length: 3 })
    .map(() => '<div class="doctor-skeleton"></div>').join('');
  $('doctorSource').textContent = 'LOADING';
  $('doctorSource').classList.remove('live');

  const params = new URLSearchParams();
  if (lat != null && lng != null) {
    params.set('city', '');
    params.set('lat', String(lat));
    params.set('lng', String(lng));
  } else {
    params.set('city', city || 'Pune');
  }

  try {
    const data = await fetchJson(`/api/doctors?${params.toString()}`, {}, 30000);
    if (!data) throw new Error('Empty specialist response.');
    currentCity = data.city || city || 'Pune';
    if (data.city) $('cityInput').value = data.city;
    renderDoctors(data);
    return data;
  } catch (error) {
    $('doctorSource').textContent = 'UNAVAILABLE';
    $('doctorNotice').textContent =
      `Could not load specialist listings: ${error.message}`;
    $('doctorList').innerHTML = `<div class="doctor-empty">
      The specialist service is not reachable. Open the Google Map to search for
      “ADHD child psychiatrist” near you.
    </div>`;
    renderReferralPreview(lastResult, []);
    return null;
  }
}

function searchCity(event) {
  event.preventDefault();
  const city = ($('cityInput').value || '').trim() || 'Pune';
  currentCity = city;
  loadDoctors({ city });
}

function useMyLocation() {
  if (!navigator.geolocation) {
    $('doctorNotice').textContent = 'Geolocation is not available in this browser. Search by city instead.';
    return;
  }

  $('doctorNotice').textContent = 'Requesting your location…';
  navigator.geolocation.getCurrentPosition(
    (position) => {
      $('cityInput').value = '';
      loadDoctors({
        city: '',
        lat: position.coords.latitude,
        lng: position.coords.longitude,
      });
    },
    () => {
      $('doctorNotice').textContent =
        'Location permission was denied. Search by city, or open the map to browse.';
    },
    { enableHighAccuracy: false, timeout: 12000, maximumAge: 300000 },
  );
}

/* ── boot ────────────────────────────────────────────────────────── */

async function loadMeta() {
  try {
    const meta = await fetchJson('/api/meta', {}, 15000);
    if (!meta) return;

    $('modelMeta').textContent =
      `${meta.model} · ${meta.n_train_subjects} subjects · ` +
      `${meta.n_train_epochs.toLocaleString()} epochs · k=${meta.k}`;

    const demos = meta.demos || {};
    Object.keys(demoAvailability).forEach((key) => {
      demoAvailability[key] = Boolean(demos[key] && demos[key].available);
    });
    $('demoAdhd').disabled = !demoAvailability.adhd;
    $('demoControl').disabled = !demoAvailability.control;
  } catch (_) {
    /* Keep static metadata in the page. */
  }
}

function initDragAndDrop() {
  const zone = $('uploadZone');
  ['dragenter', 'dragover'].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add('drag-over');
    });
  });
  ['dragleave', 'dragend'].forEach((name) => {
    zone.addEventListener(name, () => zone.classList.remove('drag-over'));
  });
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('drag-over');
    if (busy) return;
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) processFile(file);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initDragAndDrop();

  $('doctorList').addEventListener('click', (event) => {
    const button = event.target.closest('.show-map-btn');
    if (!button) return;
    showDoctorOnMap(button.dataset.query, button.dataset.name);
  });

  loadMeta();
  loadDoctors({ city: currentCity });
});
