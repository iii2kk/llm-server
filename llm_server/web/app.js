const apiKey = document.getElementById('apiKey');
const statusToggle = document.getElementById('statusToggle');
const proxyStatusDot = document.getElementById('proxyStatusDot');
const statusJson = document.getElementById('statusJson');
const recentRows = document.getElementById('recentRows');
const modelRows = document.getElementById('modelRows');
const modelFilter = document.getElementById('modelFilter');
const mmprojEnabled = document.getElementById('mmproj_enabled');
const mmprojMeta = document.getElementById('mmprojMeta');
const backendInput = document.getElementById('backend');
const modeInput = document.getElementById('mode');
const poolingInput = document.getElementById('pooling');
const gpuLayersMode = document.getElementById('gpu_layers_mode');
const gpuLayersInput = document.getElementById('gpu_layers');
const logsPanel = document.getElementById('logsPanel');
const logsPre = document.getElementById('logsPre');
const autoScroll = document.getElementById('autoScroll');
const logModel = document.getElementById('logModel');
const logStreamState = document.getElementById('logStreamState');
const toggleLogsBtn = document.getElementById('toggleLogsBtn');
const messageLine = document.getElementById('messageLine');
const settingsDialog = document.getElementById('settingsDialog');
const settingsTitle = document.getElementById('settingsTitle');
const dialogMessage = document.getElementById('dialogMessage');
const dialogStartBtn = document.getElementById('dialogStartBtn');
const dialogRestartBtn = document.getElementById('dialogRestartBtn');
let allModels = [];
let modelDir = '';
let availableBackends = [];
let defaultBackend = '';
let savedSettings = {};
let recentModels = [];
let selectedModelId = localStorage.getItem('selectedModelId') || '';
let statusData = {backends: []};
let logSource = null;
let logReconnectTimer = null;
const LOG_VIEW_MAX_CHARS = 200000;

apiKey.value = localStorage.getItem('proxyApiKey') || '';
modelFilter.value = localStorage.getItem('modelFilter') || '';

dialogStartBtn.addEventListener('click', () => startFromDialog());
dialogRestartBtn.addEventListener('click', () => restartFromDialog());
document.getElementById('settingsCancelBtn').addEventListener('click', () => settingsDialog.close());
document.getElementById('settingsCloseBtn').addEventListener('click', () => settingsDialog.close());
document.getElementById('stopAllBtn').addEventListener('click', () => stopAllBackends());
document.getElementById('refreshBtn').addEventListener('click', () => refreshAll());
document.getElementById('clearLogsBtn').addEventListener('click', () => clearLogs());
toggleLogsBtn.addEventListener('click', () => setLogsCollapsed(!logsPanel.classList.contains('collapsed')));
statusToggle.addEventListener('click', () => setStatusJsonOpen(statusJson.hidden));
apiKey.addEventListener('input', () => {
  localStorage.setItem('proxyApiKey', apiKey.value);
  scheduleLogReconnect();
});
modelFilter.addEventListener('input', () => {
  localStorage.setItem('modelFilter', modelFilter.value);
  renderModels();
});
modeInput.addEventListener('change', () => updatePoolingControl());
gpuLayersMode.addEventListener('change', () => updateGpuLayersInput(true));
logModel.addEventListener('change', () => connectLogStream());
document.addEventListener('click', (event) => {
  if (!statusJson.hidden && !event.target.closest('.status-menu')) {
    setStatusJsonOpen(false);
  }
});

setLogsCollapsed(localStorage.getItem('logsCollapsed') === 'true');

function headers() {
  const h = {'Content-Type': 'application/json'};
  if (apiKey.value) h.Authorization = `Bearer ${apiKey.value}`;
  return h;
}

async function api(path, options = {}) {
  const res = await fetch(path, {...options, headers: {...headers(), ...(options.headers || {})}});
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = {raw: text}; }
  if (!res.ok) throw new Error(data?.error?.message || text || res.statusText);
  return data;
}

function settings() {
  if (!selectedModelId) throw new Error('No model selected.');
  const payload = {
    model: selectedModelId,
    backend: backendInput.value,
    mode: modeInput.value,
    pooling: poolingInput.value,
    mmproj_enabled: mmprojEnabled.checked && !mmprojEnabled.disabled,
    flash_attn: document.getElementById('flash_attn').value,
    reasoning: document.getElementById('reasoning').value,
    reasoning_format: document.getElementById('reasoning_format').value,
  };
  if (gpuLayersMode.value === 'all') {
    payload.gpu_layers = 'all';
  } else if (gpuLayersMode.value === 'custom') {
    if (gpuLayersInput.value === '') throw new Error('GPU Layers custom value is required.');
    payload.gpu_layers = Number(gpuLayersInput.value);
  }
  for (const key of ['ctx_size', 'threads', 'batch_size', 'ubatch_size', 'parallel', 'reasoning_budget']) {
    const value = document.getElementById(key).value;
    if (value !== '') payload[key] = Number(value);
  }
  return payload;
}

function selectedModel() {
  return allModels.find((item) => item.relative_path === selectedModelId);
}

function modelName(modelId) {
  const item = allModels.find((entry) => entry.relative_path === modelId);
  return item?.display_name || modelId;
}

function backendName(backendId) {
  const backend = availableBackends.find((entry) => entry.id === backendId);
  return backend?.label || backendId || '-';
}

function updateGpuLayersInput(shouldFocus = false) {
  const custom = gpuLayersMode.value === 'custom';
  gpuLayersInput.hidden = !custom;
  gpuLayersInput.disabled = !custom;
  if (custom && shouldFocus) gpuLayersInput.focus();
}

function updatePoolingControl() {
  const item = selectedModel();
  const effectiveMode = modeInput.value === 'auto' ? item?.detected_mode : modeInput.value;
  poolingInput.disabled = effectiveMode !== 'embeddings';
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

function setSelectValue(id, value, fallback) {
  const input = document.getElementById(id);
  const next = value == null || value === '' ? fallback : String(value);
  input.value = [...input.options].some((option) => option.value === next) ? next : fallback;
}

function setNumberValue(id, settings) {
  const input = document.getElementById(id);
  input.value = hasOwn(settings, id) ? String(settings[id]) : '';
}

function applySelectedModelSettings() {
  const item = selectedModel();
  const settings = savedSettings[selectedModelId] || {};
  const mmprojPath = item?.mmproj_path || '';
  const hasMmproj = Boolean(mmprojPath);
  mmprojEnabled.disabled = !hasMmproj;
  mmprojEnabled.checked = hasMmproj && (hasOwn(settings, 'mmproj_enabled') ? Boolean(settings.mmproj_enabled) : true);
  mmprojMeta.textContent = hasMmproj ? `MMProj: ${mmprojPath}` : 'MMProj: none';

  for (const key of ['ctx_size', 'threads', 'batch_size', 'ubatch_size', 'parallel', 'reasoning_budget']) {
    setNumberValue(key, settings);
  }

  const gpuLayers = settings.gpu_layers;
  if (gpuLayers === 'all') {
    gpuLayersMode.value = 'all';
    gpuLayersInput.value = '';
  } else if (gpuLayers !== undefined && gpuLayers !== null && gpuLayers !== '') {
    gpuLayersMode.value = 'custom';
    gpuLayersInput.value = String(gpuLayers);
  } else {
    gpuLayersMode.value = 'auto';
    gpuLayersInput.value = '';
  }
  updateGpuLayersInput(false);

  setSelectValue('flash_attn', settings.flash_attn, 'auto');
  setSelectValue('reasoning', settings.reasoning, 'off');
  setSelectValue('reasoning_format', settings.reasoning_format, 'none');
  setSelectValue('backend', settings.backend, defaultBackend);
  setSelectValue('mode', settings.mode, 'auto');
  setSelectValue('pooling', settings.pooling, 'auto');
  updatePoolingControl();
}

async function loadModels(applySettings = true) {
  const data = await api('/api/models');
  allModels = data.models || [];
  modelDir = data.model_dir || '';
  availableBackends = data.backends || [];
  defaultBackend = data.default_backend || availableBackends[0]?.id || '';
  backendInput.innerHTML = '';
  for (const backend of availableBackends) {
    const option = document.createElement('option');
    option.value = backend.id;
    option.textContent = backend.label || backend.id;
    option.title = backend.bin_dir || '';
    backendInput.appendChild(option);
  }
  savedSettings = data.saved_settings || {};
  recentModels = data.recent_models || [];
  renderRecentModels();
  renderModels({applySettings});
}

function modelItem(modelId) {
  return allModels.find((item) => item.relative_path === modelId);
}

function renderModels(options = {}) {
  const previous = selectedModelId;
  const query = modelFilter.value.trim().toLowerCase();
  const filtered = allModels.filter((item) => {
    const haystack = `${item.display_name} ${item.relative_path} ${item.name} ${item.path}`.toLowerCase();
    return !query || haystack.includes(query);
  });

  if (!allModels.some((item) => item.relative_path === selectedModelId)) {
    selectedModelId = filtered[0]?.relative_path || allModels[0]?.relative_path || '';
  }

  modelRows.innerHTML = '';
  if (!filtered.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="7" style="color: var(--muted)">No matching GGUF files.</td>';
    modelRows.appendChild(tr);
  }
  for (const item of filtered) {
    const backend = backendForModel(item.relative_path);
    const dotClass = backend?.load_state === 'ready' ? 'ready' : backend?.load_state === 'loading' ? 'loading' : backend?.load_state === 'error' ? 'error' : '';
    const state = backend ? stateLabel(backend) : 'stopped';
    const running = Boolean(backend && (backend.running || backend.load_state === 'loading' || backend.load_state === 'ready'));
    const tr = document.createElement('tr');
    tr.className = item.relative_path === selectedModelId ? 'selected' : '';
    tr.innerHTML = `
      <td class="model-cell">${escapeHtml(item.display_name || item.relative_path)}<span class="subtext">${escapeHtml(item.relative_path)}</span></td>
      <td>${escapeHtml(formatBytes(item.size_bytes))}</td>
      <td><span class="pill ${item.effective_mode === 'embeddings' ? 'ok' : item.effective_mode === 'rerank' ? 'warn' : ''}">${escapeHtml(item.effective_mode || 'chat')}</span><span class="subtext">${escapeHtml(item.architecture || 'unknown')}${item.effective_pooling ? ` / ${escapeHtml(item.effective_pooling)}` : ''}</span></td>
      <td><span class="pill ${item.mmproj_path ? 'ok' : ''}">${item.mmproj_path ? 'yes' : 'none'}</span></td>
      <td><span class="pill ${savedSettings[item.relative_path] ? 'ok' : 'warn'}">${savedSettings[item.relative_path] ? 'saved' : 'default'}</span></td>
      <td><span class="state"><span class="dot ${dotClass}"></span>${escapeHtml(state)}</span></td>
      <td>
        <button class="compact ${running ? 'neutral' : ''}" data-action="start-model" data-model="${escapeAttr(item.relative_path)}" ${running ? 'disabled' : ''}>${running ? 'Running' : 'Start'}</button>
        <button class="neutral compact" data-action="edit-model" data-model="${escapeAttr(item.relative_path)}">Edit</button>
      </td>
    `;
    modelRows.appendChild(tr);
  }

  modelRows.querySelectorAll('button[data-action]').forEach((button) => {
    button.addEventListener('click', () => {
      const id = button.getAttribute('data-model');
      const action = button.getAttribute('data-action');
      if (action === 'start-model') {
        quickStartModel(id);
      } else if (action === 'edit-model') {
        openSettings(id);
      }
    });
  });

  if (selectedModelId) {
    localStorage.setItem('selectedModelId', selectedModelId);
  }
  if (options.applySettings || selectedModelId !== previous) {
    applySelectedModelSettings();
  }
  const selected = selectedModel();
  document.getElementById('modelMeta').textContent =
    `${filtered.length} / ${allModels.length} GGUF files under ${modelDir}; selected: ${selected ? (selected.display_name || selected.relative_path) : 'none'}`;
}

function renderRecentModels() {
  recentRows.innerHTML = '';
  if (!recentModels.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="3" style="color: var(--muted)">No recent models yet.</td>';
    recentRows.appendChild(tr);
    return;
  }

  for (const modelId of recentModels.slice(0, 5)) {
    const item = modelItem(modelId);
    const backend = backendForModel(modelId);
    const missing = !item;
    const running = Boolean(backend && (backend.running || backend.load_state === 'loading' || backend.load_state === 'ready'));
    const dotClass = backend?.load_state === 'ready' ? 'ready' : backend?.load_state === 'loading' ? 'loading' : backend?.load_state === 'error' ? 'error' : '';
    const state = missing ? 'missing' : backend ? stateLabel(backend) : 'stopped';
    const stateHtml = missing
      ? '<span class="pill missing">missing</span>'
      : `<span class="state"><span class="dot ${dotClass}"></span>${escapeHtml(state)}</span>`;
    const startDisabled = missing || running;
    const selectDisabled = missing;
    const tr = document.createElement('tr');
    const savedBackend = savedSettings[modelId]?.backend || defaultBackend;
    tr.innerHTML = `
      <td class="model-cell">${escapeHtml(item?.display_name || modelId)}<span class="subtext">${escapeHtml(modelId)} / ${escapeHtml(backendName(savedBackend))}</span></td>
      <td>${stateHtml}</td>
      <td>
        <button class="compact ${running ? 'neutral' : ''}" data-action="start-recent" data-model="${escapeAttr(modelId)}" ${startDisabled ? 'disabled' : ''}>${running ? 'Running' : 'Start'}</button>
        <button class="neutral compact" data-action="edit-recent" data-model="${escapeAttr(modelId)}" ${selectDisabled ? 'disabled' : ''}>Edit</button>
      </td>
    `;
    recentRows.appendChild(tr);
  }

  recentRows.querySelectorAll('button[data-action]').forEach((button) => {
    button.addEventListener('click', () => {
      const id = button.getAttribute('data-model');
      const action = button.getAttribute('data-action');
      if (action === 'start-recent') {
        quickStartModel(id);
      } else if (action === 'edit-recent') {
        openSettings(id);
      }
    });
  });
}

function openSettings(modelId) {
  if (!modelId) return;
  selectedModelId = modelId;
  localStorage.setItem('selectedModelId', selectedModelId);
  applySelectedModelSettings();
  renderModels({applySettings: false});
  const item = modelItem(modelId);
  settingsTitle.textContent = `Configure: ${item?.display_name || modelId}`;
  dialogMessage.textContent = '';
  updateDialogActions(modelId);
  if (!settingsDialog.open) settingsDialog.showModal();
}

function updateDialogActions(modelId) {
  const backend = backendForModel(modelId);
  const running = Boolean(backend && (backend.running || backend.load_state === 'loading' || backend.load_state === 'ready'));
  dialogStartBtn.disabled = running;
  dialogStartBtn.title = running ? 'Already running. Use Restart to apply changes.' : '';
}

function backendForModel(modelId) {
  return (statusData.backends || []).find((backend) => backend.model_id === modelId);
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!value) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function stateLabel(backend) {
  const state = backend.load_state || (backend.running ? 'running' : 'stopped');
  const progress = backend.load_progress;
  return state === 'loading' ? `loading ${progress ?? 0}%` : state;
}

function setStatusJsonOpen(open) {
  statusJson.hidden = !open;
  statusToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function renderStatus(data) {
  statusData = data;
  statusJson.textContent = JSON.stringify(data, null, 2);
  proxyStatusDot.className = `dot ${data.running ? 'ready' : ''}`;
  document.getElementById('activeCount').textContent = String(data.count || 0);
  document.getElementById('latestModel').textContent = data.latest_model_id ? modelName(data.latest_model_id) : 'none';
  document.getElementById('startPort').textContent = String(data.backend_start_port ?? '-');
  renderBackends(data.backends || []);
  renderLogOptions(data.backends || []);
  renderRecentModels();
  renderModels({applySettings: false});
}

function renderBackends(backends) {
  const rows = document.getElementById('backendRows');
  rows.innerHTML = '';
  if (!backends.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="7" style="color: var(--muted)">No models have been started.</td>';
    rows.appendChild(tr);
    return;
  }
  for (const backend of backends) {
    const tr = document.createElement('tr');
    const dotClass = backend.load_state === 'ready' ? 'ready' : backend.load_state === 'loading' ? 'loading' : backend.load_state === 'error' ? 'error' : '';
    const uptime = backend.uptime_seconds == null ? '-' : `${backend.uptime_seconds}s`;
    tr.innerHTML = `
      <td class="model-cell">${escapeHtml(modelName(backend.model_id))}</td>
      <td><span class="pill">${escapeHtml(backendName(backend.backend || defaultBackend))}</span></td>
      <td><span class="pill ${backend.effective_mode === 'embeddings' ? 'ok' : ''}">${escapeHtml(backend.effective_mode || 'chat')}</span>${backend.effective_pooling ? `<span class="subtext">${escapeHtml(backend.effective_pooling)}</span>` : ''}</td>
      <td><span class="state"><span class="dot ${dotClass}"></span>${escapeHtml(stateLabel(backend))}</span></td>
      <td>${backend.port ?? '-'}</td>
      <td>${uptime}</td>
      <td>
        <button class="neutral compact" data-action="logs" data-model="${escapeAttr(backend.model_id)}">Logs</button>
        <button class="neutral compact" data-action="edit" data-model="${escapeAttr(backend.model_id)}">Edit</button>
        <button class="secondary compact" data-action="restart" data-model="${escapeAttr(backend.model_id)}">Restart</button>
        <button class="danger compact" data-action="stop" data-model="${escapeAttr(backend.model_id)}">Stop</button>
      </td>
    `;
    rows.appendChild(tr);
  }
  rows.querySelectorAll('button[data-action]').forEach((button) => {
    button.addEventListener('click', () => {
      const id = button.getAttribute('data-model');
      const action = button.getAttribute('data-action');
      if (action === 'logs') {
        logModel.value = id;
        connectLogStream();
      } else if (action === 'edit') {
        openSettings(id);
      } else if (action === 'stop') {
        stopBackend(id);
      } else if (action === 'restart') {
        restartModel(id);
      }
    });
  });
}

function renderLogOptions(backends) {
  const previous = logModel.value;
  logModel.innerHTML = '<option value="">All models</option>';
  for (const backend of backends) {
    const opt = document.createElement('option');
    opt.value = backend.model_id;
    opt.textContent = modelName(backend.model_id);
    logModel.appendChild(opt);
  }
  if ([...logModel.options].some((opt) => opt.value === previous)) {
    logModel.value = previous;
  }
}

async function loadStatus() {
  const data = await api('/api/status');
  renderStatus(data);
}

async function startFromDialog() {
  const ok = await runAction(async () => {
    await api('/api/start', {method: 'POST', body: JSON.stringify(settings())});
  }, 'started', {messageEl: dialogMessage});
  if (ok) settingsDialog.close();
}

async function restartFromDialog() {
  const ok = await runAction(async () => {
    await api('/api/restart', {method: 'POST', body: JSON.stringify(settings())});
  }, 'restarted', {messageEl: dialogMessage});
  if (ok) settingsDialog.close();
}

async function quickStartModel(modelId) {
  await runAction(async () => {
    const payload = {...(savedSettings[modelId] || {}), model: modelId};
    await api('/api/start', {method: 'POST', body: JSON.stringify(payload)});
  }, 'started');
}

async function restartModel(modelId) {
  await runAction(async () => {
    const payload = {...(savedSettings[modelId] || {}), model: modelId};
    await api('/api/restart', {method: 'POST', body: JSON.stringify(payload)});
  }, 'restarted');
}

async function stopBackend(modelId) {
  await runAction(async () => {
    if (!modelId) throw new Error('No model selected.');
    await api('/api/stop', {method: 'POST', body: JSON.stringify({model: modelId})});
  }, 'stopped');
}

async function stopAllBackends() {
  await runAction(async () => {
    await api('/api/stop', {method: 'POST', body: JSON.stringify({all: true})});
  }, 'stopped all');
}

async function runAction(action, label, {messageEl = messageLine} = {}) {
  try {
    messageEl.textContent = 'working...';
    await action();
    messageEl.textContent = label;
    await loadModels(false);
    await loadStatus();
    scheduleLogReconnect();
    return true;
  } catch (err) {
    messageEl.textContent = String(err);
    return false;
  }
}

async function refreshAll() {
  try {
    await loadModels();
    await loadStatus();
  } catch (err) {
    messageLine.textContent = String(err);
  }
}

function appendLog(text) {
  logsPre.textContent += text;
  if (logsPre.textContent.length > LOG_VIEW_MAX_CHARS) {
    logsPre.textContent = logsPre.textContent.slice(-LOG_VIEW_MAX_CHARS);
  }
  if (autoScroll.checked) logsPre.scrollTop = logsPre.scrollHeight;
}

function setLogsCollapsed(collapsed) {
  document.body.classList.toggle('logs-collapsed', collapsed);
  logsPanel.classList.toggle('collapsed', collapsed);
  toggleLogsBtn.textContent = collapsed ? 'Restore' : 'Minimize';
  toggleLogsBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  localStorage.setItem('logsCollapsed', collapsed ? 'true' : 'false');
  if (!collapsed && autoScroll.checked) {
    logsPre.scrollTop = logsPre.scrollHeight;
  }
}

function clearLogs() {
  logsPre.textContent = '';
}

function parseSseData(event) {
  try { return JSON.parse(event.data); } catch { return null; }
}

function renderLogSnapshot(data) {
  logsPre.textContent = (data.entries || []).map((entry) => entry.text).join('');
  if (autoScroll.checked) logsPre.scrollTop = logsPre.scrollHeight;
}

function connectLogStream() {
  if (logSource) logSource.close();
  const url = new URL('/api/logs/stream', window.location.origin);
  if (apiKey.value) url.searchParams.set('api_key', apiKey.value);
  if (logModel.value) url.searchParams.set('model', logModel.value);
  logStreamState.textContent = 'connecting';
  logSource = new EventSource(url);
  logSource.onopen = () => { logStreamState.textContent = 'connected'; };
  logSource.onerror = () => { logStreamState.textContent = 'disconnected'; };
  logSource.addEventListener('snapshot', (event) => {
    const data = parseSseData(event);
    if (data) renderLogSnapshot(data);
  });
  logSource.addEventListener('log', (event) => {
    const data = parseSseData(event);
    if (data) appendLog(data.text || '');
  });
}

function scheduleLogReconnect() {
  clearTimeout(logReconnectTimer);
  logReconnectTimer = setTimeout(connectLogStream, 250);
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, '&#96;');
}

updateGpuLayersInput();
updatePoolingControl();
connectLogStream();
refreshAll();
setInterval(loadStatus, 5000);
