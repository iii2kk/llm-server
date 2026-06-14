const apiKey = document.getElementById('apiKey');
const statusToggle = document.getElementById('statusToggle');
const proxyStatusDot = document.getElementById('proxyStatusDot');
const statusJson = document.getElementById('statusJson');
const backendRows = document.getElementById('backendRows');
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
backendRows.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  const modelId = button?.closest('tr[data-model-id]')?.dataset.modelId;
  if (!button || !modelId) return;

  const action = button.dataset.action;
  if (action === 'logs') {
    logModel.value = modelId;
    connectLogStream();
  } else if (action === 'edit') {
    openSettings(modelId);
  } else if (action === 'stop') {
    stopBackend(modelId);
  } else if (action === 'restart') {
    restartModel(modelId);
  }
});
recentRows.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  const modelId = button?.closest('tr[data-model-id]')?.dataset.modelId;
  if (!button || !modelId) return;

  if (button.dataset.action === 'start-recent') {
    quickStartModel(modelId);
  } else if (button.dataset.action === 'edit-recent') {
    openSettings(modelId);
  }
});
modelRows.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  const modelId = button?.closest('tr[data-model-id]')?.dataset.modelId;
  if (!button || !modelId) return;

  if (button.dataset.action === 'start-model') {
    quickStartModel(modelId);
  } else if (button.dataset.action === 'edit-model') {
    openSettings(modelId);
  }
});
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

function setText(element, value) {
  const next = String(value ?? '');
  if (element.textContent !== next) element.textContent = next;
}

function setClassName(element, value) {
  if (element.className !== value) element.className = value;
}

function setButtonState(button, {disabled, label, neutral = false}) {
  button.disabled = disabled;
  button.classList.toggle('neutral', neutral);
  setText(button, label);
}

function rowByModelId(rows, modelId) {
  return [...rows.children].find((row) => row.dataset.modelId === modelId);
}

function removeUnusedRows(rows, modelIds) {
  const keep = new Set(modelIds);
  for (const row of [...rows.querySelectorAll('tr[data-model-id]')]) {
    if (!keep.has(row.dataset.modelId)) row.remove();
  }
}

function placeRowAt(rows, row, index) {
  const current = rows.children[index];
  if (current !== row) rows.insertBefore(row, current || null);
}

function setEmptyRow(rows, colspan, message) {
  let row = rows.querySelector('tr[data-empty-row]');
  if (!row) {
    row = document.createElement('tr');
    row.dataset.emptyRow = 'true';
    const cell = document.createElement('td');
    cell.colSpan = colspan;
    cell.style.color = 'var(--muted)';
    row.appendChild(cell);
    rows.appendChild(row);
  }
  setText(row.firstElementChild, message);
}

function removeEmptyRow(rows) {
  rows.querySelector('tr[data-empty-row]')?.remove();
}

function createStateContent(cell) {
  const state = document.createElement('span');
  state.className = 'state';
  state.dataset.field = 'state-wrap';
  const dot = document.createElement('span');
  dot.className = 'dot';
  dot.dataset.field = 'state-dot';
  const label = document.createElement('span');
  label.dataset.field = 'state-label';
  state.append(dot, label);
  cell.appendChild(state);
}

function updateStateContent(cell, backend, fallback = 'stopped') {
  const state = cell.querySelector('[data-field="state-wrap"]');
  const dot = cell.querySelector('[data-field="state-dot"]');
  const label = cell.querySelector('[data-field="state-label"]');
  const dotClass = backend?.load_state === 'ready'
    ? 'ready'
    : backend?.load_state === 'loading'
      ? 'loading'
      : backend?.load_state === 'error'
        ? 'error'
        : '';
  setClassName(state, 'state');
  setClassName(dot, `dot ${dotClass}`.trim());
  dot.hidden = false;
  setText(label, backend ? stateLabel(backend) : fallback);
}

function createModelRow(item) {
  const row = document.createElement('tr');
  row.dataset.modelId = item.relative_path;
  row.innerHTML = `
    <td class="model-cell"><span data-field="display-name"></span><span class="subtext" data-field="path"></span></td>
    <td data-field="size"></td>
    <td><span class="pill" data-field="mode"></span><span class="subtext" data-field="architecture"></span></td>
    <td><span class="pill" data-field="mmproj"></span></td>
    <td><span class="pill" data-field="saved"></span></td>
    <td data-field="state"></td>
    <td>
      <button class="compact" data-action="start-model"></button>
      <button class="neutral compact" data-action="edit-model">Edit</button>
    </td>
  `;
  createStateContent(row.querySelector('[data-field="state"]'));
  return row;
}

function updateModelRow(row, item) {
  const backend = backendForModel(item.relative_path);
  const running = Boolean(backend && (backend.running || backend.load_state === 'loading' || backend.load_state === 'ready'));
  const mode = row.querySelector('[data-field="mode"]');
  const mmproj = row.querySelector('[data-field="mmproj"]');
  const saved = row.querySelector('[data-field="saved"]');

  row.classList.toggle('selected', item.relative_path === selectedModelId);
  setText(row.querySelector('[data-field="display-name"]'), item.display_name || item.relative_path);
  setText(row.querySelector('[data-field="path"]'), item.relative_path);
  setText(row.querySelector('[data-field="size"]'), formatBytes(item.size_bytes));
  setClassName(mode, `pill ${item.effective_mode === 'embeddings' ? 'ok' : item.effective_mode === 'rerank' ? 'warn' : ''}`.trim());
  setText(mode, item.effective_mode || 'chat');
  setText(row.querySelector('[data-field="architecture"]'), `${item.architecture || 'unknown'}${item.effective_pooling ? ` / ${item.effective_pooling}` : ''}`);
  setClassName(mmproj, `pill ${item.mmproj_path ? 'ok' : ''}`.trim());
  setText(mmproj, item.mmproj_path ? 'yes' : 'none');
  setClassName(saved, `pill ${savedSettings[item.relative_path] ? 'ok' : 'warn'}`);
  setText(saved, savedSettings[item.relative_path] ? 'saved' : 'default');
  updateStateContent(row.querySelector('[data-field="state"]'), backend);
  setButtonState(row.querySelector('[data-action="start-model"]'), {
    disabled: running,
    label: running ? 'Running' : 'Start',
    neutral: running,
  });
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

  if (!filtered.length) {
    removeUnusedRows(modelRows, []);
    setEmptyRow(modelRows, 7, 'No matching GGUF files.');
  } else {
    removeEmptyRow(modelRows);
    const modelIds = filtered.map((item) => item.relative_path);
    removeUnusedRows(modelRows, modelIds);
    filtered.forEach((item, index) => {
      const row = rowByModelId(modelRows, item.relative_path) || createModelRow(item);
      updateModelRow(row, item);
      placeRowAt(modelRows, row, index);
    });
  }

  if (selectedModelId) {
    localStorage.setItem('selectedModelId', selectedModelId);
  }
  if (options.applySettings || selectedModelId !== previous) {
    applySelectedModelSettings();
  }
  const selected = selectedModel();
  setText(
    document.getElementById('modelMeta'),
    `${filtered.length} / ${allModels.length} GGUF files under ${modelDir}; selected: ${selected ? (selected.display_name || selected.relative_path) : 'none'}`,
  );
}

function createRecentModelRow(modelId) {
  const row = document.createElement('tr');
  row.dataset.modelId = modelId;
  row.innerHTML = `
    <td class="model-cell"><span data-field="display-name"></span><span class="subtext" data-field="details"></span></td>
    <td data-field="state"></td>
    <td>
      <button class="compact" data-action="start-recent"></button>
      <button class="neutral compact" data-action="edit-recent">Edit</button>
    </td>
  `;
  createStateContent(row.querySelector('[data-field="state"]'));
  return row;
}

function updateRecentModelRow(row, modelId) {
  const item = modelItem(modelId);
  const backend = backendForModel(modelId);
  const missing = !item;
  const running = Boolean(backend && (backend.running || backend.load_state === 'loading' || backend.load_state === 'ready'));
  const savedBackend = savedSettings[modelId]?.backend || defaultBackend;
  const stateCell = row.querySelector('[data-field="state"]');
  const state = stateCell.querySelector('[data-field="state-wrap"]');
  const dot = stateCell.querySelector('[data-field="state-dot"]');

  setText(row.querySelector('[data-field="display-name"]'), item?.display_name || modelId);
  setText(row.querySelector('[data-field="details"]'), `${modelId} / ${backendName(savedBackend)}`);
  if (missing) {
    setClassName(state, 'pill missing');
    dot.hidden = true;
    setText(stateCell.querySelector('[data-field="state-label"]'), 'missing');
  } else {
    updateStateContent(stateCell, backend);
  }
  setButtonState(row.querySelector('[data-action="start-recent"]'), {
    disabled: missing || running,
    label: running ? 'Running' : 'Start',
    neutral: running,
  });
  row.querySelector('[data-action="edit-recent"]').disabled = missing;
}

function renderRecentModels() {
  const modelIds = recentModels.slice(0, 5);
  if (!modelIds.length) {
    removeUnusedRows(recentRows, []);
    setEmptyRow(recentRows, 3, 'No recent models yet.');
    return;
  }

  removeEmptyRow(recentRows);
  removeUnusedRows(recentRows, modelIds);
  modelIds.forEach((modelId, index) => {
    const row = rowByModelId(recentRows, modelId) || createRecentModelRow(modelId);
    updateRecentModelRow(row, modelId);
    placeRowAt(recentRows, row, index);
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
  setText(statusJson, JSON.stringify(data, null, 2));
  proxyStatusDot.className = `dot ${data.running ? 'ready' : ''}`;
  setText(document.getElementById('activeCount'), data.count || 0);
  setText(document.getElementById('latestModel'), data.latest_model_id ? modelName(data.latest_model_id) : 'none');
  setText(document.getElementById('startPort'), data.backend_start_port ?? '-');
  renderBackends(data.backends || []);
  renderLogOptions(data.backends || []);
  renderRecentModels();
  renderModels({applySettings: false});
}

function createBackendRow(modelId) {
  const row = document.createElement('tr');
  row.dataset.modelId = modelId;
  row.innerHTML = `
    <td class="model-cell" data-field="model"></td>
    <td><span class="pill" data-field="backend"></span></td>
    <td><span class="pill" data-field="mode"></span><span class="subtext" data-field="pooling"></span></td>
    <td data-field="state"></td>
    <td data-field="port"></td>
    <td data-field="uptime"></td>
    <td>
      <button class="neutral compact" data-action="logs">Logs</button>
      <button class="neutral compact" data-action="edit">Edit</button>
      <button class="secondary compact" data-action="restart">Restart</button>
      <button class="danger compact" data-action="stop">Stop</button>
    </td>
  `;
  createStateContent(row.querySelector('[data-field="state"]'));
  return row;
}

function updateBackendRow(row, backend) {
  const mode = row.querySelector('[data-field="mode"]');
  setText(row.querySelector('[data-field="model"]'), modelName(backend.model_id));
  setText(row.querySelector('[data-field="backend"]'), backendName(backend.backend || defaultBackend));
  setClassName(mode, `pill ${backend.effective_mode === 'embeddings' ? 'ok' : ''}`.trim());
  setText(mode, backend.effective_mode || 'chat');
  setText(row.querySelector('[data-field="pooling"]'), backend.effective_pooling || '');
  updateStateContent(row.querySelector('[data-field="state"]'), backend);
  setText(row.querySelector('[data-field="port"]'), backend.port ?? '-');
  setText(row.querySelector('[data-field="uptime"]'), backend.uptime_seconds == null ? '-' : `${backend.uptime_seconds}s`);
}

function renderBackends(backends) {
  if (!backends.length) {
    removeUnusedRows(backendRows, []);
    setEmptyRow(backendRows, 7, 'No models have been started.');
    return;
  }

  removeEmptyRow(backendRows);
  const modelIds = backends.map((backend) => backend.model_id);
  removeUnusedRows(backendRows, modelIds);
  backends.forEach((backend, index) => {
    const row = rowByModelId(backendRows, backend.model_id) || createBackendRow(backend.model_id);
    updateBackendRow(row, backend);
    placeRowAt(backendRows, row, index);
  });
}

function renderLogOptions(backends) {
  const previous = logModel.value;
  const modelIds = new Set(backends.map((backend) => backend.model_id));
  for (const option of [...logModel.options]) {
    if (option.value && !modelIds.has(option.value)) option.remove();
  }
  for (const backend of backends) {
    let option = [...logModel.options].find((item) => item.value === backend.model_id);
    if (!option) {
      option = document.createElement('option');
      option.value = backend.model_id;
      logModel.appendChild(option);
    }
    setText(option, modelName(backend.model_id));
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

updateGpuLayersInput();
updatePoolingControl();
connectLogStream();
refreshAll();
setInterval(loadStatus, 5000);
