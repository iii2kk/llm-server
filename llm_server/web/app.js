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
const mtpInput = document.getElementById('mtp');
const mtpDraftTokensInput = document.getElementById('mtp_draft_tokens');
const mtpMeta = document.getElementById('mtpMeta');
const logsPanel = document.getElementById('logsPanel');
const logsPre = document.getElementById('logsPre');
const autoScroll = document.getElementById('autoScroll');
const logModel = document.getElementById('logModel');
const logStreamState = document.getElementById('logStreamState');
const toggleLogsBtn = document.getElementById('toggleLogsBtn');
const startupProfilePath = document.getElementById('startupProfilePath');
const startupProfileMessage = document.getElementById('startupProfileMessage');
const requestLogModel = document.getElementById('requestLogModel');
const requestLogDate = document.getElementById('requestLogDate');
const requestLogEndpoint = document.getElementById('requestLogEndpoint');
const requestLogStatus = document.getElementById('requestLogStatus');
const requestLogSearch = document.getElementById('requestLogSearch');
const requestLogFields = document.getElementById('requestLogFields');
const requestLogHead = document.getElementById('requestLogHead');
const requestLogRows = document.getElementById('requestLogRows');
const requestLogMeta = document.getElementById('requestLogMeta');
const requestLogLoadMoreBtn = document.getElementById('requestLogLoadMoreBtn');
const requestLogRefreshBtn = document.getElementById('requestLogRefreshBtn');
const requestLogDetail = document.getElementById('requestLogDetail');
const requestLogTabs = document.getElementById('requestLogTabs');
const requestLogDetailPre = document.getElementById('requestLogDetailPre');
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
let defaultModels = {};
let selectedModelId = localStorage.getItem('selectedModelId') || '';
let statusData = {backends: []};
let logSource = null;
let logReconnectTimer = null;
let requestLogRecords = [];
let requestLogNextOffset = null;
let selectedRequestLogIndex = -1;
let requestLogActiveTab = 'summary';
let requestLogSearchTimer = null;
const LOG_VIEW_MAX_CHARS = 200000;
const REQUEST_LOG_LIMIT = 100;
const REQUEST_LOG_COLUMNS = [
  {id: 'ts', label: 'Time'},
  {id: 'model', label: 'Model'},
  {id: 'endpoint', label: 'Endpoint'},
  {id: 'status', label: 'Status'},
  {id: 'duration', label: 'ms'},
  {id: 'request', label: 'Request'},
  {id: 'response', label: 'Response'},
  {id: 'usage', label: 'Usage'},
];
const DEFAULT_REQUEST_LOG_COLUMNS = ['ts', 'endpoint', 'status', 'duration', 'request', 'response'];

apiKey.value = localStorage.getItem('proxyApiKey') || '';
modelFilter.value = localStorage.getItem('modelFilter') || '';
startupProfilePath.value = localStorage.getItem('startupProfilePath') || '';

dialogStartBtn.addEventListener('click', () => startFromDialog());
dialogRestartBtn.addEventListener('click', () => restartFromDialog());
document.getElementById('settingsCancelBtn').addEventListener('click', () => settingsDialog.close());
document.getElementById('settingsCloseBtn').addEventListener('click', () => settingsDialog.close());
document.getElementById('stopAllBtn').addEventListener('click', () => stopAllBackends());
document.getElementById('saveStartupProfileBtn').addEventListener('click', () => saveStartupProfile());
document.getElementById('refreshBtn').addEventListener('click', () => refreshAll());
document.getElementById('clearLogsBtn').addEventListener('click', () => clearLogs());
toggleLogsBtn.addEventListener('click', () => setLogsCollapsed(!logsPanel.classList.contains('collapsed')));
statusToggle.addEventListener('click', () => setStatusJsonOpen(statusJson.hidden));
requestLogRefreshBtn.addEventListener('click', () => refreshRequestLogs());
requestLogLoadMoreBtn.addEventListener('click', () => loadRequestLogs(false));
for (const input of [requestLogModel, requestLogDate, requestLogEndpoint, requestLogStatus]) {
  input.addEventListener('change', () => loadRequestLogs(true));
}
requestLogSearch.addEventListener('input', () => {
  clearTimeout(requestLogSearchTimer);
  requestLogSearchTimer = setTimeout(() => loadRequestLogs(true), 250);
});
requestLogRows.addEventListener('click', (event) => {
  const row = event.target.closest('tr[data-log-index]');
  if (!row) return;
  selectRequestLogRecord(Number(row.dataset.logIndex));
});
requestLogTabs.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-log-tab]');
  if (!button) return;
  requestLogActiveTab = button.dataset.logTab;
  renderRequestLogDetail();
});
apiKey.addEventListener('input', () => {
  localStorage.setItem('proxyApiKey', apiKey.value);
  scheduleLogReconnect();
  refreshRequestLogs();
});
modelFilter.addEventListener('input', () => {
  localStorage.setItem('modelFilter', modelFilter.value);
  renderModels();
});
startupProfilePath.addEventListener('input', () => {
  localStorage.setItem('startupProfilePath', startupProfilePath.value);
});
modeInput.addEventListener('change', () => {
  updatePoolingControl();
  updateMtpControl();
});
gpuLayersMode.addEventListener('change', () => updateGpuLayersInput(true));
mtpInput.addEventListener('change', () => updateMtpControl());
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
  } else if (action === 'default') {
    setDefaultModel(modelId);
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
    mtp: mtpInput.value,
    reasoning: document.getElementById('reasoning').value,
    reasoning_format: document.getElementById('reasoning_format').value,
  };
  if (gpuLayersMode.value === 'all') {
    payload.gpu_layers = 'all';
  } else if (gpuLayersMode.value === 'custom') {
    if (gpuLayersInput.value === '') throw new Error('GPU Layers custom value is required.');
    payload.gpu_layers = Number(gpuLayersInput.value);
  }
  for (const key of ['ctx_size', 'threads', 'batch_size', 'ubatch_size', 'parallel', 'mtp_draft_tokens', 'reasoning_budget']) {
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

function updateMtpControl() {
  const item = selectedModel();
  const supported = Boolean(item?.mtp_supported);
  const effectiveMode = modeInput.value === 'auto' ? item?.detected_mode : modeInput.value;
  const enabled = effectiveMode === 'chat'
    && (mtpInput.value === 'on' || (mtpInput.value === 'auto' && supported));
  mtpDraftTokensInput.disabled = !enabled;
  const layers = Number(item?.mtp_layers || 0);
  if (effectiveMode !== 'chat') {
    mtpMeta.textContent = 'Available for chat mode only';
  } else if (supported) {
    mtpMeta.textContent = `Detected: ${layers} MTP layer${layers === 1 ? '' : 's'}`;
  } else if (layers > 0) {
    mtpMeta.textContent = `Detected: ${layers} layer${layers === 1 ? '' : 's'}; embedded MTP is unsupported for this architecture`;
  } else {
    mtpMeta.textContent = 'Detected: none';
  }
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

  for (const key of ['ctx_size', 'threads', 'batch_size', 'ubatch_size', 'parallel', 'mtp_draft_tokens', 'reasoning_budget']) {
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
  setSelectValue('mtp', settings.mtp, 'auto');
  setSelectValue('reasoning', settings.reasoning, 'off');
  setSelectValue('reasoning_format', settings.reasoning_format, 'none');
  setSelectValue('backend', settings.backend, defaultBackend);
  setSelectValue('mode', settings.mode, 'auto');
  setSelectValue('pooling', settings.pooling, 'auto');
  updatePoolingControl();
  updateMtpControl();
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
  defaultModels = data.default_models || {};
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
    <td><span class="pill" data-field="mtp"></span></td>
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
  const mtp = row.querySelector('[data-field="mtp"]');
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
  setClassName(mtp, `pill ${item.effective_mtp ? 'ok' : item.mtp_supported ? 'warn' : ''}`.trim());
  setText(mtp, item.effective_mtp ? 'on' : item.mtp_supported ? 'off' : 'none');
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
  const terms = modelFilter.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filtered = allModels.filter((item) => {
    const haystack = [item.display_name, item.relative_path, item.name, item.path]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    return !terms.length || terms.every((term) => haystack.includes(term));
  });

  if (!allModels.some((item) => item.relative_path === selectedModelId)) {
    selectedModelId = filtered[0]?.relative_path || allModels[0]?.relative_path || '';
  }

  if (!filtered.length) {
    removeUnusedRows(modelRows, []);
    setEmptyRow(modelRows, 8, 'No matching GGUF files.');
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
  defaultModels = data.default_model_ids || defaultModels || {};
  setText(statusJson, JSON.stringify(data, null, 2));
  proxyStatusDot.className = `dot ${data.running ? 'ready' : ''}`;
  setText(document.getElementById('activeCount'), data.count || 0);
  setText(document.getElementById('latestModel'), data.latest_model_id ? modelName(data.latest_model_id) : 'none');
  setText(document.getElementById('defaultChatModel'), defaultModels.chat ? modelName(defaultModels.chat) : 'latest');
  setText(document.getElementById('defaultEmbeddingModel'), defaultModels.embeddings ? modelName(defaultModels.embeddings) : 'latest');
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
    <td><span class="pill" data-field="default"></span></td>
    <td data-field="port"></td>
    <td data-field="uptime"></td>
    <td>
      <div class="backend-actions">
        <button class="neutral compact" data-action="default">Use</button>
        <button class="neutral compact" data-action="logs">Logs</button>
        <button class="neutral compact" data-action="edit">Edit</button>
        <button class="secondary compact" data-action="restart">Restart</button>
        <button class="danger compact" data-action="stop">Stop</button>
      </div>
    </td>
  `;
  createStateContent(row.querySelector('[data-field="state"]'));
  return row;
}

function updateBackendRow(row, backend) {
  const mode = row.querySelector('[data-field="mode"]');
  const defaultPill = row.querySelector('[data-field="default"]');
  const defaultFor = backend.default_for || [];
  const active = Boolean(backend.running || backend.load_state === 'loading' || backend.load_state === 'ready');
  const defaultForMode = defaultFor.includes(backend.effective_mode);
  setText(row.querySelector('[data-field="model"]'), modelName(backend.model_id));
  setText(row.querySelector('[data-field="backend"]'), backendName(backend.backend || defaultBackend));
  setClassName(mode, `pill ${backend.effective_mode === 'embeddings' ? 'ok' : ''}`.trim());
  setText(mode, backend.effective_mode || 'chat');
  const details = [
    backend.effective_pooling || '',
    backend.effective_mtp ? `MTP ×${backend.mtp_draft_tokens || 3}` : '',
  ].filter(Boolean).join(' / ');
  setText(row.querySelector('[data-field="pooling"]'), details);
  updateStateContent(row.querySelector('[data-field="state"]'), backend);
  setClassName(defaultPill, `pill ${defaultForMode ? 'ok' : ''}`.trim());
  setText(defaultPill, defaultFor.length ? defaultFor.join(', ') : '-');
  setText(row.querySelector('[data-field="port"]'), backend.port ?? '-');
  setText(row.querySelector('[data-field="uptime"]'), backend.uptime_seconds == null ? '-' : `${backend.uptime_seconds}s`);
  setButtonState(row.querySelector('[data-action="default"]'), {
    disabled: !active || defaultForMode,
    label: defaultForMode ? 'Default' : 'Use',
    neutral: true,
  });
  row.querySelector('[data-action="default"]').title = active
    ? 'Use for requests without a model'
    : 'Start the model before selecting it';
}

function renderBackends(backends) {
  if (!backends.length) {
    removeUnusedRows(backendRows, []);
    setEmptyRow(backendRows, 8, 'No models have been started.');
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
  } else {
    logModel.value = '';
  }
}

function requestLogVisibleColumns() {
  try {
    const stored = JSON.parse(localStorage.getItem('requestLogColumns') || '[]');
    const valid = stored.filter((id) => REQUEST_LOG_COLUMNS.some((column) => column.id === id));
    if (valid.length) return valid;
  } catch {
    // Ignore invalid localStorage and restore defaults below.
  }
  return DEFAULT_REQUEST_LOG_COLUMNS;
}

function setRequestLogVisibleColumns(columns) {
  localStorage.setItem('requestLogColumns', JSON.stringify(columns));
}

function renderRequestLogFields() {
  const visible = new Set(requestLogVisibleColumns());
  requestLogFields.replaceChildren();
  for (const column of REQUEST_LOG_COLUMNS) {
    const label = document.createElement('label');
    label.className = 'check-row request-log-field';
    label.htmlFor = `request-log-field-${column.id}`;
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `request-log-field-${column.id}`;
    checkbox.value = column.id;
    checkbox.checked = visible.has(column.id);
    checkbox.addEventListener('change', () => {
      const next = [...requestLogFields.querySelectorAll('input:checked')].map((input) => input.value);
      setRequestLogVisibleColumns(next.length ? next : DEFAULT_REQUEST_LOG_COLUMNS);
      renderRequestLogTable();
    });
    const text = document.createElement('span');
    text.textContent = column.label;
    label.append(checkbox, text);
    requestLogFields.appendChild(label);
  }
}

function setSelectOptions(select, items, {emptyLabel, valueKey = null, labelKey = null, preserve = true} = {}) {
  const previous = preserve ? select.value : '';
  select.replaceChildren();
  const empty = document.createElement('option');
  empty.value = '';
  empty.textContent = emptyLabel;
  select.appendChild(empty);
  for (const item of items) {
    const option = document.createElement('option');
    option.value = valueKey ? item[valueKey] : item;
    option.textContent = labelKey ? item[labelKey] : item;
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === previous)) {
    select.value = previous;
  }
}

async function loadRequestLogOptions() {
  const data = await api('/api/request-logs/options');
  setSelectOptions(
    requestLogModel,
    data.models || [],
    {
      emptyLabel: 'All models',
      valueKey: 'model',
      labelKey: 'model',
    },
  );
  setSelectOptions(requestLogDate, data.dates || [], {emptyLabel: 'All dates'});
  setSelectOptions(requestLogEndpoint, data.endpoints || [], {emptyLabel: 'All endpoints'});
  if (!requestLogDate.value && (data.dates || []).length) {
    requestLogDate.value = data.dates[0];
  }
}

function requestLogUrl(offset) {
  const url = new URL('/api/request-logs', window.location.origin);
  url.searchParams.set('limit', String(REQUEST_LOG_LIMIT));
  url.searchParams.set('offset', String(offset));
  if (requestLogModel.value) url.searchParams.set('model', requestLogModel.value);
  if (requestLogDate.value) url.searchParams.set('date', requestLogDate.value);
  if (requestLogEndpoint.value) url.searchParams.set('endpoint', requestLogEndpoint.value);
  if (requestLogStatus.value) url.searchParams.set('status', requestLogStatus.value);
  if (requestLogSearch.value.trim()) url.searchParams.set('q', requestLogSearch.value.trim());
  return `${url.pathname}${url.search}`;
}

async function refreshRequestLogs() {
  try {
    await loadRequestLogOptions();
    await loadRequestLogs(true);
  } catch (err) {
    requestLogMeta.textContent = String(err);
  }
}

async function loadRequestLogs(reset) {
  const offset = reset ? 0 : requestLogNextOffset;
  if (offset == null) return;
  requestLogMeta.textContent = 'loading...';
  const data = await api(requestLogUrl(offset));
  const nextRecords = data.records || [];
  if (reset) {
    requestLogRecords = nextRecords;
    selectedRequestLogIndex = nextRecords.length ? 0 : -1;
  } else {
    requestLogRecords = requestLogRecords.concat(nextRecords);
  }
  requestLogNextOffset = data.next_offset;
  requestLogLoadMoreBtn.disabled = requestLogNextOffset == null;
  requestLogLoadMoreBtn.hidden = !requestLogRecords.length;
  requestLogMeta.textContent = `${requestLogRecords.length} / ${data.total || 0} records`;
  renderRequestLogTable();
  renderRequestLogDetail();
}

function renderRequestLogTable() {
  const visible = requestLogVisibleColumns();
  const headRow = document.createElement('tr');
  for (const columnId of visible) {
    const column = REQUEST_LOG_COLUMNS.find((item) => item.id === columnId);
    if (!column) continue;
    const th = document.createElement('th');
    th.textContent = column.label;
    headRow.appendChild(th);
  }
  requestLogHead.replaceChildren(headRow);
  requestLogRows.replaceChildren();

  if (!requestLogRecords.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = Math.max(1, visible.length);
    cell.style.color = 'var(--muted)';
    cell.textContent = 'No request logs match the current filters.';
    row.appendChild(cell);
    requestLogRows.appendChild(row);
    return;
  }

  requestLogRecords.forEach((entry, index) => {
    const record = entry.record || {};
    const row = document.createElement('tr');
    row.dataset.logIndex = String(index);
    row.classList.toggle('selected', index === selectedRequestLogIndex);
    for (const columnId of visible) {
      const cell = document.createElement('td');
      cell.className = `request-log-cell request-log-cell-${columnId}`;
      const value = requestLogCellValue(record, columnId);
      if (columnId === 'status') {
        const pill = document.createElement('span');
        pill.className = `pill ${value === 'success' ? 'ok' : 'missing'}`;
        pill.textContent = value;
        cell.appendChild(pill);
      } else {
        cell.textContent = value;
      }
      row.appendChild(cell);
    }
    requestLogRows.appendChild(row);
  });
}

function requestLogCellValue(record, columnId) {
  if (columnId === 'ts') return formatTimestamp(record.ts);
  if (columnId === 'model') return modelName(record.model || '');
  if (columnId === 'endpoint') return shortEndpoint(record.endpoint || '');
  if (columnId === 'status') return record.status || 'success';
  if (columnId === 'duration') return record.duration_ms == null ? '-' : String(record.duration_ms);
  if (columnId === 'request') return requestSummary(record);
  if (columnId === 'response') return responseSummary(record);
  if (columnId === 'usage') return usageSummary(record);
  return '';
}

function formatTimestamp(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function shortEndpoint(endpoint) {
  if (endpoint === '/v1/chat/completions') return 'chat';
  if (endpoint === '/v1/embeddings') return 'embeddings';
  return endpoint || '-';
}

function requestSummary(record) {
  const request = record.request || {};
  if (Array.isArray(request.messages)) {
    const message = [...request.messages].reverse().find((item) => item?.role === 'user') || request.messages.at(-1);
    return compactText(messageContentText(message?.content), 260);
  }
  if (hasOwn(request, 'input')) return compactText(inputSummary(request.input), 260);
  return compactText(JSON.stringify(request), 260);
}

function responseSummary(record) {
  if (record.error) return compactText(record.error, 260);
  const body = record.response?.body;
  if (typeof body === 'string') return compactText(body, 260);
  if (!body || typeof body !== 'object') return '-';
  const error = body.error?.message || body.error;
  if (error) return compactText(valueText(error), 260);
  const choice = Array.isArray(body.choices) ? body.choices[0] : null;
  const message = choice?.message || choice?.delta || {};
  const content = message.content || message.reasoning_content || choice?.text;
  if (content) return compactText(valueText(content), 260);
  if (Array.isArray(body.data)) {
    const first = body.data[0] || {};
    const embedding = first.embedding;
    const dimensions = Array.isArray(embedding)
      ? embedding.length
      : embedding?.length;
    return `${body.data.length} embedding${body.data.length === 1 ? '' : 's'}${dimensions ? ` / ${dimensions} dim` : ''}`;
  }
  return compactText(JSON.stringify(body), 260);
}

function usageSummary(record) {
  const usage = record.response?.body?.usage;
  if (!usage || typeof usage !== 'object') return '-';
  return Object.entries(usage).map(([key, value]) => `${key}: ${value}`).join(', ');
}

function messageContentText(content) {
  if (typeof content === 'string') return content;
  if (content == null) return '';
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === 'string') return part;
      if (part?.type === 'text') return part.text || '';
      if (part?.type === 'image_url') return '[image]';
      if (part?.type === 'input_audio') return '[audio]';
      return valueText(part);
    }).filter(Boolean).join(' ');
  }
  return valueText(content);
}

function inputSummary(input) {
  if (typeof input === 'string') return input;
  if (Array.isArray(input)) {
    if (input.length && input.every((item) => typeof item === 'number')) {
      return `${input.length} token ids`;
    }
    return `${input.length} item${input.length === 1 ? '' : 's'}: ${compactText(input.map(valueText).join(' | '), 220)}`;
  }
  return valueText(input);
}

function valueText(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value.omitted) {
    const preview = value.preview == null ? '' : ` ${compactText(valueText(value.preview), 160)}`;
    return `[${value.omitted}, length=${value.length ?? '?'}]${preview}`;
  }
  return JSON.stringify(value);
}

function compactText(value, maxLength) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (text.length <= maxLength) return text || '-';
  return `${text.slice(0, maxLength - 1)}…`;
}

function selectRequestLogRecord(index) {
  selectedRequestLogIndex = Number.isFinite(index) ? index : -1;
  renderRequestLogTable();
  renderRequestLogDetail();
}

function renderRequestLogDetail() {
  const entry = requestLogRecords[selectedRequestLogIndex];
  requestLogDetail.hidden = !entry;
  for (const button of requestLogTabs.querySelectorAll('button[data-log-tab]')) {
    button.classList.toggle('active', button.dataset.logTab === requestLogActiveTab);
  }
  if (!entry) {
    requestLogDetailPre.textContent = '';
    return;
  }
  const record = entry.record || {};
  if (requestLogActiveTab === 'summary') {
    requestLogDetailPre.textContent = requestLogSummaryText(entry);
  } else if (requestLogActiveTab === 'request') {
    requestLogDetailPre.textContent = JSON.stringify(record.request || {}, null, 2);
  } else if (requestLogActiveTab === 'response') {
    requestLogDetailPre.textContent = JSON.stringify(record.response || {}, null, 2);
  } else {
    requestLogDetailPre.textContent = JSON.stringify(entry, null, 2);
  }
}

function requestLogSummaryText(entry) {
  const record = entry.record || {};
  return [
    `Time: ${formatTimestamp(record.ts)}`,
    `Model: ${record.model || '-'}`,
    `Endpoint: ${record.endpoint || '-'}`,
    `Status: ${record.status || '-'} (${record.response?.status_code ?? '-'})`,
    `Duration: ${record.duration_ms ?? '-'} ms`,
    `File: ${entry.file || '-'}:${entry.line || '-'}`,
    '',
    'Request',
    requestSummary(record),
    '',
    'Response',
    responseSummary(record),
    '',
    'Usage',
    usageSummary(record),
    record.error ? ['', 'Error', record.error].join('\n') : '',
  ].filter((line) => line !== '').join('\n');
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

async function setDefaultModel(modelId) {
  await runAction(async () => {
    if (!modelId) throw new Error('No model selected.');
    await api('/api/default-model', {method: 'POST', body: JSON.stringify({model: modelId})});
  }, 'default updated');
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

async function saveStartupProfile() {
  try {
    startupProfileMessage.textContent = 'working...';
    const path = startupProfilePath.value.trim();
    const payload = path ? {path} : {};
    const data = await api('/api/startup-profile', {method: 'POST', body: JSON.stringify(payload)});
    startupProfileMessage.textContent = `saved ${data.count || 0} model(s): ${data.path}`;
  } catch (err) {
    startupProfileMessage.textContent = String(err);
  }
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
  if (logModel.value && ![...logModel.options].some((option) => option.value === logModel.value)) {
    logModel.value = '';
  }
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
renderRequestLogFields();
connectLogStream();
refreshAll();
refreshRequestLogs();
setInterval(loadStatus, 5000);
