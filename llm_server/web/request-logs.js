const apiKey = document.getElementById('apiKey');
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
let requestLogRecords = [];
let requestLogNextOffset = null;
let selectedRequestLogIndex = -1;
let requestLogActiveTab = 'summary';
let requestLogSearchTimer = null;
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
apiKey.addEventListener('input', () => {
  localStorage.setItem('proxyApiKey', apiKey.value);
  refreshRequestLogs();
});
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

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

function modelName(modelId) {
  return modelId || '-';
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
    requestLogDetailPre.replaceChildren();
    return;
  }
  const record = entry.record || {};
  if (requestLogActiveTab === 'summary') {
    setRequestLogDetailText(requestLogSummaryText(entry));
  } else if (requestLogActiveTab === 'request') {
    setRequestLogDetailText(JSON.stringify(record.request || {}, null, 2));
  } else if (requestLogActiveTab === 'response') {
    renderRequestLogResponse(record);
  } else {
    setRequestLogDetailText(JSON.stringify(entry, null, 2));
  }
}

function setRequestLogDetailText(text) {
  const pre = document.createElement('pre');
  pre.className = 'request-log-detail-pre';
  pre.textContent = text;
  requestLogDetailPre.replaceChildren(pre);
}

function renderRequestLogResponse(record) {
  const view = document.createElement('div');
  view.className = 'request-response-view';
  const response = record.response || {};
  const body = response.body;
  const statusCode = response.status_code ?? '-';

  if (typeof body === 'string') {
    const parsed = parseJsonText(body);
    if (parsed && typeof parsed === 'object') {
      view.appendChild(responseBodyView(parsed, statusCode, record.error));
    } else {
      if (record.error) {
        view.appendChild(responseSection('Error', record.error, {tone: 'error'}));
      }
      view.appendChild(responseMetaGrid([
        ['Status', String(statusCode)],
        ['Body', `${body.length} chars`],
      ]));
      view.appendChild(responseSection('Body', body || '-'));
    }
    requestLogDetailPre.replaceChildren(view);
    return;
  }

  if (!body || typeof body !== 'object') {
    if (record.error) {
      view.appendChild(responseSection('Error', record.error, {tone: 'error'}));
    }
    view.appendChild(responseMetaGrid([
      ['Status', String(statusCode)],
      ['Body', '-'],
    ]));
    view.appendChild(responseSection('Response', JSON.stringify(response || {}, null, 2) || '-'));
    requestLogDetailPre.replaceChildren(view);
    return;
  }

  view.appendChild(responseBodyView(body, statusCode, record.error));
  requestLogDetailPre.replaceChildren(view);
}

function responseBodyView(body, statusCode, recordError = '') {
  const fragment = document.createDocumentFragment();
  const error = body.error?.message || body.error || recordError;
  const usage = body.usage;
  const choice = Array.isArray(body.choices) ? body.choices[0] : null;
  const message = choice?.message || choice?.delta || {};
  const content = firstNonEmpty(
    messageContentDetailText(message.content),
    messageContentDetailText(choice?.text),
    messageContentDetailText(body.content),
    messageContentDetailText(body.output_text),
  );
  const reasoning = firstNonEmpty(
    messageContentDetailText(message.reasoning_content),
    messageContentDetailText(choice?.reasoning_content),
    messageContentDetailText(message.reasoning),
    messageContentDetailText(body.reasoning),
  );
  const toolCalls = responseToolCalls(message, choice);
  const embedding = responseEmbeddingSummary(body);
  const finishReason = choice?.finish_reason || message.finish_reason || '-';
  const metrics = [
    ['Status', String(statusCode)],
    ['Object', body.object || '-'],
  ];
  if (choice) metrics.push(['Finish', finishReason]);
  if (usage && typeof usage === 'object') {
    metrics.push(['Prompt', String(usage.prompt_tokens ?? '-')]);
    metrics.push(['Completion', String(usage.completion_tokens ?? '-')]);
    metrics.push(['Total', String(usage.total_tokens ?? '-')]);
  }
  if (embedding) {
    metrics.push(['Embeddings', String(embedding.count)]);
    metrics.push(['Dimensions', String(embedding.dimensions || '-')]);
  }
  fragment.appendChild(responseMetaGrid(metrics));

  if (error) {
    fragment.appendChild(responseSection('Error', valueText(error), {tone: 'error'}));
  }
  if (content) {
    fragment.appendChild(responseSection('Answer', content));
  }
  if (reasoning) {
    fragment.appendChild(responseSection('Reasoning', reasoning, {collapsed: true}));
  }
  if (toolCalls.length) {
    fragment.appendChild(responseToolCallSection(toolCalls));
  }
  if (embedding) {
    fragment.appendChild(responseEmbeddingSection(embedding));
  }
  if (usage && typeof usage === 'object') {
    fragment.appendChild(responseSection('Usage', JSON.stringify(usage, null, 2), {variant: 'code'}));
  }
  if (!error && !content && !reasoning && !toolCalls.length && !embedding) {
    fragment.appendChild(responseSection('Body', JSON.stringify(body, null, 2), {variant: 'code'}));
  }
  return fragment;
}

function responseMetaGrid(items) {
  const grid = document.createElement('div');
  grid.className = 'response-meta-grid';
  for (const [label, value] of items) {
    const item = document.createElement('div');
    item.className = 'response-meta-item';
    const labelEl = document.createElement('span');
    labelEl.textContent = label;
    const valueEl = document.createElement('strong');
    valueEl.textContent = value;
    item.append(labelEl, valueEl);
    grid.appendChild(item);
  }
  return grid;
}

function responseSection(title, content, {variant = 'text', tone = '', collapsed = false} = {}) {
  const section = document.createElement(collapsed ? 'details' : 'section');
  section.className = `response-section${tone ? ` ${tone}` : ''}`;
  if (collapsed) section.open = false;
  const heading = document.createElement(collapsed ? 'summary' : 'h3');
  heading.textContent = title;
  const body = document.createElement(variant === 'code' ? 'pre' : 'div');
  body.className = variant === 'code' ? 'response-code' : 'response-text';
  body.textContent = content || '-';
  section.append(heading, body);
  return section;
}

function responseToolCallSection(toolCalls) {
  const section = document.createElement('section');
  section.className = 'response-section';
  const heading = document.createElement('h3');
  heading.textContent = 'Tool Calls';
  section.appendChild(heading);
  for (const call of toolCalls) {
    const item = document.createElement('details');
    item.className = 'response-tool-call';
    item.open = true;
    const summary = document.createElement('summary');
    summary.textContent = call.function?.name || call.name || call.type || 'tool';
    const pre = document.createElement('pre');
    pre.className = 'response-code';
    pre.textContent = formatToolCall(call);
    item.append(summary, pre);
    section.appendChild(item);
  }
  return section;
}

function responseEmbeddingSection(embedding) {
  const lines = [
    `${embedding.count} embedding${embedding.count === 1 ? '' : 's'}`,
    embedding.dimensions ? `${embedding.dimensions} dimensions` : '',
    embedding.preview.length ? `Preview: ${embedding.preview.join(', ')}` : '',
  ].filter(Boolean);
  return responseSection('Embeddings', lines.join('\n'));
}

function responseToolCalls(message, choice) {
  const calls = [];
  if (Array.isArray(message?.tool_calls)) calls.push(...message.tool_calls);
  if (Array.isArray(choice?.tool_calls)) calls.push(...choice.tool_calls);
  if (message?.function_call) calls.push(message.function_call);
  return calls;
}

function formatToolCall(call) {
  const fn = call.function || call;
  const args = fn.arguments ?? call.arguments;
  let parsedArgs = args;
  if (typeof args === 'string') {
    parsedArgs = parseJsonText(args) ?? args;
  }
  return JSON.stringify({
    id: call.id,
    type: call.type,
    name: fn.name || call.name,
    arguments: parsedArgs,
  }, null, 2);
}

function responseEmbeddingSummary(body) {
  if (!Array.isArray(body.data)) return null;
  const first = body.data[0] || {};
  if (!hasOwn(first, 'embedding')) return null;
  const embedding = first.embedding;
  const dimensions = Array.isArray(embedding) ? embedding.length : embedding?.length;
  const preview = Array.isArray(embedding)
    ? embedding.slice(0, 8).map((value) => Number.isFinite(value) ? Number(value).toPrecision(5) : String(value))
    : [];
  return {
    count: body.data.length,
    dimensions,
    preview,
  };
}

function messageContentDetailText(content) {
  if (typeof content === 'string') return content;
  if (content == null) return '';
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === 'string') return part;
      if (part?.type === 'text') return part.text || '';
      if (part?.text) return part.text;
      if (part?.type === 'image_url') return '[image]';
      if (part?.type === 'input_audio') return '[audio]';
      return valueText(part);
    }).filter(Boolean).join('\n');
  }
  return valueText(content);
}

function firstNonEmpty(...values) {
  return values.find((value) => String(value || '').trim()) || '';
}

function parseJsonText(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
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

renderRequestLogFields();
refreshRequestLogs();
