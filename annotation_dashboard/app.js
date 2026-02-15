const fileInput = document.getElementById("fileInput");
const uploadZone = document.getElementById("uploadZone");
const uploadInfo = document.getElementById("uploadInfo");
const delimiterSelect = document.getElementById("delimiter");
const headerToggle = document.getElementById("headerToggle");
const textColumnSelect = document.getElementById("textColumn");
const startBtn = document.getElementById("startBtn");
const resumeBtn = document.getElementById("resumeBtn");
const statusPill = document.getElementById("statusPill");
const resetApp = document.getElementById("resetApp");
const splitSentencesToggle = document.getElementById("splitSentences");
const loginBtn = document.getElementById("loginBtn");
const logoutBtn = document.getElementById("logoutBtn");
const userChip = document.getElementById("userChip");
const authSignInBtn = document.getElementById("authSignInBtn");
const refreshSavedBtn = document.getElementById("refreshSavedBtn");
const savedDatasetList = document.getElementById("savedDatasetList");
const savedMeta = document.getElementById("savedMeta");

const pageAuth = document.getElementById("page-auth");
const pageUpload = document.getElementById("page-upload");
const pageAnnotate = document.getElementById("page-annotate");

const progressPercent = document.getElementById("progressPercent");
const progressCount = document.getElementById("progressCount");
const remainingCount = document.getElementById("remainingCount");
const sessionMeta = document.getElementById("sessionMeta");
const sentenceIndex = document.getElementById("sentenceIndex");
const sentenceText = document.getElementById("sentenceText");
const currentLabel = document.getElementById("currentLabel");
const queueList = document.getElementById("queueList");
const queueMeta = document.getElementById("queueMeta");
const ringProgress = document.querySelector(".ring-progress");

const autoAdvanceToggle = document.getElementById("autoAdvance");
const includeUnlabeledToggle = document.getElementById("includeUnlabeled");

const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const factBtn = document.getElementById("factBtn");
const nonFactBtn = document.getElementById("nonFactBtn");
const saveBtn = document.getElementById("saveBtn");
const exportJson = document.getElementById("exportJson");
const exportCsv = document.getElementById("exportCsv");
const backBtn = document.getElementById("backBtn");
const saveHint = document.getElementById("saveHint");

const STORAGE_PREFIX = "annotation_deck";
const CLOUD_BATCH_SIZE = 500;

const state = {
  fileName: null,
  fileHash: null,
  rawText: "",
  rawRows: [],
  headers: [],
  items: [],
  labels: {},
  cursor: 0,
  storageKey: null,
  format: "csv",
  detectedDelimiter: ",",
  sentenceMode: true,
  cloudLabels: {},
  userLabels: {},
  textColumn: "",
  savedDatasets: [],
  profileMap: {},
};

const cloud = {
  enabled: false,
  user: null,
  datasetId: null,
  datasetReady: false,
  sentencesSynced: false,
};

const config = window.APP_CONFIG || {};
const oauthProvider = (config.OAUTH_PROVIDER || "github").toLowerCase();
const oauthProviderLabel =
  oauthProvider.charAt(0).toUpperCase() + oauthProvider.slice(1);
const hasSupabase =
  window.supabase && config.SUPABASE_URL && config.SUPABASE_ANON_KEY;
const supabaseClient = hasSupabase
  ? window.supabase.createClient(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)
  : null;

function setStatus(text, tone = "idle") {
  statusPill.textContent = text;
  const colors = {
    idle: "rgba(108, 246, 255, 0.12)",
    ready: "rgba(108, 246, 255, 0.2)",
    active: "rgba(91, 139, 255, 0.25)",
    warn: "rgba(255, 184, 108, 0.3)",
  };
  statusPill.style.background = colors[tone] || colors.idle;
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getInitialsFromIdentity(email, fullName) {
  if (fullName) {
    const parts = fullName
      .split(/\s+/)
      .map((part) => part.trim())
      .filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    if (parts.length === 1) {
      return parts[0].slice(0, 2).toUpperCase();
    }
  }

  if (email) {
    const local = email.split("@")[0] || "";
    const alpha = local.replace(/[^a-zA-Z]/g, "");
    if (alpha.length >= 2) return alpha.slice(0, 2).toUpperCase();
    if (local.length >= 2) return local.slice(0, 2).toUpperCase();
    if (local.length === 1) return `${local[0]}X`.toUpperCase();
  }

  return "NA";
}

function formatDateTime(isoString) {
  if (!isoString) return "Unknown time";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return date.toLocaleString();
}

function renderSavedDatasets() {
  if (!cloud.user) {
    savedMeta.textContent = "Sign in to load saved datasets.";
    savedDatasetList.innerHTML = "";
    return;
  }

  const datasets = state.savedDatasets;
  if (!datasets.length) {
    savedMeta.textContent = "No datasets yet. Upload a file to create one.";
    savedDatasetList.innerHTML = "";
    return;
  }

  savedMeta.textContent = `${datasets.length} saved dataset(s)`;
  savedDatasetList.innerHTML = datasets
    .map((dataset) => {
      const progress = dataset.totalSentences
        ? Math.round((dataset.annotatedSentences / dataset.totalSentences) * 100)
        : 0;
      const contributorChips =
        dataset.contributorInitials && dataset.contributorInitials.length
          ? dataset.contributorInitials
              .map((initials) => `<span class="initials">${escapeHtml(initials)}</span>`)
              .join("")
          : '<span class="initials">--</span>';

      return `
        <div class="saved-item">
          <div class="saved-item-main">
            <div class="saved-item-title">${escapeHtml(dataset.fileName)}</div>
            <div class="saved-item-sub">Saved ${escapeHtml(formatDateTime(dataset.createdAt))}</div>
            <div class="saved-item-progress">${dataset.annotatedSentences}/${dataset.totalSentences} annotated (${progress}%)</div>
            <div class="saved-item-progress">Contributors: <span class="initials-row">${contributorChips}</span></div>
          </div>
          <button class="ghost open-saved-btn" data-dataset-id="${dataset.id}">Open</button>
        </div>
      `;
    })
    .join("");
}

function setUser(user) {
  cloud.user = user;
  if (!hasSupabase) {
    userChip.textContent = "Cloud not configured";
    loginBtn.disabled = true;
    authSignInBtn.disabled = true;
    logoutBtn.hidden = true;
    savedMeta.textContent = "Supabase not configured.";
    showPage(pageAuth);
    return;
  }

  if (user) {
    const email = user.email || "Signed in";
    userChip.textContent = email;
    loginBtn.hidden = true;
    logoutBtn.hidden = false;
    cloud.enabled = true;
    if (pageAuth.classList.contains("active")) {
      showPage(pageUpload);
    }
    loadSavedDatasets();
    if (state.items.length && !cloud.datasetReady) {
      syncCloudAfterLogin();
    }
  } else {
    userChip.textContent = "Not signed in";
    loginBtn.hidden = false;
    logoutBtn.hidden = true;
    cloud.enabled = false;
    state.savedDatasets = [];
    renderSavedDatasets();
    showPage(pageAuth);
  }
}

async function syncCloudAfterLogin() {
  if (!state.items.length) return;
  setStatus("Syncing cloud", "active");
  await ensureDataset(state.textColumn);
  await syncSentencesToCloud();
  await syncLocalLabelsToCloud();
  await loadCloudLabels();
  await loadSavedDatasets();
  render();
  setStatus("Cloud synced", "ready");
}

async function initCloud() {
  if (!hasSupabase) {
    setUser(null);
    return;
  }

  loginBtn.textContent = `Sign in with ${oauthProviderLabel}`;
  loginBtn.title = `Sign in with ${oauthProviderLabel}`;
  authSignInBtn.textContent = `Sign in with ${oauthProviderLabel}`;

  const { data } = await supabaseClient.auth.getSession();
  setUser(data.session?.user || null);

  supabaseClient.auth.onAuthStateChange((_event, session) => {
    setUser(session?.user || null);
  });
}

async function signIn() {
  if (!supabaseClient) return;
  await supabaseClient.auth.signInWithOAuth({
    provider: oauthProvider,
    options: {
      redirectTo: window.location.origin,
    },
  });
}

async function signOut() {
  if (!supabaseClient) return;
  await supabaseClient.auth.signOut();
}

function showPage(page) {
  [pageAuth, pageUpload, pageAnnotate].forEach((el) => el.classList.remove("active"));
  page.classList.add("active");
}

function hashString(input) {
  let hash = 5381;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 33) ^ input.charCodeAt(i);
  }
  return (hash >>> 0).toString(16);
}

function detectDelimiter(line) {
  const candidates = [",", "\t", ";", "|"];
  let best = ",";
  let maxCount = -1;
  candidates.forEach((candidate) => {
    const count = line.split(candidate).length - 1;
    if (count > maxCount) {
      maxCount = count;
      best = candidate;
    }
  });
  return best;
}

function parseCSV(text, delimiter) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }

    if (char === delimiter) {
      row.push(field);
      field = "";
      continue;
    }

    if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }

    if (char === "\r") {
      continue;
    }

    field += char;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function parseJSONL(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length);
  lines.forEach((line, index) => {
    try {
      rows.push(JSON.parse(line));
    } catch (err) {
      throw new Error(`Invalid JSON on line ${index + 1}`);
    }
  });
  return rows;
}

function splitIntoSentences(text) {
  const cleaned = String(text || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) return [];

  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
    return Array.from(segmenter.segment(cleaned))
      .map((segment) => segment.segment.trim())
      .filter((segment) => segment.length);
  }

  const fallback = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
  return fallback.map((part) => part.trim()).filter((part) => part.length);
}

function buildItemsFromCSV(rows, useHeader, textColumnName) {
  if (!rows.length) return [];
  const headerRow = useHeader ? rows[0] : rows[0].map((_, idx) => `col_${idx + 1}`);
  const dataRows = useHeader ? rows.slice(1) : rows;
  const headers = headerRow.map((h) => (h || "").trim() || "column");

  state.headers = headers;
  state.rawRows = dataRows.map((row) => {
    const obj = {};
    headers.forEach((header, idx) => {
      obj[header] = row[idx] ?? "";
    });
    return obj;
  });

  if (!state.sentenceMode) {
    const items = state.rawRows.map((row, idx) => {
      const sentence = row[textColumnName] ?? "";
      return {
        id: idx + 1,
        text: sentence,
        data: {
          ...row,
          sentence,
          sentence_index: 1,
          sentence_count: 1,
          source_row: idx + 1,
        },
      };
    });
    if (!state.headers.includes("sentence")) {
      state.headers = [...state.headers, "sentence", "sentence_index", "sentence_count", "source_row"];
    }
    return items;
  }

  const items = [];
  state.rawRows.forEach((row, idx) => {
    const sentences = splitIntoSentences(row[textColumnName] ?? "");
    sentences.forEach((sentence, sentenceIndex) => {
      items.push({
        id: items.length + 1,
        text: sentence,
        data: {
          ...row,
          sentence,
          sentence_index: sentenceIndex + 1,
          sentence_count: sentences.length,
          source_row: idx + 1,
        },
      });
    });
  });

  if (!state.headers.includes("sentence")) {
    state.headers = [...state.headers, "sentence", "sentence_index", "sentence_count", "source_row"];
  }

  return items;
}

function buildItemsFromJSONL(rows, textColumnName) {
  state.headers = Object.keys(rows[0] || {});
  state.rawRows = rows;

  if (!state.sentenceMode) {
    const items = rows.map((row, idx) => {
      const sentence = row[textColumnName] ?? "";
      return {
        id: idx + 1,
        text: sentence,
        data: {
          ...row,
          sentence,
          sentence_index: 1,
          sentence_count: 1,
          source_row: idx + 1,
        },
      };
    });
    if (!state.headers.includes("sentence")) {
      state.headers = [...state.headers, "sentence", "sentence_index", "sentence_count", "source_row"];
    }
    return items;
  }

  const items = [];
  rows.forEach((row, idx) => {
    const sentences = splitIntoSentences(row[textColumnName] ?? "");
    sentences.forEach((sentence, sentenceIndex) => {
      items.push({
        id: items.length + 1,
        text: sentence,
        data: {
          ...row,
          sentence,
          sentence_index: sentenceIndex + 1,
          sentence_count: sentences.length,
          source_row: idx + 1,
        },
      });
    });
  });

  if (!state.headers.includes("sentence")) {
    state.headers = [...state.headers, "sentence", "sentence_index", "sentence_count", "source_row"];
  }

  return items;
}

function updateColumnOptions(columns) {
  textColumnSelect.innerHTML = "";
  columns.forEach((col) => {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    textColumnSelect.appendChild(option);
  });
  textColumnSelect.disabled = columns.length === 0;
  if (columns.length > 0) {
    const preferred = columns.find((col) =>
      ["sentence", "text", "document_text", "content"].includes(col.toLowerCase())
    );
    if (preferred) {
      textColumnSelect.value = preferred;
    }
  }
}

function loadSavedProgress() {
  if (!state.storageKey) return null;
  const raw = localStorage.getItem(state.storageKey);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (err) {
    return null;
  }
}

function persistProgress() {
  if (!state.storageKey) return;
  const payload = {
    labels: state.labels,
    cursor: state.cursor,
    updatedAt: new Date().toISOString(),
    fileName: state.fileName,
  };
  localStorage.setItem(state.storageKey, JSON.stringify(payload));
  saveHint.textContent = `Saved ${new Date().toLocaleTimeString()}`;
}

function clearProgress() {
  if (state.storageKey) {
    localStorage.removeItem(state.storageKey);
  }
  state.labels = {};
  state.cloudLabels = {};
  state.userLabels = {};
  state.cursor = 0;
}

function labelToDisplay(label) {
  if (!label) return "Unlabeled";
  return label === "fact" ? "Fact" : label === "non-fact" ? "Non-Fact" : label;
}

function getLabelEntry(itemId) {
  if (state.cloudLabels[itemId]) return { ...state.cloudLabels[itemId], source: "cloud" };
  if (state.labels[itemId]) return { label: state.labels[itemId], source: "local" };
  return null;
}

function getEffectiveLabel(itemId) {
  const entry = getLabelEntry(itemId);
  return entry ? entry.label : "";
}

function countEffectiveLabels() {
  const ids = new Set();
  Object.keys(state.labels).forEach((id) => ids.add(id));
  Object.keys(state.cloudLabels).forEach((id) => ids.add(id));
  return ids.size;
}

function updateProgressUI() {
  const total = state.items.length;
  const labeled = countEffectiveLabels();
  const remaining = total - labeled;
  const percent = total ? Math.round((labeled / total) * 100) : 0;

  progressPercent.textContent = `${percent}%`;
  progressCount.textContent = `${labeled} / ${total}`;
  remainingCount.textContent = `${remaining}`;

  const circumference = 2 * Math.PI * 52;
  const offset = circumference - (percent / 100) * circumference;
  ringProgress.style.strokeDasharray = `${circumference}`;
  ringProgress.style.strokeDashoffset = `${offset}`;
}

function updateSentenceUI() {
  if (!state.items.length) {
    sentenceText.textContent = "Upload a dataset to begin.";
    sentenceIndex.textContent = "0 / 0";
    currentLabel.textContent = "Unlabeled";
    return;
  }

  const item = state.items[state.cursor];
  sentenceText.textContent = item.text || "(empty)";
  sentenceIndex.textContent = `${state.cursor + 1} / ${state.items.length}`;
  const entry = getLabelEntry(item.id);
  const pretty = labelToDisplay(entry?.label);
  const suffix =
    entry?.source === "cloud" && entry?.user_initials
      ? ` • ${entry.user_initials}`
      : entry?.source === "local"
      ? " • local"
      : "";
  currentLabel.textContent = `${pretty}${suffix}`;
}

function updateQueueUI() {
  if (!state.items.length) {
    queueList.innerHTML = "";
    queueMeta.textContent = "0 items";
    return;
  }

  const windowSize = 8;
  const start = Math.max(state.cursor - 2, 0);
  const end = Math.min(state.items.length, start + windowSize);
  const viewItems = state.items.slice(start, end);

  queueList.innerHTML = "";
  viewItems.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "queue-item" + (item.id - 1 === state.cursor ? " active" : "");
    const entry = getLabelEntry(item.id);
    const labelText = entry
      ? `[${labelToDisplay(entry.label)}${entry.user_initials ? ` ${entry.user_initials}` : ""}] `
      : "";
    div.textContent = `${labelText}${item.text.slice(0, 120)}`;
    div.addEventListener("click", () => {
      state.cursor = item.id - 1;
      render();
    });
    queueList.appendChild(div);
  });

  queueMeta.textContent = `${state.items.length} items`;
}

function render() {
  updateProgressUI();
  updateSentenceUI();
  updateQueueUI();
}

function setLabel(label) {
  const item = state.items[state.cursor];
  if (!item) return;
  state.labels[item.id] = label;
  state.userLabels[item.id] = label;
  persistProgress();
  if (cloud.enabled && cloud.datasetReady) {
    saveLabelToCloud(item, label);
  }
  if (autoAdvanceToggle.checked) {
    state.cursor = Math.min(state.cursor + 1, state.items.length - 1);
  }
  render();
}

function nextSentence() {
  state.cursor = Math.min(state.cursor + 1, state.items.length - 1);
  render();
}

function prevSentence() {
  state.cursor = Math.max(state.cursor - 1, 0);
  render();
}

function exportData(type) {
  const includeUnlabeled = includeUnlabeledToggle.checked;
  const rows = state.items
    .filter((item) => includeUnlabeled || getEffectiveLabel(item.id))
    .map((item) => ({
      ...item.data,
      label: getEffectiveLabel(item.id) || "",
    }));

  if (type === "json") {
    const blob = new Blob([JSON.stringify(rows, null, 2)], {
      type: "application/json",
    });
    downloadBlob(blob, `annotations_${state.fileHash}.json`);
    return;
  }

  const headers = [...state.headers, "label"];
  const csv = toCSV(headers, rows, ",");
  const blob = new Blob([csv], { type: "text/csv" });
  downloadBlob(blob, `annotations_${state.fileHash}.csv`);
}

function toCSV(headers, rows, delimiter) {
  const escape = (value) => {
    const stringValue = value === undefined || value === null ? "" : String(value);
    const needsQuote =
      stringValue.includes('"') ||
      stringValue.includes("\n") ||
      stringValue.includes(delimiter);
    if (!needsQuote) return stringValue;
    return `"${stringValue.replace(/"/g, '""')}"`;
  };

  const lines = [];
  lines.push(headers.map(escape).join(delimiter));
  rows.forEach((row) => {
    const line = headers.map((header) => escape(row[header] ?? "")).join(delimiter);
    lines.push(line);
  });
  return lines.join("\n");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function buildSentenceRows(datasetId) {
  return state.items.map((item) => ({
    dataset_id: datasetId,
    item_index: item.id,
    source_row: item.data?.source_row ?? item.id,
    sentence_index: item.data?.sentence_index ?? 1,
    sentence_count: item.data?.sentence_count ?? 1,
    text: item.text,
  }));
}

async function loadProfiles(profileIds) {
  if (!supabaseClient || !profileIds.length) return {};
  const { data, error } = await supabaseClient
    .from("profiles")
    .select("id,email,full_name")
    .in("id", profileIds);
  if (error) {
    console.error(error);
    return {};
  }
  const map = {};
  data.forEach((profile) => {
    map[profile.id] = profile;
  });
  return map;
}

async function loadSavedDatasets() {
  if (!supabaseClient || !cloud.user) return;

  savedMeta.textContent = "Loading saved datasets...";
  savedDatasetList.innerHTML = "";

  const { data: datasets, error } = await supabaseClient
    .from("datasets")
    .select("id,hash,file_name,text_column,sentence_mode,created_at")
    .order("created_at", { ascending: false })
    .limit(20);

  if (error) {
    console.error(error);
    savedMeta.textContent = "Failed to load saved datasets.";
    return;
  }

  const datasetRows = datasets || [];
  const stats = await Promise.all(
    datasetRows.map(async (dataset) => {
      const [sentenceCountRes, annotationRowsRes] = await Promise.all([
        supabaseClient
          .from("sentences")
          .select("*", { count: "exact", head: true })
          .eq("dataset_id", dataset.id),
        supabaseClient
          .from("annotations")
          .select("item_index,user_id")
          .eq("dataset_id", dataset.id),
      ]);

      const sentenceCount = sentenceCountRes.count || 0;
      const annotationRows = annotationRowsRes.data || [];
      const annotatedSet = new Set(annotationRows.map((row) => row.item_index));
      const contributorIds = [...new Set(annotationRows.map((row) => row.user_id).filter(Boolean))];

      return {
        id: dataset.id,
        hash: dataset.hash,
        fileName: dataset.file_name || `Dataset ${dataset.id.slice(0, 8)}`,
        textColumn: dataset.text_column || "sentence",
        sentenceMode: dataset.sentence_mode,
        createdAt: dataset.created_at,
        totalSentences: sentenceCount,
        annotatedSentences: annotatedSet.size,
        contributorIds,
      };
    })
  );

  const allContributorIds = [...new Set(stats.flatMap((dataset) => dataset.contributorIds))];
  state.profileMap = await loadProfiles(allContributorIds);
  state.savedDatasets = stats.map((dataset) => ({
    ...dataset,
    contributorInitials: dataset.contributorIds
      .map((id) => {
        const profile = state.profileMap[id];
        return getInitialsFromIdentity(profile?.email, profile?.full_name);
      })
      .filter((value, index, arr) => arr.indexOf(value) === index)
      .slice(0, 5),
  }));

  renderSavedDatasets();
}

async function openDatasetFromCloud(datasetId) {
  if (!supabaseClient || !cloud.user) return;

  setStatus("Loading dataset", "active");
  const { data: dataset, error: datasetError } = await supabaseClient
    .from("datasets")
    .select("id,hash,file_name,text_column,sentence_mode,created_at")
    .eq("id", datasetId)
    .single();

  if (datasetError || !dataset) {
    console.error(datasetError);
    setStatus("Load failed", "warn");
    return;
  }

  const { data: sentences, error: sentenceError } = await supabaseClient
    .from("sentences")
    .select("item_index,source_row,sentence_index,sentence_count,text")
    .eq("dataset_id", datasetId)
    .order("item_index", { ascending: true });

  if (sentenceError || !sentences || !sentences.length) {
    console.error(sentenceError);
    setStatus("No sentence data", "warn");
    return;
  }

  state.fileName = dataset.file_name || `Dataset ${datasetId.slice(0, 8)}`;
  state.fileHash = dataset.hash || datasetId;
  state.storageKey = `${STORAGE_PREFIX}:${state.fileHash}`;
  state.textColumn = dataset.text_column || "sentence";
  state.sentenceMode = Boolean(dataset.sentence_mode);
  state.format = "cloud";
  state.headers = ["sentence", "sentence_index", "sentence_count", "source_row"];
  state.rawRows = [];
  state.rawText = "";
  state.items = sentences.map((row) => ({
    id: row.item_index,
    text: row.text,
    data: {
      sentence: row.text,
      sentence_index: row.sentence_index ?? 1,
      sentence_count: row.sentence_count ?? 1,
      source_row: row.source_row ?? row.item_index,
    },
  }));
  state.labels = {};
  state.cloudLabels = {};
  state.userLabels = {};
  state.cursor = 0;

  cloud.datasetId = datasetId;
  cloud.datasetReady = true;
  cloud.sentencesSynced = true;

  const saved = loadSavedProgress();
  if (saved) {
    state.labels = saved.labels || {};
    state.cursor = Math.min(saved.cursor || 0, state.items.length - 1);
  }

  await loadCloudLabels();

  const uniqueSourceRows = new Set(state.items.map((item) => item.data.source_row));
  sessionMeta.innerHTML = `${state.fileName}<br>${uniqueSourceRows.size} docs -> ${state.items.length} sentences`;
  showPage(pageAnnotate);
  render();
  setStatus("Dataset loaded", "ready");
}

async function ensureDataset(textColumn) {
  if (!supabaseClient || !cloud.user) return;

  const payload = {
    hash: state.fileHash,
    file_name: state.fileName,
    text_column: textColumn,
    sentence_mode: state.sentenceMode,
    created_by: cloud.user.id,
  };

  const { data, error } = await supabaseClient
    .from("datasets")
    .upsert(payload, { onConflict: "hash" })
    .select("id,hash")
    .single();

  if (error) {
    setStatus("Cloud error", "warn");
    console.error(error);
    return;
  }

  cloud.datasetId = data.id;
  cloud.datasetReady = true;
}

async function syncSentencesToCloud() {
  if (!supabaseClient || !cloud.datasetReady || cloud.sentencesSynced) return;

  const rows = buildSentenceRows(cloud.datasetId);
  for (let i = 0; i < rows.length; i += CLOUD_BATCH_SIZE) {
    const slice = rows.slice(i, i + CLOUD_BATCH_SIZE);
    const { error } = await supabaseClient
      .from("sentences")
      .upsert(slice, { onConflict: "dataset_id,item_index" });
    if (error) {
      setStatus("Cloud sync error", "warn");
      console.error(error);
      return;
    }
  }

  cloud.sentencesSynced = true;
}

async function loadCloudLabels() {
  if (!supabaseClient || !cloud.datasetReady) return;

  const { data, error } = await supabaseClient
    .from("annotations")
    .select("item_index,label,updated_at,user_id")
    .eq("dataset_id", cloud.datasetId);

  if (error) {
    setStatus("Cloud labels error", "warn");
    console.error(error);
    return;
  }

  const profileIds = [...new Set(data.map((row) => row.user_id).filter(Boolean))];
  const profileMap = await loadProfiles(profileIds);
  state.profileMap = { ...state.profileMap, ...profileMap };

  const latest = {};
  const mine = {};
  data.forEach((row) => {
    const profile = profileMap[row.user_id];
    const userEmail = profile?.email;
    const userInitials = getInitialsFromIdentity(userEmail, profile?.full_name);
    if (row.user_id === cloud.user?.id) {
      mine[row.item_index] = row.label;
    }
    const prev = latest[row.item_index];
    if (!prev || new Date(row.updated_at) > new Date(prev.updated_at)) {
      latest[row.item_index] = {
        label: row.label,
        user_id: row.user_id,
        user_email: userEmail,
        user_initials: userInitials,
        updated_at: row.updated_at,
      };
    }
  });

  state.cloudLabels = latest;
  state.userLabels = mine;
}

async function syncLocalLabelsToCloud() {
  if (!supabaseClient || !cloud.datasetReady || !cloud.user) return;
  const entries = Object.entries(state.labels);
  if (!entries.length) return;

  const payload = entries.map(([itemId, label]) => ({
    dataset_id: cloud.datasetId,
    item_index: Number(itemId),
    label,
    user_id: cloud.user.id,
  }));

  for (let i = 0; i < payload.length; i += CLOUD_BATCH_SIZE) {
    const slice = payload.slice(i, i + CLOUD_BATCH_SIZE);
    const { error } = await supabaseClient
      .from("annotations")
      .upsert(slice, { onConflict: "dataset_id,item_index,user_id" });
    if (error) {
      setStatus("Cloud sync error", "warn");
      console.error(error);
      return;
    }
  }
}

async function saveLabelToCloud(item, label) {
  if (!supabaseClient || !cloud.datasetReady || !cloud.user) return;

  const payload = {
    dataset_id: cloud.datasetId,
    item_index: item.id,
    label,
    user_id: cloud.user.id,
  };

  const { error } = await supabaseClient
    .from("annotations")
    .upsert(payload, { onConflict: "dataset_id,item_index,user_id" });

  if (error) {
    setStatus("Cloud save error", "warn");
    console.error(error);
    return;
  }

  state.cloudLabels[item.id] = {
    label,
    user_id: cloud.user.id,
    user_email: cloud.user.email,
    user_initials: getInitialsFromIdentity(cloud.user.email, cloud.user.user_metadata?.full_name),
    updated_at: new Date().toISOString(),
  };
}

function handleFile(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const rawText = reader.result;
      state.rawText = rawText;
      state.fileName = file.name;
      state.fileHash = hashString(rawText);
      state.storageKey = `${STORAGE_PREFIX}:${state.fileHash}`;
      cloud.datasetId = null;
      cloud.datasetReady = false;
      cloud.sentencesSynced = false;

      const extension = file.name.toLowerCase();
      const isJSONL = extension.endsWith(".jsonl") || extension.endsWith(".json");
      state.format = isJSONL ? "jsonl" : "csv";

      if (isJSONL) {
        const rows = parseJSONL(rawText);
        if (!rows.length) {
          throw new Error("No JSON rows detected.");
        }
        const columns = Object.keys(rows[0] || {});
        updateColumnOptions(columns);
        state.rawRows = rows;
        delimiterSelect.disabled = true;
        headerToggle.disabled = true;
      } else {
        delimiterSelect.disabled = false;
        headerToggle.disabled = false;
        const rawLines = rawText.split(/\r?\n/).filter((line) => line.trim().length);
        const firstLine = rawLines[0] || "";
        const delimiter = detectDelimiter(firstLine);
        state.detectedDelimiter = delimiter;
        const effectiveDelimiter =
          delimiterSelect.value === "auto" ? delimiter : delimiterSelect.value;
        const rows = parseCSV(rawText, effectiveDelimiter);
        state.rawRows = rows;
        const useHeader = headerToggle.value === "yes";
        const headerRow = useHeader ? rows[0] : rows[0] || [];
        const headers = useHeader
          ? headerRow
          : headerRow.map((_, idx) => `col_${idx + 1}`);
        updateColumnOptions(headers);
      }

      uploadInfo.textContent = `${file.name} loaded`;
      startBtn.disabled = false;

      const saved = loadSavedProgress();
      resumeBtn.disabled = !saved;
      setStatus(saved ? "Saved session found" : "Ready", saved ? "warn" : "ready");
    } catch (err) {
      uploadInfo.textContent = `Error: ${err.message}`;
      startBtn.disabled = true;
      resumeBtn.disabled = true;
      setStatus("Error", "warn");
    }
  };
  reader.readAsText(file);
}

function refreshColumns() {
  if (!state.rawText || state.format !== "csv") return;
  const rawLines = state.rawText.split(/\r?\n/).filter((line) => line.trim().length);
  const firstLine = rawLines[0] || "";
  state.detectedDelimiter = detectDelimiter(firstLine);
  const delimiter =
    delimiterSelect.value === "auto" ? state.detectedDelimiter : delimiterSelect.value;
  const rows = parseCSV(state.rawText, delimiter);
  const useHeader = headerToggle.value === "yes";
  const headerRow = useHeader ? rows[0] : rows[0] || [];
  const headers = useHeader
    ? headerRow
    : headerRow.map((_, idx) => `col_${idx + 1}`);
  updateColumnOptions(headers);
}

async function startSession(resume) {
  if (!cloud.user) {
    setStatus("Sign in required", "warn");
    showPage(pageAuth);
    return;
  }

  const textColumn = textColumnSelect.value;
  if (!textColumn) return;

  state.sentenceMode = splitSentencesToggle.checked;
  state.textColumn = textColumn;

  if (state.format === "jsonl") {
    state.items = buildItemsFromJSONL(state.rawRows, textColumn);
  } else {
    const delimiter =
      delimiterSelect.value === "auto"
        ? state.detectedDelimiter
        : delimiterSelect.value;
    const rows = parseCSV(state.rawText, delimiter);
    const useHeader = headerToggle.value === "yes";
    state.items = buildItemsFromCSV(rows, useHeader, textColumn);
  }

  if (!state.items.length) {
    uploadInfo.textContent = "No rows found.";
    return;
  }

  state.labels = {};
  state.cursor = 0;
  if (resume) {
    const saved = loadSavedProgress();
    if (saved) {
      state.labels = saved.labels || {};
      state.cursor = saved.cursor || 0;
    }
  } else {
    clearProgress();
  }

  const baseRows = state.rawRows.length;
  const metaSuffix = state.sentenceMode
    ? `${baseRows} docs → ${state.items.length} sentences`
    : `${state.items.length} rows`;
  sessionMeta.innerHTML = `${state.fileName}<br>${metaSuffix}`;
  setStatus("Annotating", "active");
  showPage(pageAnnotate);
  render();

  if (cloud.enabled) {
    setStatus("Syncing cloud", "active");
    await ensureDataset(textColumn);
    await syncSentencesToCloud();
    await syncLocalLabelsToCloud();
    await loadCloudLabels();
    render();
    await loadSavedDatasets();
    setStatus("Cloud synced", "ready");
  }
}

fileInput.addEventListener("change", (event) => {
  if (!cloud.user) {
    setStatus("Sign in required", "warn");
    showPage(pageAuth);
    return;
  }
  const file = event.target.files[0];
  if (file) {
    handleFile(file);
  }
});

uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("drag");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag");
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("drag");
  if (!cloud.user) {
    setStatus("Sign in required", "warn");
    showPage(pageAuth);
    return;
  }
  const file = event.dataTransfer.files[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    handleFile(file);
  }
});

startBtn.addEventListener("click", () => startSession(false));
resumeBtn.addEventListener("click", () => startSession(true));

delimiterSelect.addEventListener("change", refreshColumns);
headerToggle.addEventListener("change", refreshColumns);

loginBtn.addEventListener("click", () => {
  signIn();
});
authSignInBtn.addEventListener("click", () => {
  signIn();
});
logoutBtn.addEventListener("click", () => {
  signOut();
});
refreshSavedBtn.addEventListener("click", () => {
  loadSavedDatasets();
});
savedDatasetList.addEventListener("click", (event) => {
  const button = event.target.closest(".open-saved-btn");
  if (!button) return;
  const datasetId = button.dataset.datasetId;
  if (!datasetId) return;
  openDatasetFromCloud(datasetId);
});

resetApp.addEventListener("click", () => {
  clearProgress();
  state.items = [];
  state.rawRows = [];
  state.headers = [];
  state.rawText = "";
  state.fileName = null;
  state.fileHash = null;
  state.storageKey = null;
  state.format = "csv";
  state.sentenceMode = true;
  uploadInfo.textContent = "No file selected";
  textColumnSelect.innerHTML = "<option>Upload a file first</option>";
  textColumnSelect.disabled = true;
  startBtn.disabled = true;
  resumeBtn.disabled = true;
  splitSentencesToggle.checked = true;
  delimiterSelect.disabled = false;
  headerToggle.disabled = false;
  cloud.datasetId = null;
  cloud.datasetReady = false;
  cloud.sentencesSynced = false;
  showPage(cloud.user ? pageUpload : pageAuth);
  setStatus("Idle", "idle");
});

prevBtn.addEventListener("click", prevSentence);
nextBtn.addEventListener("click", nextSentence);

factBtn.addEventListener("click", () => setLabel("fact"));
nonFactBtn.addEventListener("click", () => setLabel("non-fact"));

saveBtn.addEventListener("click", () => {
  persistProgress();
});

exportJson.addEventListener("click", () => exportData("json"));
exportCsv.addEventListener("click", () => exportData("csv"));
backBtn.addEventListener("click", () => {
  showPage(cloud.user ? pageUpload : pageAuth);
  setStatus("Ready", "ready");
});

window.addEventListener("keydown", (event) => {
  if (pageAnnotate.classList.contains("active")) {
    if (event.key.toLowerCase() === "f") {
      setLabel("fact");
    }
    if (event.key.toLowerCase() === "n") {
      setLabel("non-fact");
    }
    if (event.key === "ArrowRight") {
      nextSentence();
    }
    if (event.key === "ArrowLeft") {
      prevSentence();
    }
  }
});

setStatus("Idle", "idle");
initCloud();
