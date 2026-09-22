const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const molDisplay = document.getElementById('molDisplay');
const terminal = document.getElementById('terminal');
const cmdCursor = document.getElementById('cmdCursor');
const resultArea = document.getElementById('result-area');
const resultMsg = document.getElementById('resultMsg');
const downloadLink = document.getElementById('downloadLink');
const submitBtn = document.getElementById('submitBtn');
const helpBtn = document.getElementById('helpBtn');

const TASK_STORAGE_KEY = "domd_active_task_v1";
const TERMINAL_STATES = new Set(["SUCCESS", "PARTIAL", "ERROR", "NOT_FOUND"]);


function detectAppBasePath() {
    const currentScript = document.currentScript;
    if (currentScript && currentScript.src) {
        const scriptUrl = new URL(currentScript.src, window.location.href);
        const staticMarker = "/static/";
        const staticIndex = scriptUrl.pathname.indexOf(staticMarker);
        if (staticIndex >= 0) {
            return scriptUrl.pathname.slice(0, staticIndex + 1) || "/";
        }
    }

    const baseElement = document.querySelector('base[href]');
    const baseHref = baseElement ? baseElement.getAttribute('href') : document.baseURI;
    const baseUrl = new URL(baseHref, window.location.href);
    let basePath = baseUrl.pathname || "/";

    if (!basePath.endsWith("/")) {
        basePath = basePath.replace(/[^/]*$/, "");
    }

    return basePath || "/";
}

const APP_BASE_PATH = detectAppBasePath();

function appUrl(path) {
    if (path === null || path === undefined) return APP_BASE_PATH;

    const rawPath = String(path);
    if (/^[a-z][a-z0-9+.-]*:/i.test(rawPath) || rawPath.startsWith("//")) {
        return rawPath;
    }

    const baseWithoutTrailingSlash = APP_BASE_PATH.endsWith("/")
        ? APP_BASE_PATH.slice(0, -1)
        : APP_BASE_PATH;

    if (
        rawPath.startsWith(APP_BASE_PATH) ||
        (baseWithoutTrailingSlash && rawPath === baseWithoutTrailingSlash) ||
        (baseWithoutTrailingSlash && rawPath.startsWith(`${baseWithoutTrailingSlash}/`))
    ) {
        return rawPath;
    }

    const normalizedPath = rawPath.replace(/^\/+/, "");
    return `${APP_BASE_PATH}${normalizedPath}`;
}

let stateFiles = { mol: null };
let activeEventSource = null;
let activeTaskId = null;
let recoveryInFlight = false;

let typeQueue = [];
let isTyping = false;

async function processTypeQueue() {
    if (isTyping) return;
    isTyping = true;

    while (typeQueue.length > 0) {
        const task = typeQueue.shift();

        if (task.isAction) {
            task.action();
            continue;
        }

        const { lineContainer, fragments } = task;
        terminal.insertBefore(lineContainer, cmdCursor);

        for (let frag of fragments) {
            let text = frag.text;
            let i = 0;
            while (i < text.length) {
                let chunk = Math.floor(Math.random() * 4) + 2;
                frag.node.textContent += text.slice(i, i + chunk);
                i += chunk;
                terminal.scrollTop = terminal.scrollHeight;
                await new Promise(r => setTimeout(r, Math.floor(Math.random() * 8) + 4));
            }
        }
    }
    isTyping = false;
}

async function waitForTypeQueue() {
    while (isTyping || typeQueue.length > 0) {
        await new Promise(r => setTimeout(r, 50));
    }
}

function buildTypedFragmentFromTemplate(template) {
    const wrapper = document.createElement('span');
    wrapper.className = 'terminal-template-output';

    const source = document.createElement('div');
    source.innerHTML = template.innerHTML.trim();

    const fragments = [];

    function cloneNodeForTyping(sourceNode, targetParent) {
        if (sourceNode.nodeType === Node.TEXT_NODE) {
            const text = sourceNode.textContent || '';
            if (text.length === 0) return;

            const targetTextNode = document.createTextNode('');
            targetParent.appendChild(targetTextNode);
            fragments.push({ node: targetTextNode, text });
            return;
        }

        if (sourceNode.nodeType !== Node.ELEMENT_NODE) {
            return;
        }

        const tagName = sourceNode.tagName.toLowerCase();
        const targetElement = document.createElement(tagName);

        for (const attr of sourceNode.attributes) {
            targetElement.setAttribute(attr.name, attr.value);
        }

        targetParent.appendChild(targetElement);

        for (const child of sourceNode.childNodes) {
            cloneNodeForTyping(child, targetElement);
        }
    }

    for (const child of source.childNodes) {
        cloneNodeForTyping(child, wrapper);
    }

    return { lineContainer: wrapper, fragments };
}

function typeTemplateToTerminal(selector) {
    const template = document.querySelector(selector);
    if (!template) {
        logToTerminal('WARNING: Help template is missing.');
        return;
    }

    const typedBlock = buildTypedFragmentFromTemplate(template);
    if (typedBlock.fragments.length === 0) {
        logToTerminal('WARNING: Help template is empty.');
        return;
    }

    typeQueue.push(typedBlock);
    processTypeQueue();
}

async function showHelpDocument() {
    if (!helpBtn) return;

    helpBtn.disabled = true;
    try {
        logToTerminal('INFO: Loading embedded help document.');
        await waitForTypeQueue();
        typeTemplateToTerminal('template[data-role="help"][data-topic="main"]');
    } finally {
        helpBtn.disabled = false;
    }
}

function showModal(msg, isError = true) {
    document.getElementById('modalTitleText').textContent = isError ? "CRITICAL_ERROR" : "SYS_INFO";
    document.querySelector('.modal-title').style.background = isError
        ? "linear-gradient(90deg, #990000, #ff0033)"
        : "linear-gradient(90deg, #0f172a, #1d4ed8)";
    document.getElementById('modalMsg').textContent = msg;
    document.getElementById('modalOverlay').style.display = "flex";
}

function closeModal() {
    document.getElementById('modalOverlay').style.display = "none";
}

function generateTaskId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return `task_${window.crypto.randomUUID().replaceAll("-", "").toLowerCase()}`;
    }
    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
        const bytes = new Uint8Array(16);
        window.crypto.getRandomValues(bytes);
        const hex = Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
        return `task_${hex}`;
    }
    throw new Error("Browser cryptographic random generator is unavailable.");
}

function getStoredTask() {
    try {
        const raw = localStorage.getItem(TASK_STORAGE_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (err) {
        localStorage.removeItem(TASK_STORAGE_KEY);
        return null;
    }
}

function saveStoredTask(record) {
    const existing = getStoredTask();
    const merged = existing && existing.taskId === record.taskId
        ? { ...existing, ...record }
        : record;
    localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify({ ...merged, updatedAt: Date.now() }));
}

function updateStoredLogSeq(taskId, seq) {
    if (!Number.isFinite(seq) || seq <= 0) return;

    const existing = getStoredTask();
    if (!existing || existing.taskId !== taskId) return;

    const currentSeq = Number(existing.lastLogSeq || 0);
    if (seq > currentSeq) {
        saveStoredTask({ ...existing, lastLogSeq: seq });
    }
}

function getStoredLogSeq(taskId) {
    const existing = getStoredTask();
    if (!existing || existing.taskId !== taskId) return 0;

    const seq = Number(existing.lastLogSeq || 0);
    return Number.isFinite(seq) && seq > 0 ? seq : 0;
}

function markStoredTaskFinal(taskId, state) {
    const existing = getStoredTask();
    if (!existing || existing.taskId !== taskId) return;
    saveStoredTask({ ...existing, state, terminal: true, checked: true });
}

function clearStoredTaskIfExpired() {
    const existing = getStoredTask();
    if (!existing) return;
    const ageMs = Date.now() - (existing.createdAt || 0);
    if (ageMs > 24 * 60 * 60 * 1000 || existing.checked === true) {
        localStorage.removeItem(TASK_STORAGE_KEY);
    }
}

function logToTerminal(msg) {
    const time = new Date().toTimeString().split(' ')[0];
    const lineContainer = document.createElement('span');

    lineContainer.style.color = '#39ff14';

    let fragments = [];

    const timeNode = document.createTextNode('');
    lineContainer.appendChild(timeNode);
    fragments.push({ node: timeNode, text: `\n[${time}] ` });

    const match = msg.match(/^([\[\s]*)(ERROR|FATAL|WARNING|WARN|PARTIAL|INFO)([\s\]:\-]*)([\s\S]*)$/i);

    if (match) {
        const prePunctuation = match[1];
        const levelWord = match[2];
        const postPunctuation = match[3];
        const restOfMsg = match[4];

        let levelColor = '#39ff14';
        const upLevel = levelWord.toUpperCase();
        if (upLevel === 'ERROR' || upLevel === 'FATAL') levelColor = '#ff003c';
        else if (upLevel === 'WARNING' || upLevel === 'WARN' || upLevel === 'PARTIAL') levelColor = '#ffb000';
        else if (upLevel === 'INFO') levelColor = '#00e5ff';

        if (prePunctuation) {
            const preNode = document.createTextNode('');
            lineContainer.appendChild(preNode);
            fragments.push({ node: preNode, text: prePunctuation });
        }

        const levelSpan = document.createElement('span');
        levelSpan.style.color = levelColor;
        levelSpan.style.fontWeight = 'bold';
        lineContainer.appendChild(levelSpan);
        fragments.push({ node: levelSpan, text: levelWord });

        if (postPunctuation) {
            const postNode = document.createTextNode('');
            lineContainer.appendChild(postNode);
            fragments.push({ node: postNode, text: postPunctuation });
        }

        const restNode = document.createTextNode('');
        lineContainer.appendChild(restNode);
        fragments.push({ node: restNode, text: restOfMsg });
    } else {
        const textNode = document.createTextNode('');
        if (msg.trim().startsWith('!')) {
            const errSpan = document.createElement('span');
            errSpan.style.color = '#ff003c';
            lineContainer.appendChild(errSpan);
            fragments.push({ node: errSpan, text: msg });
        } else {
            lineContainer.appendChild(textNode);
            fragments.push({ node: textNode, text: msg });
        }
    }

    typeQueue.push({ lineContainer, fragments });
    processTypeQueue();
}

function resetResultArea() {
    resultArea.className = "";
    downloadLink.classList.remove('active');
    downloadLink.removeAttribute('href');
}

function applyTerminalStatus(taskId, statusType, options = {}) {
    const hasResult = options.hasResult !== false;
    const downloadUrl = appUrl(options.downloadUrl || `api/download/${taskId}`);

    resultArea.className = `status-${statusType.toLowerCase()}`;

    if (statusType === "SUCCESS") {
        resultMsg.textContent = "COMPUTATION COMPLETE - All results generated successfully.";
        if (!options.silentLog) logToTerminal("INFO: Process finished with exit code 0.");
    } else if (statusType === "PARTIAL") {
        resultMsg.textContent = "PARTIAL SUCCESS - Some modules failed. Review logs for warnings.";
        if (!options.silentLog) logToTerminal("WARNING: Process finished with partial errors. See debug logs.");
    } else if (statusType === "ERROR") {
        resultMsg.textContent = "FATAL ERROR - Computation failed. Download logs for details.";
        if (!options.silentLog) logToTerminal("FATAL: Process aborted. See debug_error.log.");
    } else {
        resultArea.className = "status-error";
        resultMsg.textContent = "TASK NOT FOUND - The task may have expired or never reached the queue.";
        if (!options.silentLog) logToTerminal("WARNING: Task not found or expired on the server.");
    }

    if (hasResult && statusType !== "NOT_FOUND") {
        downloadLink.href = downloadUrl;
        downloadLink.classList.add('active');
    } else {
        downloadLink.classList.remove('active');
        downloadLink.removeAttribute('href');
    }

    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }

    activeTaskId = null;
    submitBtn.disabled = false;
    markStoredTaskFinal(taskId, statusType);
}

function applyNonTerminalStatus(taskId, statusType) {
    activeTaskId = taskId;
    submitBtn.disabled = true;
    resetResultArea();

    if (statusType === "QUEUED") {
        resultMsg.textContent = "[PROCESSING] Task is queued. Waiting for compute node...";
        logToTerminal(`INFO: Recovered task ${taskId}. Current state: QUEUED.`);
    } else {
        resultMsg.textContent = "[PROCESSING] Task is running. Reconnecting to live logs...";
        logToTerminal(`INFO: Recovered task ${taskId}. Current state: RUNNING.`);
    }

    connectLogStream(taskId);
}

async function queryTaskStatus(taskId, options = {}) {
    const response = await fetch(appUrl(`api/task_status/${encodeURIComponent(taskId)}`), { method: "GET", cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const payload = await response.json();
    const state = payload.task_state || "UNKNOWN";

    if (TERMINAL_STATES.has(state)) {
        if (state === "NOT_FOUND" && options.deferNotFound === true) {
            saveStoredTask({
                taskId, state: options.keepState || "SUBMITTING", terminal: false, checked: false,
                createdAt: options.createdAt || Date.now(), lastNotFoundAt: Date.now()
            });
            if (!options.silentLog) logToTerminal("INFO: Task is not visible on the server yet. Keeping the saved id for another check.");
            return payload;
        }

        // During recovery, if this browser has already seen part of the log,
        // reconnect once to fetch only the missing tail and terminal marker.
        // If lastLogSeq is 0, do not replay the whole history automatically.
        if (options.resumeStream === true && state !== "NOT_FOUND" && getStoredLogSeq(taskId) > 0) {
            connectLogStream(taskId);
            return payload;
        }

        applyTerminalStatus(taskId, state, { hasResult: payload.has_result, downloadUrl: payload.download_url, silentLog: options.silentLog });
    } else if (state === "QUEUED" || state === "RUNNING") {
        saveStoredTask({ taskId, state, terminal: false, checked: false, createdAt: options.createdAt || Date.now() });
        if (options.resumeStream !== false) applyNonTerminalStatus(taskId, state);
    } else {
        logToTerminal(`WARNING: Server returned unknown task state: ${state}.`);
    }
    return payload;
}

async function recoverStoredTask(reason = "startup") {
    if (recoveryInFlight) return;
    clearStoredTaskIfExpired();

    const stored = getStoredTask();
    if (!stored || stored.checked === true || stored.terminal === true) return;

    recoveryInFlight = true;
    try {
        await queryTaskStatus(stored.taskId, { createdAt: stored.createdAt, resumeStream: true });
    } catch (err) {
        logToTerminal(`WARNING: Could not check saved task status yet: ${err}.`);
        saveStoredTask({ ...stored, lastCheckFailed: true });
    } finally {
        recoveryInFlight = false;
    }
}

function connectLogStream(taskId) {
    if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

    activeTaskId = taskId;

    const afterSeq = getStoredLogSeq(taskId);
    const streamUrl = appUrl(`api/stream_logs/${encodeURIComponent(taskId)}?after=${encodeURIComponent(afterSeq)}`);
    activeEventSource = new EventSource(streamUrl);

    activeEventSource.onmessage = function(event) {
        const seq = Number.parseInt(event.lastEventId || "0", 10);
        if (Number.isFinite(seq) && seq > 0) {
            updateStoredLogSeq(taskId, seq);
        }

        if (event.data.startsWith("[[DONE_")) {
            const statusType = event.data.replace("[[DONE_", "").replace("]]", "");
            applyTerminalStatus(taskId, statusType);
        } else {
            logToTerminal(event.data);
        }
    };

    activeEventSource.onerror = () => {
        logToTerminal("WARNING: SSE stream interrupted. Task id and log cursor are saved locally.");
        if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }
        resultArea.className = "";
        resultMsg.textContent = "[SYSTEM ALERT] Connection lost. Task status will be checked again when possible.";
        submitBtn.disabled = false;

        const stored = getStoredTask();
        if (stored && stored.taskId === taskId) {
            saveStoredTask({ ...stored, state: stored.state || "RUNNING", terminal: false, checked: false, lastStreamErrorAt: Date.now() });
        }
        if (navigator.onLine) {
            setTimeout(() => { queryTaskStatus(taskId, { resumeStream: true }).catch(err => logToTerminal(`WARNING: Status retry failed: ${err}.`)); }, 2000);
        }
    };
}

function processFiles(fileList) {
    Array.from(fileList).forEach(file => {
        const name = file.name.toLowerCase();
        if (name.endsWith('.pdb') || name.endsWith('.sdf')) {
            stateFiles.mol = file;
            molDisplay.textContent = file.name;
            logToTerminal(`INFO: Memory loaded ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB).`);
        } else {
            logToTerminal(`WARN: Ignored unsupported file type: ${file.name}.`);
        }
    });
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); processFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', (e) => processFiles(e.target.files));

async function uploadWithProgress(formData) {
    await waitForTypeQueue();

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', appUrl('api/upload_and_run'));

        const time = new Date().toTimeString().split(' ')[0];

        const prefixSpan = document.createElement('span');
        prefixSpan.style.color = '#39ff14'; // 纯净绿
        prefixSpan.textContent = `\n[${time}] DATA_TRANSFER: `;

        const progressSpan = document.createElement('span');
        progressSpan.style.color = '#00e5ff'; // 赛博青
        progressSpan.textContent = `[····················] 0%`;

        typeQueue.push({
            isAction: true,
            action: () => {
                terminal.insertBefore(prefixSpan, cmdCursor);
                terminal.insertBefore(progressSpan, cmdCursor);
                terminal.scrollTop = terminal.scrollHeight;
            }
        });
        processTypeQueue();

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                const filled = Math.floor(percent / 5);
                const bar = '█'.repeat(filled) + '·'.repeat(20 - filled);
                progressSpan.textContent = `[${bar}] ${percent}%`;
            }
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                progressSpan.textContent = `[████████████████████] 100% (OK)`;
                progressSpan.style.color = '#39ff14';
                try { resolve(JSON.parse(xhr.responseText)); }
                catch (err) { reject("JSON parse error"); }
            } else {
                progressSpan.textContent = `[SERVER_REJECTED]`;
                progressSpan.style.color = '#ff003c';
                reject(`HTTP ${xhr.status}`);
            }
        };

        xhr.onerror = () => {
            progressSpan.textContent = `[NETWORK_LINK_DEAD]`;
            progressSpan.style.color = '#ff003c';
            reject("Connection to server lost");
        };

        xhr.send(formData);
    });
}

function probeTaskAfterUnconfirmedUpload(taskId, createdAt, attempt = 1) {
    const maxAttempts = 3;
    const delayMs = attempt === 1 ? 2000 : 5000;
    setTimeout(() => {
        queryTaskStatus(taskId, { createdAt, resumeStream: true, deferNotFound: attempt < maxAttempts, keepState: "SUBMITTING" }).then(payload => {
            if (payload.task_state === "NOT_FOUND" && attempt < maxAttempts) probeTaskAfterUnconfirmedUpload(taskId, createdAt, attempt + 1);
        }).catch(err => {
            logToTerminal(`WARNING: Delayed task status check failed: ${err}.`);
            const stored = getStoredTask();
            if (stored && stored.taskId === taskId) saveStoredTask({ ...stored, state: "SUBMITTING", terminal: false, checked: false, lastCheckFailed: true });
        });
    }, delayMs);
}

if (helpBtn) {
    helpBtn.addEventListener('click', showHelpDocument);
}

submitBtn.addEventListener('click', async () => {
    if (!stateFiles.mol) { showModal("Missing input: provide at least one .PDB or .SDF file to proceed.", true); return; }

    let taskId;
    try { taskId = generateTaskId(); }
    catch (err) { showModal(String(err), true); return; }

    const createdAt = Date.now();
    saveStoredTask({ taskId, state: "SUBMITTING", terminal: false, checked: false, createdAt, lastLogSeq: 0 });
    activeTaskId = taskId;

    submitBtn.disabled = true;
    resetResultArea();
    resultMsg.textContent = "[PROCESSING] Uploading payload to remote cluster...";

    logToTerminal("=========================================");
    logToTerminal("INFO: INITIATING UPLOAD SEQUENCE...");
    logToTerminal(`INFO: Client task id ${taskId} saved locally before upload.`);

    const formData = new FormData();
    formData.append('task_id', taskId);
    formData.append('files', stateFiles.mol);

    const chargeVal = parseFloat(document.getElementById('chargeFactor').value);
    formData.append('params_json', JSON.stringify({
        useGMX: document.getElementById('useGMX').checked,
        useBOSS: document.getElementById('useBOSS').checked,
        useML: document.getElementById('useML').checked,
        overwrite: document.getElementById('overwrite').checked,
        run_mode: document.querySelector('input[name="run_mode"]:checked').value,
        charge_factor: isNaN(chargeVal) ? 1.0 : chargeVal
    }));

    try {
        const data = await uploadWithProgress(formData);
        if (data.status !== 'success') throw new Error(data.error || "Upload rejected by server");

        const confirmedTaskId = data.task_id || taskId;
        if (confirmedTaskId !== taskId) logToTerminal(`WARNING: Server returned a different task id: ${confirmedTaskId}. Using server id for recovery.`);

        saveStoredTask({ taskId: confirmedTaskId, state: data.task_state || "QUEUED", terminal: false, checked: false, createdAt, lastLogSeq: getStoredLogSeq(confirmedTaskId) });

        resultMsg.textContent = "[PROCESSING] Data received. Awaiting computation logs...";
        logToTerminal(`INFO: Task id ${confirmedTaskId} confirmed by server.`);
        logToTerminal("INFO: WAITING FOR COMPUTE NODE...");

        connectLogStream(confirmedTaskId);

    } catch (err) {
        logToTerminal(`WARNING: Upload did not return a confirmed response: ${err}`);
        logToTerminal(`INFO: The saved task id ${taskId} will be checked automatically.`);
        showModal("Upload confirmation was interrupted. The saved task id will be checked automatically when the server is reachable.", false);
        submitBtn.disabled = false;
        resultArea.className = "";
        resultMsg.textContent = "[SYSTEM ALERT] Upload confirmation interrupted. Saved task id will be checked.";

        const stored = getStoredTask();
        if (stored && stored.taskId === taskId) saveStoredTask({ ...stored, state: "SUBMITTING", terminal: false, checked: false, lastUploadErrorAt: Date.now() });

        if (navigator.onLine) probeTaskAfterUnconfirmedUpload(taskId, createdAt);
    }
});

window.addEventListener('online', () => { logToTerminal("INFO: Browser reports network link restored."); recoverStoredTask("network restore"); });
window.addEventListener('offline', () => { logToTerminal("WARNING: Browser reports network link lost. Active task id remains saved locally."); });
document.addEventListener('visibilitychange', () => { if (!document.hidden) recoverStoredTask("page focus"); });

clearStoredTaskIfExpired();
recoverStoredTask("page load");