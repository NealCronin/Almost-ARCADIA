(() => {
  const qs = (selector, root = document) => root.querySelector(selector);
  const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const terminal = new Set(['completed', 'failed', 'cancelled', 'idle']);
  const csrfToken = () => document.cookie.split('; ').find((item) => item.startsWith('csrftoken='))?.split('=')[1] || qs('[name=csrfmiddlewaretoken]')?.value || '';

  qsa('[data-tabs]').forEach((tabs) => {
    const buttons = qsa('[data-tab]', tabs);
    const panels = qsa('.tab-panel', tabs);
    buttons.forEach((button) => button.addEventListener('click', () => {
      buttons.forEach((item) => {
        const active = item === button;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
      });
      panels.forEach((panel) => {
        const active = panel.id === button.dataset.tab;
        panel.classList.toggle('is-active', active);
        panel.hidden = !active;
      });
    }));
  });

  const hostListener = qs('[data-host-listener-status-url]');
  if (hostListener) {
    const stateElement = qs('[data-host-listener-state]', hostListener);
    const addressElement = qs('[data-host-listener-address]', hostListener);
    const uptimeElement = qs('[data-host-listener-uptime]', hostListener);
    const errorElement = qs('[data-host-listener-error]', hostListener);
    const form = qs('[data-host-listener-form]', hostListener);
    const save = qs('[data-host-listener-save]', hostListener);
    const renderListener = (data) => {
      const state = data.state || 'failed';
      if (stateElement) {
        stateElement.textContent = state.replace(/\b\w/g, (character) => character.toUpperCase());
        stateElement.className = `status-pill status-pill--${state}`;
      }
      if (addressElement) addressElement.textContent = `Listening on ${data.host}:${data.port}`;
      if (uptimeElement) uptimeElement.textContent = data.uptime_seconds == null ? '' : `Uptime: ${data.uptime_seconds} seconds`;
      if (errorElement) { errorElement.hidden = !data.last_error; errorElement.textContent = data.last_error || ''; }
      if (save) save.disabled = ['starting', 'restarting', 'rollback'].includes(state);
    };
    const refreshListener = async () => {
      try {
        const response = await fetch(hostListener.dataset.hostListenerStatusUrl, {headers: {'Accept': 'application/json'}});
        if (response.ok) renderListener(await response.json());
      } finally {
        window.setTimeout(refreshListener, 1200);
      }
    };
    form?.addEventListener('submit', () => { if (save) save.disabled = true; });
    refreshListener();
  }

  const modelSource = document.getElementById('llm_model_source');
  if (modelSource) {
    const sourceFields = qsa('[data-model-source]');
    const updateSource = () => sourceFields.forEach((field) => { field.hidden = field.dataset.modelSource !== modelSource.value; });
    modelSource.addEventListener('change', updateSource);
    updateSource();
  }

  const fileInput = document.getElementById('priority-map-files');
  const folderInput = document.getElementById('priority-map-folder');
  const filePanel = qs('[data-selected-files]');
  const fileList = qs('[data-file-list]');
  const fileSummary = qs('[data-file-summary]');
  const stageButton = qs('[data-stage-upload]');
  const progress = qs('[data-upload-progress]');
  const progressBar = qs('[data-upload-progress] span');
  const progressText = qs('[data-upload-progress-text]');
  const retained = qs('[data-retained-uploads]');
  let selectedFiles = [];

  const renderFiles = () => {
    if (!filePanel || !fileList || !fileSummary) return;
    filePanel.hidden = selectedFiles.length === 0;
    fileSummary.textContent = `${selectedFiles.length} file${selectedFiles.length === 1 ? '' : 's'} selected`;
    fileList.replaceChildren(...selectedFiles.slice(0, 12).map((file) => {
      const item = document.createElement('li');
      item.textContent = file.webkitRelativePath || file.name;
      return item;
    }));
  };
  const renderUploads = (uploads) => {
    if (!retained) return;
    retained.replaceChildren(...uploads.map((upload) => {
      const row = document.createElement('div');
      row.className = 'artifact-row';
      const title = document.createElement('span');
      title.textContent = `${upload.source_type}: ${upload.file_count} file${upload.file_count === 1 ? '' : 's'}, ${upload.size_bytes} bytes, ${upload.created_at}`;
      const run = document.createElement('button');
      run.type = 'button'; run.className = 'button button--ghost'; run.textContent = 'Run';
      run.addEventListener('click', () => submitRetainedUpload(upload.id, stageButton?.dataset.runUrl));
      const remove = document.createElement('button');
      remove.type = 'button'; remove.className = 'text-button'; remove.textContent = 'Delete';
      remove.addEventListener('click', async () => {
        try {
          const response = await fetch(upload.delete_url, {
            method: 'POST',
            headers: {'X-CSRFToken': csrfToken(), 'Accept': 'application/json'},
          });
          const data = await response.json();
          if (!response.ok) {
            if (progressText) { progressText.hidden = false; progressText.textContent = data.detail || 'Delete failed.'; }
            return;
          }
          loadUploads();
        } catch (_) {
          if (progressText) { progressText.hidden = false; progressText.textContent = 'Delete failed.'; }
        }
      });
      row.append(title, run, remove);
      return row;
    }));
  };
  const loadUploads = async () => {
    if (!retained) return;
    try {
      const response = await fetch(retained.dataset.uploadListUrl, {headers: {'Accept': 'application/json'}});
      if (response.ok) renderUploads((await response.json()).uploads || []);
    } catch (_) { /* Retained uploads remain available after the next refresh. */ }
  };
  const submitRetainedUpload = (uploadId, runUrl) => {
    const form = document.createElement('form');
    form.method = 'post'; form.action = runUrl;
    for (const [name, value] of [['csrfmiddlewaretoken', csrfToken()], ['upload_id', uploadId]]) {
      const input = document.createElement('input'); input.type = 'hidden'; input.name = name; input.value = value; form.appendChild(input);
    }
    document.body.appendChild(form); form.submit();
  };
  qs('[data-choose-files]')?.addEventListener('click', () => fileInput?.click());
  qs('[data-choose-folder]')?.addEventListener('click', () => folderInput?.click());
  fileInput?.addEventListener('change', () => { selectedFiles = Array.from(fileInput.files || []); renderFiles(); });
  folderInput?.addEventListener('change', () => { selectedFiles = Array.from(folderInput.files || []); renderFiles(); });
  qs('[data-clear-files]')?.addEventListener('click', () => { selectedFiles = []; if (fileInput) fileInput.value = ''; if (folderInput) folderInput.value = ''; renderFiles(); });
  stageButton?.addEventListener('click', () => {
    if (!selectedFiles.length) return;
    stageButton.disabled = true;
    progress.hidden = false; progressText.hidden = false; progressText.textContent = 'Uploading 0%';
    const data = new FormData();
    selectedFiles.forEach((file) => { data.append('files', file); data.append('relative_paths', file.webkitRelativePath || file.name); });
    const xhr = new XMLHttpRequest();
    xhr.open('POST', stageButton.dataset.uploadUrl);
    xhr.setRequestHeader('X-CSRFToken', csrfToken());
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percentage = Math.round((event.loaded / event.total) * 100);
      if (progressBar) progressBar.style.width = `${percentage}%`;
      progressText.textContent = `Uploading ${percentage}%`;
    };
    xhr.onload = () => {
      stageButton.disabled = false;
      if (xhr.status < 200 || xhr.status >= 300) { progressText.textContent = 'Upload failed.'; return; }
      selectedFiles = [];
      if (fileInput) fileInput.value = '';
      if (folderInput) folderInput.value = '';
      renderFiles();
      loadUploads();
      progressText.hidden = false;
      progressText.textContent = 'Upload staged. Choose Run when ready.';
    };
    xhr.onerror = () => { stageButton.disabled = false; progressText.textContent = 'Upload failed.'; };
    xhr.send(data);
  });
  loadUploads();

  const statusRoot = qs('[data-analysis-status-url]');
  if (statusRoot) {
    const stateElement = document.getElementById('analysis-state');
    const messageElement = document.getElementById('analysis-message');
    const framesElement = document.getElementById('analysis-frames');
    const errorElement = document.getElementById('analysis-error');
    const streamFrame = qs('[data-stream-frame]');
    const artifactList = qs('[data-artifact-list]');
    let artifactsUrl = null;
    const renderArtifacts = async (url) => {
      if (!artifactList || !url || url === artifactsUrl) return;
      artifactsUrl = url;
      const response = await fetch(url, {headers: {'Accept': 'application/json'}});
      if (!response.ok) return;
      const {artifacts = []} = await response.json();
      artifactList.replaceChildren(...artifacts.map((artifact) => {
        const row = document.createElement('article'); row.className = 'artifact-row';
        const label = document.createElement('code'); label.textContent = artifact.path;
        const view = document.createElement('a'); view.href = artifact.inline_url; view.textContent = 'View'; view.className = 'button button--ghost';
        const download = document.createElement('a'); download.href = artifact.download_url; download.textContent = 'Download'; download.className = 'button button--ghost';
        row.append(label, view, download); return row;
      }));
    };
    const updateStream = (data) => {
      if (!streamFrame) return;
      if (data.state === 'idle' && !data.run_id) {
        streamFrame.textContent = 'No in-memory run is available. Django may have restarted; previously generated artifacts remain on disk.';
        return;
      }
      if (!data.stream_url) return;
      let image = qs('img', streamFrame);
      if (!image) { image = document.createElement('img'); image.alt = 'Latest Priority Map preview'; streamFrame.replaceChildren(image); }
      if (image.dataset.streamUrl !== data.stream_url) { image.dataset.streamUrl = data.stream_url; image.src = data.stream_url; }
      image.onerror = () => { if (!terminal.has(data.state)) window.setTimeout(() => { image.src = `${data.stream_url}?reconnect=${Date.now()}`; }, 1200); };
    };
    const updateCancelButtons = (data) => qsa('[data-cancel-run]').forEach((button) => {
      button.disabled = terminal.has(data.state) || !data.run_id || data.state === 'cancelling';
      button.onclick = async () => {
        button.disabled = true;
        await fetch(`/client/priority-map/runs/${data.run_id}/cancel/`, {method: 'POST', headers: {'X-CSRFToken': csrfToken()}});
      };
    });
    const refresh = async () => {
      try {
        const response = await fetch(statusRoot.dataset.analysisStatusUrl, {headers: {'Accept': 'application/json'}});
        if (!response.ok) return;
        const data = await response.json();
        if (stateElement) { stateElement.textContent = String(data.state || 'unknown').replace(/\b\w/g, (char) => char.toUpperCase()); stateElement.className = `status-pill status-pill--${data.state || 'unknown'}`; }
        if (messageElement) messageElement.textContent = data.error || data.message || 'Ready';
        if (framesElement) framesElement.textContent = data.frames_processed ?? 0;
        if (errorElement) { errorElement.hidden = !data.error; errorElement.textContent = data.error || ''; }
        updateCancelButtons(data);
        updateStream(data);
        if (!(data.state === 'idle' && !data.run_id)) renderArtifacts(data.artifacts_url);
        if (!terminal.has(data.state)) window.setTimeout(refresh, 1200);
      } catch (_) { window.setTimeout(refresh, 2500); }
    };
    refresh();
  }

  qsa('[data-copy-text]').forEach((button) => button.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(button.dataset.copyText || ''); const previous = button.textContent; button.textContent = 'Copied'; window.setTimeout(() => { button.textContent = previous; }, 1400); } catch (_) { /* Clipboard access may be unavailable. */ }
  }));
})();
