(() => {
  const qs = (selector, root = document) => root.querySelector(selector);
  const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const terminal = new Set(['completed', 'failed', 'cancelled', 'idle']);
  const csrfToken = () => document.cookie.split('; ').find((item) => item.startsWith('csrftoken='))?.split('=')[1] || qs('[name=csrfmiddlewaretoken]')?.value || '';

  // Tabs
  qsa('[data-tabs]').forEach((tabs) => {
    const buttons = qsa('[data-tab]', tabs);
    const panels = qsa('[data-panel]', tabs);
    buttons.forEach((btn) => btn.addEventListener('click', () => {
      buttons.forEach((b) => b.removeAttribute('aria-selected'));
      panels.forEach((p) => p.hidden = true);
      btn.setAttribute('aria-selected', 'true');
      const panel = qs(btn.dataset.tab, tabs);
      if (panel) panel.hidden = false;
    }));
  });

  // Host listener
  const hostListener = qs('[data-host-listener-status-url]');
  if (hostListener) {
    let hostPollTimer = null;
    const hostPoll = async () => {
      try {
        const response = await fetch(hostListener.dataset.hostListenerStatusUrl, { headers: { 'Accept': 'application/json' } });
        const data = await response.json();
        const indicator = qs('[data-host-listener-indicator]');
        if (indicator) {
          indicator.textContent = data.running ? 'Running' : 'Stopped';
          indicator.className = 'status-pill' + (data.running ? ' status-pill--ready' : '');
        }
        const instructionHost = qs('[data-instruction-host]');
        if (instructionHost) instructionHost.textContent = `${data.host}:${data.port}`;
      } catch (_) { /* polling is best-effort */ }
    };
    hostPoll();
    hostPollTimer = setInterval(hostPoll, 30000);
  }

  // ======== ALL LLM forms (iterate, don't just pick first) ========
  qsa('[data-llm-form]').forEach((llmForm) => {
    const role = llmForm.dataset.role || 'llm';
    const rolePrefix = role === 'visual_llm' ? 'vis' : 'llm';

    // Node / bind-host toggling
    const nodeSelect = qs(`[name="node"]`, llmForm);
    const bindInput = qs(`[name="local_bind_host"]`, llmForm);
    const bindSection = qs('[data-local-bind-section]', llmForm);
    const remoteNotice = qs('[data-remote-bind-notice]', llmForm);

    // Find node hosts data
    const nodeHostsScript = qs(`#${role}-node-hosts`) || qs(`#${rolePrefix}-node-hosts`) ||
                            qs(`[id$="-node-hosts"]`, llmForm);
    let nodeHosts = {};
    try { nodeHosts = JSON.parse(nodeHostsScript?.textContent || '{}'); } catch (_) {}

    const updateBindHost = () => {
      const remote = nodeSelect && nodeSelect.value !== 'local';
      if (bindInput) bindInput.disabled = !!remote;
      if (bindSection) bindSection.hidden = !!remote;
      if (remoteNotice) {
        remoteNotice.hidden = !remote;
        remoteNotice.textContent = remote
          ? `Inference will listen on ${nodeHosts[nodeSelect?.value] || 'the selected remote computer'}.`
          : '';
      }
    };
    nodeSelect?.addEventListener('change', updateBindHost);
    updateBindHost();

    // Repository inspection
    qs('[data-inspect-repository]', llmForm)?.addEventListener('click', async () => {
      const advanced = qs('[data-llm-advanced]', llmForm);
      const status = qs('[data-inspect-status]', llmForm);
      if (advanced) advanced.open = true;
      if (status) status.textContent = 'Inspecting repository…';
      try {
        const hfRepo = qs(`[name="hf_repo"]`, llmForm)?.value || '';
        const mmprojRepo = qs(`[name="mmproj_repo"]`, llmForm)?.value || '';
        const draftRepo = qs(`[name="draft_repo"]`, llmForm)?.value || '';

        const response = await fetch(llmForm.dataset.inspectUrl || '', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken(), 'Accept': 'application/json' },
          body: JSON.stringify({ hf_repo: hfRepo, mmproj_repo: mmprojRepo, draft_repo: draftRepo }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Inspection failed.');

        const fill = (inputName, listId, values) => {
          const input = qs(`[name="${inputName}"]`, llmForm);
          const list = qs(`#${listId}`, llmForm);
          if (input && list && values) {
            input.setAttribute('list', list.id);
            list.replaceChildren(...values.map((v) => Object.assign(document.createElement('option'), { value: v })));
          }
        };
        fill('model_file_pattern', `${rolePrefix}-model-patterns`, data.models);
        fill('mmproj_file_pattern', `${rolePrefix}-mmproj-patterns`, data.mmproj);
        fill('draft_file_pattern', `${rolePrefix}-draft-patterns`, data.drafts);
        if (status) status.textContent = data.message || 'Repository inspected.';
      } catch (error) {
        if (status) status.textContent = error.message || 'Inspection failed.';
      }
    });

    // Vision field visibility
    const visionCheckbox = qs(`[name="vision_enabled"]`, llmForm);
    const mmprojRepo = qs(`[name="mmproj_repo"]`, llmForm);
    const mmprojPattern = qs(`[name="mmproj_file_pattern"]`, llmForm);
    const updateVisionFields = () => {
      const enabled = visionCheckbox?.checked || visionCheckbox?.value === 'true' || visionCheckbox?.disabled;
      if (mmprojRepo) mmprojRepo.closest('.form-field')?.style && (mmprojRepo.closest('.form-field').style.display = enabled ? '' : 'none');
      if (mmprojPattern) mmprojPattern.closest('.form-field')?.style && (mmprojPattern.closest('.form-field').style.display = enabled ? '' : 'none');
    };
    visionCheckbox?.addEventListener('change', updateVisionFields);
    updateVisionFields();

    // Draft field visibility
    const draftCheckbox = qs(`[name="draft_enabled"]`, llmForm);
    const draftFields = ['draft_repo', 'draft_file_pattern', 'draft_method', 'draft_max_tokens', 'draft_min_prob', 'draft_cache_type_k', 'draft_cache_type_v'];
    const updateDraftFields = () => {
      const enabled = draftCheckbox?.checked;
      draftFields.forEach((name) => {
        const field = qs(`[name="${name}"]`, llmForm);
        if (field?.closest('.form-field')?.style) {
          field.closest('.form-field').style.display = enabled ? '' : 'none';
        }
      });
    };
    draftCheckbox?.addEventListener('change', updateDraftFields);
    updateDraftFields();

    // Dirty tracking
    let isDirty = false;
    const allInputs = qsa('input, select, textarea', llmForm);
    allInputs.forEach((el) => {
      el.addEventListener('change', () => { isDirty = true; });
      el.addEventListener('input', () => { isDirty = true; });
    });

    // Test chat button
    const testBtn = qs('[data-test-chat]', llmForm);
    if (testBtn) {
      testBtn.addEventListener('click', async () => {
        if (isDirty) {
          alert('Save changes before testing.');
          return;
        }
        const originalText = testBtn.textContent;
        testBtn.textContent = 'Testing…';
        testBtn.disabled = true;

        try {
          const url = role === 'llm'
            ? '/client/priority-map/models/test-llm-chat/'
            : '/client/priority-map/models/test-visual-llm-chat/';
          const formData = new FormData();
          const promptEl = qs('[data-test-prompt]', llmForm);
          formData.append('prompt', promptEl?.value || '');
          const imageEl = qs('[data-test-image]', llmForm);
          if (imageEl?.files?.length) {
            formData.append('image', imageEl.files[0]);
          }
          const response = await fetch(url, {
            method: 'POST',
            headers: { 'X-CSRFToken': csrfToken(), 'Accept': 'application/json' },
            body: formData,
          });
          const data = await response.json();
          const resultEl = qs('[data-test-result]', llmForm);
          if (resultEl) {
            if (response.ok) {
              resultEl.textContent = data.response || JSON.stringify(data);
              resultEl.className = 'help-copy';
            } else {
              resultEl.textContent = data.error || 'Unknown error';
              resultEl.className = 'inline-error';
            }
          }
        } catch (error) {
          const resultEl = qs('[data-test-result]', llmForm);
          if (resultEl) { resultEl.textContent = error.message || 'Request failed'; resultEl.className = 'inline-error'; }
        } finally {
          testBtn.textContent = originalText;
          testBtn.disabled = false;
        }
      });
    }
  });

  // Compute nodes
  const computeNodes = qs('[data-compute-nodes]');
  if (computeNodes) {
    qsa('[data-test-node]', computeNodes).forEach((form) => {
      qs('button', form)?.addEventListener('click', async () => {
        const btn = qs('button', form);
        const original = btn.textContent;
        btn.textContent = 'Testing…';
        btn.disabled = true;
        try {
          const response = await fetch(form.getAttribute('action'), {
            method: 'POST',
            headers: { 'X-CSRFToken': csrfToken(), 'Accept': 'application/json' },
          });
          const data = await response.json();
          const nodeName = form.dataset.testNode;
          const pill = qs(`[data-node-status="${nodeName}"]`, computeNodes);
          if (pill) { pill.textContent = data.state; pill.className = `status-pill${data.state === 'reachable' ? ' status-pill--ready' : ''}`; }
        } catch (_) {
          const nodeName = form.dataset.testNode;
          const pill = qs(`[data-node-status="${nodeName}"]`, computeNodes);
          if (pill) { pill.textContent = 'error'; pill.className = 'status-pill'; }
        } finally {
          btn.textContent = original;
          btn.disabled = false;
        }
      });
    });
    qsa('[data-delete-node]', computeNodes).forEach((form) => {
      qs('button', form)?.addEventListener('click', (ev) => {
        if (!confirm('Remove this remote computer and its saved model configurations?')) {
          ev.preventDefault();
        }
      });
    });
  }

  // Visual mode selector
  const visualModeSelect = qs('[data-visual-mode-select]');
  visualModeSelect?.addEventListener('change', () => {
    visualModeSelect.closest('form')?.submit();
  });

  // File uploads
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
    if (filePanel) filePanel.hidden = !selectedFiles.length;
    if (fileList) fileList.replaceChildren(...selectedFiles.map((f) => {
      const li = document.createElement('li');
      const size = f.size > 1048576 ? `${(f.size / 1048576).toFixed(1)} MB` : `${(f.size / 1024).toFixed(0)} kB`;
      li.textContent = `${f.webkitRelativePath || f.name} (${size})`;
      return li;
    }));
    if (fileSummary) fileSummary.textContent = selectedFiles.length ? `${selectedFiles.length} file(s) selected` : '';
  };

  const renderUploads = (uploads) => {
    if (!retained) return;
    retained.replaceChildren(...uploads.map((u) => {
      const div = document.createElement('div');
      div.className = 'retained-upload';
      div.innerHTML = `<strong>${u.name}</strong> <span class="muted">${(u.size / 1048576).toFixed(1)} MB</span>`;
      const form = document.createElement('form');
      form.method = 'post';
      form.action = u.submit_url;
      form.className = 'inline-form';
      form.innerHTML = `<input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken()}"><button class="button button--primary button--small" type="submit">Run</button>`;
      div.appendChild(form);
      return div;
    }));
  };

  const loadUploads = async () => {
    try {
      const resp = await fetch('/client/priority-map/uploads/', { headers: { 'Accept': 'application/json' } });
      if (resp.ok) renderUploads((await resp.json()).uploads || []);
    } catch (_) {}
  };

  qs('[data-choose-files]')?.addEventListener('click', () => fileInput?.click());
  qs('[data-choose-folder]')?.addEventListener('click', () => folderInput?.click());
  fileInput?.addEventListener('change', () => { selectedFiles = Array.from(fileInput.files || []); renderFiles(); });
  folderInput?.addEventListener('change', () => { selectedFiles = Array.from(folderInput.files || []); renderFiles(); });
  qs('[data-clear-files]')?.addEventListener('click', () => { selectedFiles = []; if (fileInput) fileInput.value = ''; if (folderInput) folderInput.value = ''; renderFiles(); });
  stageButton?.addEventListener('click', async () => {
    if (!selectedFiles.length) return;
    if (progress) progress.hidden = false;
    const formData = new FormData();
    selectedFiles.forEach((f) => formData.append('files', f));
    try {
      const resp = await fetch('/client/priority-map/uploads/', { method: 'POST', headers: { 'X-CSRFToken': csrfToken() }, body: formData });
      const data = await resp.json();
      if (resp.ok) { selectedFiles = []; renderFiles(); loadUploads(); }
      if (progress) progress.hidden = true;
    } catch (_) { if (progress) progress.hidden = true; }
  });
  loadUploads();

  // Analysis status
  const statusRoot = qs('[data-analysis-status-url]');
  if (statusRoot) {
    let pollTimer = null;
    const poll = async () => {
      try {
        const resp = await fetch(statusRoot.dataset.analysisStatusUrl, { headers: { 'Accept': 'application/json' } });
        const state = await resp.json();
        const el = qs('[data-analysis-state]');
        if (el) { el.textContent = state.state; el.className = 'status-pill' + (state.state === 'running' ? ' status-pill--ready' : ''); }
        const msgEl = qs('[data-analysis-message]');
        if (msgEl) msgEl.textContent = state.message || '';
        const framesEl = qs('[data-analysis-frames]');
        if (framesEl) framesEl.textContent = state.frames_processed || '0';
        if (terminal.has(state.state)) {
          clearInterval(pollTimer);
          // Reload artifacts
          try {
            const artResp = await fetch('/client/priority-map/runs/' + state.run_id + '/artifacts/', { headers: { 'Accept': 'application/json' } });
            if (artResp.ok) {
              const arts = await artResp.json();
              const artList = qs('[data-artifact-list]');
              if (artList) artList.replaceChildren(...(arts.artifacts || []).map((a) => {
                const li = document.createElement('li');
                const aEl = document.createElement('a');
                aEl.href = a.download_url; aEl.textContent = a.path; aEl.download = '';
                li.appendChild(aEl);
                return li;
              }));
            }
          } catch (_) {}
        }
      } catch (_) {}
    };
    poll();
    pollTimer = setInterval(poll, 2000);
  }

  qsa('[data-copy-text]').forEach((button) => button.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(button.dataset.copyText || ''); const previous = button.textContent; button.textContent = 'Copied'; window.setTimeout(() => { button.textContent = previous; }, 1400); } catch (_) { /* Clipboard access may be unavailable. */ }
  }));
})();