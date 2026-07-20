(() => {
  const qs = (selector, root = document) => root.querySelector(selector);
  const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const csrfToken = () => document.cookie.split('; ').find((item) => item.startsWith('csrftoken='))?.split('=')[1] || qs('[name=csrfmiddlewaretoken]')?.value || '';

  const hostStatus = qs('[data-host-listener-status-url]');
  if (hostStatus) {
    const poll = async () => {
      try {
        const response = await fetch(hostStatus.dataset.hostListenerStatusUrl, { headers: { Accept: 'application/json' } });
        const data = await response.json();
        const indicator = qs('[data-host-listener-indicator]');
        if (indicator) {
          indicator.textContent = data.running ? 'Running' : 'Stopped';
          indicator.className = `status-pill${data.running ? ' status-pill--ready' : ''}`;
        }
        const address = qs('[data-instruction-host]');
        if (address) address.textContent = `${data.host}:${data.port}`;
      } catch (_) { /* status polling is best effort */ }
    };
    poll();
    setInterval(poll, 30000);
  }

  const configureNodeForm = (form, role) => {
    const nodeSelect = qs('[name="node"]', form);
    const bindInput = qs('[name="bind_host"]', form);
    let nodeHosts = {};
    try { nodeHosts = JSON.parse(document.getElementById(role)?.textContent || '{}'); } catch (_) {}
    if (role === 'sam3') {
      try { nodeHosts = JSON.parse(document.getElementById('sam3-node-hosts')?.textContent || '{}'); } catch (_) {}
    }
    const setDefaultIP = () => {
      const host = nodeHosts[nodeSelect?.value];
      if (bindInput && host) bindInput.value = host;
    };
    nodeSelect?.addEventListener('change', setDefaultIP);
  };

  const dirty = window.__arcadiaDirtyForms = window.__arcadiaDirtyForms || {};

  qsa('[data-node-service-form]').forEach((form) => {
    const role = form.dataset.role || 'sam3';
    configureNodeForm(form, role);
    dirty[role] = false;
    qsa('input:not([data-ignore-dirty]), select, textarea', form).forEach((element) => {
      element.addEventListener('input', () => { dirty[role] = true; });
      element.addEventListener('change', () => { dirty[role] = true; });
    });
  });

  qsa('[data-llm-form]').forEach((form) => {
    const role = form.dataset.role || 'llm';
    configureNodeForm(form, role);
    dirty[role] = false;

    const vision = qs('[name="vision_enabled"]', form);
    const projector = qs('[data-projector-field]', form);
    const updateVision = () => {
      if (projector) projector.hidden = !(vision?.checked || vision?.disabled);
    };
    vision?.addEventListener('change', updateVision);
    updateVision();

    qsa('input:not([data-ignore-dirty]), select, textarea', form).forEach((element) => {
      element.addEventListener('input', () => { dirty[role] = true; });
      element.addEventListener('change', () => { dirty[role] = true; });
    });

    qsa('[data-inspect-source]', form).forEach((button) => {
      button.addEventListener('click', async () => {
        const kind = button.dataset.inspectSource;
        const field = qs(kind === 'projector' ? '[name="mmproj_source"]' : '[name="hf_source"]', form);
        const status = qs(`[data-inspect-status="${kind}"]`, form);
        if (status) status.textContent = 'Inspecting…';
        try {
          const response = await fetch(form.dataset.inspectUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken(), Accept: 'application/json' },
            body: JSON.stringify({ source: field?.value || '', projector: kind === 'projector' }),
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || 'Inspection failed.');
          if (status) status.textContent = data.message;
        } catch (error) {
          if (status) status.textContent = error.message || 'Inspection failed.';
        }
      });
    });
  });

  const samSettingsForm = qs('[data-node-service-form][data-role="sam3"]');
  if (samSettingsForm) {
    const browseButton = qs('[data-sam-checkpoint-browse]', samSettingsForm);
    const fileInput = qs('[data-sam-checkpoint-file]', samSettingsForm);
    const checkpointInput = qs('[name="checkpoint"]', samSettingsForm);
    const nodeSelect = qs('[name="node"]', samSettingsForm);
    const status = qs('[data-sam-checkpoint-status]', samSettingsForm);
    let uploadedNode = '';
    let uploadedPath = '';

    const setCheckpointStatus = (message, error = false) => {
      if (!status) return;
      status.textContent = message;
      status.className = error ? 'inline-error' : 'help-copy';
    };

    browseButton?.addEventListener('click', () => fileInput?.click());

    fileInput?.addEventListener('change', async () => {
      const file = fileInput.files?.[0];
      const node = nodeSelect?.value || '';
      if (!file) return;
      if (!file.name.toLowerCase().endsWith('.pt')) {
        setCheckpointStatus('Choose a .pt checkpoint file.', true);
        fileInput.value = '';
        return;
      }
      if (!node) {
        setCheckpointStatus('Choose a compute node before browsing for a checkpoint.', true);
        fileInput.value = '';
        return;
      }

      const original = browseButton?.textContent || 'Browse…';
      if (browseButton) {
        browseButton.disabled = true;
        browseButton.textContent = 'Uploading…';
      }
      if (nodeSelect) nodeSelect.disabled = true;
      setCheckpointStatus(`Uploading ${file.name} to ${node === 'local' ? 'this computer' : node}…`);

      try {
        const body = new FormData();
        body.append('node', node);
        body.append('checkpoint', file);
        const response = await fetch(samSettingsForm.dataset.checkpointUploadUrl, {
          method: 'POST',
          headers: { 'X-CSRFToken': csrfToken(), Accept: 'application/json' },
          body,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Checkpoint upload failed.');
        uploadedNode = node;
        uploadedPath = data.checkpoint;
        if (checkpointInput) {
          checkpointInput.value = uploadedPath;
          checkpointInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
        setCheckpointStatus(`Uploaded ${data.filename} to ${node === 'local' ? 'this computer' : node}. Save SAM3 settings to use it.`);
      } catch (error) {
        setCheckpointStatus(error.message || 'Checkpoint upload failed.', true);
      } finally {
        if (browseButton) {
          browseButton.disabled = false;
          browseButton.textContent = original;
        }
        if (nodeSelect) nodeSelect.disabled = false;
        if (fileInput) fileInput.value = '';
      }
    });

    nodeSelect?.addEventListener('change', () => {
      if (!uploadedNode || nodeSelect.value === uploadedNode || checkpointInput?.value !== uploadedPath) return;
      if (checkpointInput) {
        checkpointInput.value = '';
        checkpointInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      setCheckpointStatus(`The uploaded checkpoint is stored on ${uploadedNode === 'local' ? 'this computer' : uploadedNode}. Browse again for the newly selected node.`, true);
      uploadedNode = '';
      uploadedPath = '';
    });
  }

  const hostCheckpoint = qs('[data-host-sam-checkpoint-form]');
  if (hostCheckpoint) {
    const browseButton = qs('[data-host-sam-checkpoint-browse]', hostCheckpoint);
    const fileInput = qs('[data-host-sam-checkpoint-file]', hostCheckpoint);
    const checkpointInput = qs('[name="checkpoint"]', hostCheckpoint);
    const status = qs('[data-host-sam-checkpoint-status]', hostCheckpoint);
    const setStatus = (message, error = false) => {
      if (!status) return;
      status.textContent = message;
      status.className = error ? 'inline-error' : 'help-copy';
    };

    browseButton?.addEventListener('click', () => fileInput?.click());
    fileInput?.addEventListener('change', async () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith('.pt')) {
        setStatus('Choose a .pt checkpoint file.', true);
        fileInput.value = '';
        return;
      }
      const original = browseButton?.textContent || 'Browse or upload…';
      if (browseButton) {
        browseButton.disabled = true;
        browseButton.textContent = 'Uploading…';
      }
      try {
        const body = new FormData();
        body.append('node', 'local');
        body.append('checkpoint', file);
        const response = await fetch(hostCheckpoint.dataset.checkpointUploadUrl, {
          method: 'POST',
          headers: { 'X-CSRFToken': csrfToken(), Accept: 'application/json' },
          body,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Checkpoint upload failed.');
        if (checkpointInput) checkpointInput.value = data.checkpoint;
        setStatus(`Uploaded ${data.filename}. Save checkpoint to make it the host default.`);
      } catch (error) {
        setStatus(error.message || 'Checkpoint upload failed.', true);
      } finally {
        if (browseButton) {
          browseButton.disabled = false;
          browseButton.textContent = original;
        }
        if (fileInput) fileInput.value = '';
      }
    });
  }

  qsa('[data-test-chat-component]').forEach((component) => {
    const role = component.dataset.role || 'llm';
    const button = qs('[data-test-chat]', component);
    button?.addEventListener('click', async () => {
      if (dirty[role] || (role === 'visual_llm' && dirty.llm)) {
        alert('Save model changes before testing.');
        return;
      }
      const original = button.textContent;
      button.disabled = true;
      button.textContent = 'Testing…';
      const output = qs('[data-test-result]', component);
      try {
        const body = new FormData();
        body.append('prompt', qs('[data-test-prompt]', component)?.value || '');
        const image = qs('[data-test-image]', component);
        if (role === 'visual_llm') {
          if (!image?.files?.length) throw new Error('Choose an image for visual chat.');
          body.append('image', image.files[0]);
        }
        const response = await fetch(component.dataset.testUrl, {
          method: 'POST',
          headers: { 'X-CSRFToken': csrfToken(), Accept: 'text/plain' },
          body,
        });
        const text = await response.text();
        if (!response.ok) throw new Error(text || 'Test failed.');
        if (output) {
          output.textContent = text;
          output.className = 'help-copy';
        }
      } catch (error) {
        if (output) {
          output.textContent = error.message || 'Test failed.';
          output.className = 'inline-error';
        }
      } finally {
        button.disabled = false;
        button.textContent = original;
      }
    });
  });

  qsa('[data-test-sam-component]').forEach((component) => {
    const button = qs('[data-test-sam]', component);
    const imageInput = qs('[data-test-sam-image]', component);
    const termInput = qs('[data-test-sam-term]', component);
    const status = qs('[data-test-sam-status]', component);
    const figure = qs('[data-test-sam-result]', component);
    const resultImage = qs('[data-test-sam-result-image]', component);
    const download = qs('[data-test-sam-download]', component);
    let resultUrl = '';

    const releaseResult = () => {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      resultUrl = '';
    };

    button?.addEventListener('click', async () => {
      if (dirty.sam3) {
        alert('Save SAM3 changes before testing.');
        return;
      }
      const term = termInput?.value?.trim() || '';
      if (!term) {
        if (status) status.textContent = 'Enter a search term.';
        termInput?.focus();
        return;
      }
      if (!imageInput?.files?.length) {
        if (status) status.textContent = 'Choose a JPEG, PNG, or WebP image.';
        return;
      }

      const original = button.textContent;
      button.disabled = true;
      button.textContent = 'Segmenting…';
      if (status) {
        status.textContent = 'Starting the saved SAM3 service and running segmentation…';
        status.className = 'help-copy';
      }

      try {
        const body = new FormData();
        body.append('search_term', term);
        body.append('image', imageInput.files[0]);
        const response = await fetch(component.dataset.testUrl, {
          method: 'POST',
          headers: { 'X-CSRFToken': csrfToken(), Accept: 'image/png,text/plain' },
          body,
        });
        if (!response.ok) {
          throw new Error((await response.text()) || 'SAM3 test failed.');
        }
        const blob = await response.blob();
        releaseResult();
        resultUrl = URL.createObjectURL(blob);
        if (resultImage) resultImage.src = resultUrl;
        if (figure) figure.hidden = false;
        if (download) {
          download.href = resultUrl;
          download.hidden = false;
        }
        const count = response.headers.get('X-Arcadia-Segment-Count');
        if (status) status.textContent = count ? `Rendered ${count} segment(s).` : 'Segmentation completed.';
      } catch (error) {
        if (status) {
          status.textContent = error.message || 'SAM3 test failed.';
          status.className = 'inline-error';
        }
      } finally {
        button.disabled = false;
        button.textContent = original;
      }
    });

    window.addEventListener('beforeunload', releaseResult, { once: true });
  });

  const computeNodes = qs('[data-compute-nodes]');
  if (computeNodes) {
    qsa('[data-test-node]', computeNodes).forEach((form) => {
      const button = qs('button', form);
      button?.addEventListener('click', async () => {
        const original = button.textContent;
        button.disabled = true;
        button.textContent = 'Testing…';
        try {
          const response = await fetch(form.action, {
            method: 'POST',
            headers: { 'X-CSRFToken': csrfToken(), Accept: 'application/json' },
          });
          const data = await response.json();
          const pill = qs(`[data-node-status="${form.dataset.testNode}"]`, computeNodes);
          if (pill) {
            pill.textContent = data.state;
            pill.className = `status-pill${data.state === 'reachable' ? ' status-pill--ready' : ''}`;
          }
        } finally {
          button.disabled = false;
          button.textContent = original;
        }
      });
    });
    qsa('[data-delete-node]', computeNodes).forEach((form) => {
      form.addEventListener('submit', (event) => {
        if (!confirm('Remove this compute node?')) event.preventDefault();
      });
    });
  }

  const uploadForm = qs('[data-upload-form]');
  const uploadButton = qs('[data-stage-upload]');
  uploadButton?.addEventListener('click', async () => {
    const files = qs('#priority-map-files')?.files;
    const folder = qs('#priority-map-folder')?.files;
    const selected = files?.length ? files : folder;
    const status = qs('[data-upload-status]');
    if (!selected?.length) {
      if (status) status.textContent = 'Choose files or a folder first.';
      return;
    }
    const body = new FormData();
    Array.from(selected).forEach((file) => {
      body.append('files', file);
      body.append('relative_paths', file.webkitRelativePath || file.name);
    });
    uploadButton.disabled = true;
    if (status) status.textContent = 'Uploading…';
    try {
      const response = await fetch(uploadForm.action, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken(), Accept: 'application/json' },
        body,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Upload failed.');
      if (status) status.textContent = `Uploaded ${data.upload.file_count} file(s). Reload to select it.`;
    } catch (error) {
      if (status) status.textContent = error.message || 'Upload failed.';
    } finally {
      uploadButton.disabled = false;
    }
  });

  const runPanel = qs('[data-analysis-state]');
  if (runPanel) {
    const poll = async () => {
      try {
        const response = await fetch(runPanel.dataset.statusUrl, { headers: { Accept: 'application/json' } });
        const data = await response.json();
        const state = qs('[data-run-state]');
        const frames = qs('[data-frames]');
        const error = qs('[data-error]');
        if (state) state.textContent = data.state.charAt(0).toUpperCase() + data.state.slice(1);
        if (frames) frames.textContent = data.frames_processed;
        if (error) error.textContent = data.error || '—';
      } catch (_) { /* best effort */ }
    };
    poll();
    setInterval(poll, 2000);
  }

  const cancelButton = qs('[data-cancel-url]');
  cancelButton?.addEventListener('click', async () => {
    cancelButton.disabled = true;
    await fetch(cancelButton.dataset.cancelUrl, { method: 'POST', headers: { 'X-CSRFToken': csrfToken() } });
  });

  const artifactsPanel = qs('[data-artifacts-url]');
  if (artifactsPanel) {
    const loadArtifacts = async () => {
      try {
        const response = await fetch(artifactsPanel.dataset.artifactsUrl, { headers: { Accept: 'application/json' } });
        if (!response.ok) return;
        const data = await response.json();
        const list = qs('[data-artifact-list]', artifactsPanel);
        if (!list || !data.artifacts?.length) return;
        list.replaceChildren(...data.artifacts.map((artifact) => {
          const row = document.createElement('div');
          row.className = 'artifact-row';
          const name = document.createElement('span');
          name.textContent = artifact.path;
          const link = document.createElement('a');
          link.href = artifact.download_url;
          link.textContent = 'Download';
          row.append(name, link);
          return row;
        }));
      } catch (_) { /* best effort */ }
    };
    loadArtifacts();
    setInterval(loadArtifacts, 3000);
  }
})();
