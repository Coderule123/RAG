(function () {
  const POLL_INTERVAL_MS = 1000;

  const state = {
    activeTab: "data",
    files: [],
    tags: [],
    selectedFile: null,
    previewCache: {},
    documents: [],
    selectedDocument: null,
    chunkFilter: "",
    jobPolling: null,
    logOffset: 0,
    logTimer: null,
    jobBusy: false,
  };

  const els = {
    tabData: document.getElementById("tabData"),
    tabIndex: document.getElementById("tabIndex"),
    tabBuild: document.getElementById("tabBuild"),
    dataPane: document.getElementById("dataPane"),
    indexPane: document.getElementById("indexPane"),
    buildPane: document.getElementById("buildPane"),
    fileList: document.getElementById("fileList"),
    fileDetail: document.getElementById("fileDetail"),
    dataStatus: document.getElementById("dataStatus"),
    refreshDataBtn: document.getElementById("refreshDataBtn"),
    uploadTag: document.getElementById("uploadTag"),
    uploadTagNew: document.getElementById("uploadTagNew"),
    uploadFile: document.getElementById("uploadFile"),
    uploadBtn: document.getElementById("uploadBtn"),
    indexSummary: document.getElementById("indexSummary"),
    docIndexList: document.getElementById("docIndexList"),
    chunkDetail: document.getElementById("chunkDetail"),
    chunkFilter: document.getElementById("chunkFilter"),
    indexStatus: document.getElementById("indexStatus"),
    refreshIndexBtn: document.getElementById("refreshIndexBtn"),
    buildConfigInfo: document.getElementById("buildConfigInfo"),
    incrementalCheck: document.getElementById("incrementalCheck"),
    buildBtn: document.getElementById("buildBtn"),
    buildStatus: document.getElementById("buildStatus"),
    jobStatusBox: document.getElementById("jobStatusBox"),
    jobStatusText: document.getElementById("jobStatusText"),
    jobResult: document.getElementById("jobResult"),
    docLogViewer: document.getElementById("docLogViewer"),
    clearLogBtn: document.getElementById("clearLogBtn"),
  };

  const INDEX_STATUS_LABEL = {
    indexed: "已索引",
    not_indexed: "未索引",
    changed: "已变更",
  };

  async function fetchJson(url, options) {
    const resp = await fetch(url, options);
    let data = {};
    try {
      data = await resp.json();
    } catch (_) {
      data = {};
    }
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  function setDataStatus(text) {
    els.dataStatus.textContent = text;
  }

  function setIndexStatus(text) {
    els.indexStatus.textContent = text;
  }

  function setBuildStatus(text) {
    els.buildStatus.textContent = text;
  }

  function switchTab(tab) {
    state.activeTab = tab;
    const map = {
      data: [els.tabData, els.dataPane],
      index: [els.tabIndex, els.indexPane],
      build: [els.tabBuild, els.buildPane],
    };
    [els.tabData, els.tabIndex, els.tabBuild].forEach((el) => el.classList.remove("active"));
    [els.dataPane, els.indexPane, els.buildPane].forEach((el) => el.classList.add("hidden"));
    const pair = map[tab];
    if (pair) {
      pair[0].classList.add("active");
      pair[1].classList.remove("hidden");
    }
    if (tab === "data") loadDataFiles();
    if (tab === "index") loadIndexData();
    if (tab === "build") {
      loadBuildConfig();
      startLogPolling();
    } else {
      stopLogPolling();
    }
  }

  function renderTagSelect() {
    const tags = state.tags.length ? state.tags : ["general"];
    els.uploadTag.innerHTML = tags
      .map((t) => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`)
      .join("");
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function groupFilesByTag(files) {
    const groups = {};
    files.forEach((f) => {
      const tag = f.tag || "general";
      if (!groups[tag]) groups[tag] = [];
      groups[tag].push(f);
    });
    return Object.keys(groups)
      .sort()
      .map((tag) => ({ tag, files: groups[tag] }));
  }

  function renderFileList() {
    const groups = groupFilesByTag(state.files);
    if (!groups.length) {
      els.fileList.innerHTML = '<p style="color:var(--muted);padding:8px;">暂无文档</p>';
      return;
    }
    let html = "";
    groups.forEach(({ tag, files }) => {
      html += `<div class="tag-group-title">${escapeHtml(tag)}</div>`;
      files.forEach((f) => {
        const active = state.selectedFile && state.selectedFile.relpath === f.relpath;
        const badgeClass = f.index_status || "not_indexed";
        const badgeLabel = INDEX_STATUS_LABEL[badgeClass] || badgeClass;
        html += `
          <div class="file-item ${active ? "active" : ""}" data-relpath="${escapeHtml(f.relpath)}">
            <div class="name">${escapeHtml(f.relpath)}</div>
            <div class="meta">
              ${formatSize(f.size)}
              <span class="badge ${badgeClass}">${badgeLabel}</span>
            </div>
          </div>`;
      });
    });
    els.fileList.innerHTML = html;
    els.fileList.querySelectorAll(".file-item").forEach((node) => {
      node.addEventListener("click", () => {
        const rel = node.getAttribute("data-relpath");
        const file = state.files.find((f) => f.relpath === rel);
        if (file) selectFile(file);
      });
    });
  }

  function selectFile(file) {
    state.selectedFile = file;
    renderFileList();
    const tags = state.tags.filter((t) => t !== "general");
    const moveOptions = tags
      .map((t) => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`)
      .join("");
    els.fileDetail.innerHTML = `
      <dl>
        <dt>路径</dt><dd>${escapeHtml(file.relpath)}</dd>
        <dt>Tag</dt><dd>${escapeHtml(file.tag)}</dd>
        <dt>Source</dt><dd>${escapeHtml(file.source)}</dd>
        <dt>大小</dt><dd>${formatSize(file.size)}</dd>
        <dt>修改时间</dt><dd>${escapeHtml(file.mtime)}</dd>
        <dt>索引状态</dt><dd>${INDEX_STATUS_LABEL[file.index_status] || file.index_status}</dd>
      </dl>
      <div class="form-row" style="margin-top:16px;">
        <label>移动到 tag</label>
        <select id="moveTargetTag">${moveOptions}</select>
        <button class="btn" id="moveFileBtn" type="button">移动</button>
        <button class="btn danger" id="deleteFileBtn" type="button">删除文件</button>
      </div>
      <p style="font-size:12px;color:var(--muted);margin-top:8px;">
        移动或删除 data 文件不会自动更新向量库，请在建库页重新索引或于向量页手动删除。
      </p>
      <div class="doc-preview-wrap">
        <h3>文档预览</h3>
        <pre class="doc-preview" id="docPreview">加载中…</pre>
        <div class="doc-preview-note hidden" id="docPreviewNote"></div>
      </div>`;

    document.getElementById("moveFileBtn").addEventListener("click", () => moveFile(file));
    document.getElementById("deleteFileBtn").addEventListener("click", () => deleteFile(file));
    loadFilePreview(file);
  }

  async function loadFilePreview(file) {
    const previewEl = document.getElementById("docPreview");
    const noteEl = document.getElementById("docPreviewNote");
    if (!previewEl) return;

    if (state.previewCache[file.relpath]) {
      renderFilePreview(state.previewCache[file.relpath], previewEl, noteEl);
      return;
    }

    previewEl.textContent = "加载中…";
    noteEl.classList.add("hidden");
    try {
      const data = await fetchJson(
        "/api/data/preview/" + file.relpath.split("/").map(encodeURIComponent).join("/")
      );
      state.previewCache[file.relpath] = data;
      renderFilePreview(data, previewEl, noteEl);
    } catch (e) {
      previewEl.textContent = "预览失败: " + e.message;
    }
  }

  function renderFilePreview(data, previewEl, noteEl) {
    previewEl.textContent = data.content || "(空内容)";
    if (data.truncated) {
      noteEl.textContent = "内容过长，仅显示部分内容";
      noteEl.classList.remove("hidden");
    } else if (data.format === "extracted") {
      noteEl.textContent = "PDF/DOCX 为抽取文本预览，可能与原文排版不同";
      noteEl.classList.remove("hidden");
    } else {
      noteEl.classList.add("hidden");
    }
  }

  async function loadDataFiles() {
    try {
      setDataStatus("加载中…");
      const data = await fetchJson("/api/data/files");
      state.files = data.files || [];
      state.tags = data.tags || ["general"];
      renderTagSelect();
      renderFileList();
      if (state.selectedFile) {
        const updated = state.files.find((f) => f.relpath === state.selectedFile.relpath);
        if (updated) selectFile(updated);
        else {
          state.selectedFile = null;
          els.fileDetail.innerHTML = '<p style="color:var(--muted)">选择左侧文件查看详情</p>';
        }
      }
      setDataStatus(`共 ${state.files.length} 个文档`);
    } catch (e) {
      setDataStatus("加载失败: " + e.message);
    }
  }

  async function uploadFile() {
    const fileInput = els.uploadFile;
    if (!fileInput.files || !fileInput.files[0]) {
      setDataStatus("请先选择文件");
      return;
    }
    const tag = (els.uploadTagNew.value || els.uploadTag.value || "general").trim();
    const form = new FormData();
    form.append("file", fileInput.files[0]);
    form.append("tag", tag);
    try {
      setDataStatus("上传中…");
      els.uploadBtn.disabled = true;
      const data = await fetchJson("/api/data/upload", { method: "POST", body: form });
      setDataStatus(`已上传: ${data.relpath}`);
      fileInput.value = "";
      els.uploadTagNew.value = "";
      delete state.previewCache[data.relpath];
      await loadDataFiles();
    } catch (e) {
      setDataStatus("上传失败: " + e.message);
    } finally {
      els.uploadBtn.disabled = false;
    }
  }

  async function moveFile(file) {
    const targetTag = document.getElementById("moveTargetTag").value;
    const basename = file.relpath.split("/").pop();
    const to = `${targetTag}/${basename}`;
    if (!confirm(`将 ${file.relpath} 移动到 ${to}？`)) return;
    try {
      setDataStatus("移动中…");
      await fetchJson("/api/data/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from: file.relpath, to }),
      });
      setDataStatus(`已移动到 ${to}`);
      delete state.previewCache[file.relpath];
      state.selectedFile = null;
      await loadDataFiles();
    } catch (e) {
      setDataStatus("移动失败: " + e.message);
    }
  }

  async function deleteFile(file) {
    if (!confirm(`确定删除 ${file.relpath}？此操作不可恢复。`)) return;
    try {
      setDataStatus("删除中…");
      await fetchJson(`/api/data/files/${encodeURIComponent(file.relpath)}`, {
        method: "DELETE",
      });
      setDataStatus(`已删除 ${file.relpath}`);
      delete state.previewCache[file.relpath];
      state.selectedFile = null;
      els.fileDetail.innerHTML = '<p style="color:var(--muted)">选择左侧文件查看详情</p>';
      await loadDataFiles();
    } catch (e) {
      setDataStatus("删除失败: " + e.message);
    }
  }

  function renderIndexSummary(summary) {
    els.indexSummary.innerHTML = `
      <div class="card"><div class="label">FAISS</div><div class="value">${summary.faiss_ready ? "就绪" : "无"}</div></div>
      <div class="card"><div class="label">已索引文档</div><div class="value">${summary.doc_hash_count}</div></div>
      <div class="card"><div class="label">Chunk 数</div><div class="value">${summary.chunk_count}</div></div>
      <div class="card"><div class="label">Metadata 文件</div><div class="value">${(summary.metadata_files || []).length}</div></div>`;
  }

  function filteredDocuments() {
    const q = (state.chunkFilter || "").trim().toLowerCase();
    if (!q) return state.documents;
    return state.documents.filter((doc) => {
      const hay = [doc.doc_name, doc.metadata_path, doc.tag, String(doc.chunk_count)]
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
  }

  function renderDocIndexList() {
    const docs = filteredDocuments();
    if (!docs.length) {
      els.docIndexList.innerHTML = '<p style="color:var(--muted);padding:8px;">暂无向量文档</p>';
      return;
    }
    els.docIndexList.innerHTML = docs
      .map((doc) => {
        const active =
          state.selectedDocument &&
          state.selectedDocument.metadata_path === doc.metadata_path;
        return `
          <div class="doc-index-item ${active ? "active" : ""}" data-meta-path="${escapeHtml(doc.metadata_path)}">
            <div class="name">${escapeHtml(doc.doc_name)}</div>
            <div class="meta">${escapeHtml(doc.metadata_path)} · ${doc.chunk_count} chunks · tag=${escapeHtml(doc.tag || "—")}</div>
          </div>`;
      })
      .join("");

    els.docIndexList.querySelectorAll(".doc-index-item").forEach((node) => {
      node.addEventListener("click", () => {
        const metaPath = node.getAttribute("data-meta-path");
        const doc = state.documents.find((d) => d.metadata_path === metaPath);
        if (doc) selectDocument(doc);
      });
    });
  }

  function selectDocument(doc) {
    state.selectedDocument = doc;
    renderDocIndexList();
    renderChunkDetail(doc);
  }

  function renderChunkDetail(doc) {
    if (!doc || !doc.chunks || !doc.chunks.length) {
      els.chunkDetail.innerHTML = '<p class="index-detail-empty">该文档暂无向量 chunk</p>';
      return;
    }

    const metaPath = doc.metadata_path;
    const header = `
      <div style="margin-bottom:8px;">
        <strong>${escapeHtml(doc.doc_name)}</strong>
        <span style="color:var(--muted);font-size:12px;margin-left:8px;">${escapeHtml(metaPath)}</span>
        <button class="btn danger" style="margin-left:12px;" id="deleteAllDocBtn" type="button">删除该文档全部向量</button>
      </div>`;

    const cards = doc.chunks
      .map((c, idx) => {
        return `
        <div class="chunk-card" data-chunk-idx="${idx}">
          <div class="chunk-card-header">
            <span class="field">tag<strong>${escapeHtml(c.tag || "")}</strong></span>
            <span class="field">doc_id<strong>${c.doc_id}</strong></span>
            <span class="field">chunk_id<strong>${c.chunk_id}</strong></span>
            <div class="chunk-card-actions">
              <button class="btn" data-action="doc" data-chunk-idx="${idx}" type="button">删 doc_id</button>
              <button class="btn" data-action="chunk" data-chunk-idx="${idx}" type="button">删 chunk</button>
            </div>
          </div>
          <div class="chunk-text">${escapeHtml(c.text || "")}</div>
        </div>`;
      })
      .join("");

    els.chunkDetail.innerHTML = header + cards;

    document.getElementById("deleteAllDocBtn").addEventListener("click", () => {
      handleChunkDelete({ metadata_path: metaPath, action: "all" });
    });

    els.chunkDetail.querySelectorAll("button[data-action]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.getAttribute("data-chunk-idx"), 10);
        const chunk = doc.chunks[idx];
        handleChunkDelete({
          metadata_path: metaPath,
          action: btn.getAttribute("data-action"),
          doc_id: chunk.doc_id,
          chunk_id: chunk.chunk_id,
        });
      });
    });
  }

  async function handleChunkDelete(params) {
    if (state.jobBusy) {
      setIndexStatus("有任务正在执行，请稍候");
      return;
    }
    const metadata = params.metadata_path;
    let body = { metadata };
    let confirmMsg = "";

    if (params.action === "all") {
      body.all = true;
      confirmMsg = `删除 ${metadata} 的全部向量？`;
    } else if (params.action === "doc") {
      body.doc_ids = [parseInt(params.doc_id, 10)];
      confirmMsg = `删除 ${metadata} doc_id=${body.doc_ids[0]} 的全部 chunk？`;
    } else {
      body.doc_ids = [parseInt(params.doc_id, 10)];
      body.chunk_id = parseInt(params.chunk_id, 10);
      confirmMsg = `删除 ${metadata} doc_id=${body.doc_ids[0]} chunk_id=${body.chunk_id}？`;
    }

    if (!confirm(confirmMsg)) return;

    try {
      setIndexStatus("提交删除任务…");
      await fetchJson("/api/index/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setIndexStatus("删除任务已提交，等待完成…");
      startJobPolling(() => {
        loadIndexData();
        setIndexStatus("删除完成");
      });
    } catch (e) {
      setIndexStatus("删除失败: " + e.message);
    }
  }

  async function loadIndexData() {
    try {
      setIndexStatus("加载中…");
      const [summary, meta] = await Promise.all([
        fetchJson("/api/index/summary"),
        fetchJson("/api/index/metadata"),
      ]);
      renderIndexSummary(summary);
      state.documents = meta.documents || [];
      const prevPath = state.selectedDocument && state.selectedDocument.metadata_path;
      renderDocIndexList();
      if (prevPath) {
        const updated = state.documents.find((d) => d.metadata_path === prevPath);
        if (updated) selectDocument(updated);
        else {
          state.selectedDocument = null;
          els.chunkDetail.innerHTML = '<p class="index-detail-empty">选择左侧文档查看向量 chunk</p>';
        }
      } else if (state.documents.length === 1) {
        selectDocument(state.documents[0]);
      }
      setIndexStatus(`共 ${meta.total_documents || state.documents.length} 篇文档，${meta.total_chunks || 0} 条 chunk`);
    } catch (e) {
      setIndexStatus("加载失败: " + e.message);
    }
  }

  async function loadBuildConfig() {
    try {
      const summary = await fetchJson("/api/index/summary");
      const vi = summary.vector_index || {};
      const models = summary.models || {};
      els.buildConfigInfo.innerHTML = `
        <strong>当前配置（只读）</strong><br>
        chunk_size: ${vi.chunk_size ?? "—"} &nbsp; overlap: ${vi.overlap ?? "—"} &nbsp;
        batch_size: ${vi.batch_size ?? "—"} &nbsp; 默认增量: ${vi.incremental ? "是" : "否"}<br>
        embedding: ${escapeHtml(models.embedding_model || "—")}<br>
        index_dir: ${escapeHtml(summary.index_dir || "—")}`;
      els.incrementalCheck.checked = vi.incremental !== false;
    } catch (e) {
      els.buildConfigInfo.textContent = "加载配置失败: " + e.message;
    }
  }

  async function startBuild() {
    if (state.jobBusy) {
      setBuildStatus("有任务正在执行");
      return;
    }
    const incremental = els.incrementalCheck.checked;
    const mode = incremental ? "增量" : "全量";
    if (!incremental && !confirm("全量模式将重建向量库，确定继续？")) return;

    try {
      setBuildStatus(`${mode}建库任务提交中…`);
      els.buildBtn.disabled = true;
      state.logOffset = 0;
      els.docLogViewer.textContent = "";
      await fetchJson("/api/index/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incremental }),
      });
      setBuildStatus(`${mode}建库任务已提交`);
      els.jobStatusBox.classList.remove("hidden");
      startJobPolling(() => {
        loadDataFiles();
        setBuildStatus("建库完成");
        els.buildBtn.disabled = false;
      });
    } catch (e) {
      setBuildStatus("建库失败: " + e.message);
      els.buildBtn.disabled = false;
    }
  }

  function startJobPolling(onSuccess) {
    stopJobPolling();
    state.jobBusy = true;
    els.jobStatusBox.classList.remove("hidden");

    const poll = async () => {
      try {
        const data = await fetchJson("/api/jobs/current");
        const job = data.job || {};
        els.jobStatusText.textContent = `${job.status || "—"} — ${job.message || ""}`;
        if (job.status === "running") {
          els.jobResult.classList.add("hidden");
        } else if (job.status === "success") {
          els.jobResult.classList.remove("hidden");
          els.jobResult.textContent = JSON.stringify(job.result, null, 2);
          stopJobPolling();
          state.jobBusy = false;
          els.buildBtn.disabled = false;
          if (onSuccess) onSuccess();
        } else if (job.status === "error") {
          els.jobResult.classList.remove("hidden");
          els.jobResult.textContent = job.error || "未知错误";
          stopJobPolling();
          state.jobBusy = false;
          els.buildBtn.disabled = false;
        }
      } catch (_) {
        /* ignore transient poll errors */
      }
    };

    poll();
    state.jobPolling = setInterval(poll, POLL_INTERVAL_MS);
  }

  function stopJobPolling() {
    if (state.jobPolling) {
      clearInterval(state.jobPolling);
      state.jobPolling = null;
    }
  }

  async function pollDocLog() {
    try {
      const data = await fetchJson(`/api/logs/doc?from=${state.logOffset}`);
      if (data.content) {
        els.docLogViewer.textContent += data.content;
        els.docLogViewer.scrollTop = els.docLogViewer.scrollHeight;
      }
      state.logOffset = data.next_offset || state.logOffset;
    } catch (_) {
      /* ignore */
    }
  }

  function startLogPolling() {
    stopLogPolling();
    pollDocLog();
    state.logTimer = setInterval(pollDocLog, POLL_INTERVAL_MS);
  }

  function stopLogPolling() {
    if (state.logTimer) {
      clearInterval(state.logTimer);
      state.logTimer = null;
    }
  }

  els.tabData.addEventListener("click", () => switchTab("data"));
  els.tabIndex.addEventListener("click", () => switchTab("index"));
  els.tabBuild.addEventListener("click", () => switchTab("build"));
  els.refreshDataBtn.addEventListener("click", loadDataFiles);
  els.uploadBtn.addEventListener("click", uploadFile);
  els.refreshIndexBtn.addEventListener("click", loadIndexData);
  els.chunkFilter.addEventListener("input", () => {
    state.chunkFilter = els.chunkFilter.value;
    renderDocIndexList();
  });
  els.buildBtn.addEventListener("click", startBuild);
  els.clearLogBtn.addEventListener("click", () => {
    els.docLogViewer.textContent = "";
    state.logOffset = 0;
  });

  loadDataFiles();
})();
