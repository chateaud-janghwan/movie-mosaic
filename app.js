const $ = (selector) => document.querySelector(selector);

const MEDIAPIPE_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MEDIAPIPE_MODULE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const MEDIAPIPE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";
const SAMPLE_HOLD_SECONDS = 0.7;
const SAMPLE_INTERPOLATE_SECONDS = 1.6;
const TRACK_HOLD_SECONDS = 0.58;
const TRACK_IOU_THRESHOLD = 0.16;
const TRACK_SIGNATURE_THRESHOLD = 0.68;
const TRACK_DETECTION_BLEND = 0.46;
const TRACK_VELOCITY_BLEND = 0.32;
const TRACK_MISSED_DAMPING = 0.78;

const els = {
  videoInput: $("#videoInput"),
  fileName: $("#fileName"),
  video: $("#video"),
  overlay: $("#overlay"),
  renderCanvas: $("#renderCanvas"),
  stage: $("#stage"),
  emptyState: $("#emptyState"),
  timeline: $("#timeline"),
  playBtn: $("#playBtn"),
  currentTime: $("#currentTime"),
  duration: $("#duration"),
  scanBtn: $("#scanBtn"),
  clearFacesBtn: $("#clearFacesBtn"),
  renderBtn: $("#renderBtn"),
  protectedFaces: $("#protectedFaces"),
  faceCount: $("#faceCount"),
  detectModeBtn: $("#detectModeBtn"),
  manualModeBtn: $("#manualModeBtn"),
  pixelSize: $("#pixelSize"),
  pixelSizeValue: $("#pixelSizeValue"),
  matchThreshold: $("#matchThreshold"),
  matchThresholdValue: $("#matchThresholdValue"),
  detectorState: $("#detectorState"),
  statusText: $("#statusText"),
  renderProgress: $("#renderProgress"),
  selectionHint: $("#selectionHint"),
  downloadLink: $("#downloadLink"),
  faceBoard: $("#faceBoard"),
  scanSummary: $("#scanSummary"),
  mosaicFaces: $("#mosaicFaces"),
  protectedFacesBoard: $("#protectedFacesBoard"),
  originalPreview: $("#originalPreview"),
  mosaicPreview: $("#mosaicPreview"),
  comparePanel: $("#comparePanel"),
};

const state = {
  detector: null,
  detectorKind: null,
  mode: "detect",
  faces: [],
  protected: [],
  candidates: [],
  annotations: [],
  manualStart: null,
  manualDraft: null,
  isRendering: false,
  isScanningLibrary: false,
  sourceUrl: null,
};

const overlayCtx = els.overlay.getContext("2d");
const renderCtx = els.renderCanvas.getContext("2d", { willReadFrequently: true });
const originalPreviewCtx = els.originalPreview.getContext("2d");
const mosaicPreviewCtx = els.mosaicPreview.getContext("2d");

init();

async function init() {
  bindEvents();
  updateControls(false);
  drawOverlay();
  await setupDetector();
  updateControls(Boolean(els.video.duration));
}

async function setupDetector() {
  setDetectorState("", "감지 모델 확인 중");

  if ("FaceDetector" in window) {
    state.detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 30 });
    state.detectorKind = "native";
    setDetectorState("ready", "FaceDetector 사용 중");
    return;
  }

  try {
    setDetectorState("", "MediaPipe 로딩 중");
    const vision = await import(MEDIAPIPE_MODULE_URL);
    const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    state.detector = await createMediaPipeDetector(vision.FaceDetector, fileset, "GPU").catch(() =>
      createMediaPipeDetector(vision.FaceDetector, fileset, "CPU"),
    );
    state.detectorKind = "mediapipe";
    setDetectorState("ready", "MediaPipe 감지 사용 중");
    setMode("detect");
  } catch (error) {
    console.warn(error);
    state.detector = null;
    state.detectorKind = null;
    setDetectorState("warn", "직접 지정 모드");
    state.mode = "manual";
    els.detectModeBtn.disabled = true;
    setMode("manual");
    setStatus("자동 감지 모델을 불러오지 못했습니다. 네트워크를 확인해 주세요.");
  }
}

function createMediaPipeDetector(FaceDetectorTask, fileset, delegate) {
  return FaceDetectorTask.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MEDIAPIPE_MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.45,
    minSuppressionThreshold: 0.3,
  });
}

function bindEvents() {
  els.videoInput.addEventListener("change", loadVideo);
  els.video.addEventListener("loadedmetadata", onMetadata);
  els.video.addEventListener("timeupdate", syncTimeline);
  els.video.addEventListener("seeked", () => {
    if (state.isScanningLibrary || state.isRendering) return;
    scanCurrentFrame();
    drawOverlay();
  });
  els.video.addEventListener("play", () => (els.playBtn.textContent = "Ⅱ"));
  els.video.addEventListener("pause", () => (els.playBtn.textContent = "▶"));

  els.timeline.addEventListener("input", () => {
    els.video.currentTime = Number(els.timeline.value);
    syncTimeline();
  });

  els.playBtn.addEventListener("click", () => {
    if (els.video.paused) els.video.play();
    else els.video.pause();
  });

  els.scanBtn.addEventListener("click", scanCurrentFrame);
  els.clearFacesBtn.addEventListener("click", clearProtectedFaces);
  els.renderBtn.addEventListener("click", renderMosaicVideo);
  els.detectModeBtn.addEventListener("click", () => setMode("detect"));
  els.manualModeBtn.addEventListener("click", () => setMode("manual"));
  els.pixelSize.addEventListener("input", updateSliderLabels);
  els.matchThreshold.addEventListener("input", updateSliderLabels);

  els.overlay.addEventListener("pointerdown", onPointerDown);
  els.overlay.addEventListener("pointermove", onPointerMove);
  els.overlay.addEventListener("pointerup", onPointerUp);
  els.overlay.addEventListener("click", onOverlayClick);
  document.querySelectorAll(".drop-zone").forEach((zone) => {
    zone.addEventListener("dragover", onDragOver);
    zone.addEventListener("dragleave", onDragLeave);
    zone.addEventListener("drop", onDropFace);
  });
  window.addEventListener("resize", drawOverlay);
}

function loadVideo(event) {
  const [file] = event.target.files;
  if (!file) return;

  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = URL.createObjectURL(file);
  els.video.src = state.sourceUrl;
  els.fileName.textContent = file.name;
  els.emptyState.style.display = "none";
  els.faceBoard.classList.add("hidden");
  els.comparePanel.classList.add("hidden");
  els.downloadLink.classList.add("hidden");
  clearProtectedFaces();
  state.candidates = [];
  renderCandidateBoard();
  setStatus("메타데이터를 읽는 중입니다.");
}

async function onMetadata() {
  els.timeline.max = String(els.video.duration || 0);
  els.duration.textContent = formatTime(els.video.duration);
  els.video.currentTime = 0;
  resizeCanvases();
  resizePreviewCanvases();
  updateControls(true);
  syncTimeline();
  await scanVideoLibrary();
  await scanCurrentFrame();
  setStatus("얼굴 카드를 드래그해서 모자이크/보호를 나누세요.");
}

function resizeCanvases() {
  const width = els.video.videoWidth || 1280;
  const height = els.video.videoHeight || 720;
  if (els.overlay.width !== width) els.overlay.width = width;
  if (els.overlay.height !== height) els.overlay.height = height;
  if (els.renderCanvas.width !== width) els.renderCanvas.width = width;
  if (els.renderCanvas.height !== height) els.renderCanvas.height = height;
}

function resizePreviewCanvases() {
  const width = els.video.videoWidth || 1280;
  const height = els.video.videoHeight || 720;
  [els.originalPreview, els.mosaicPreview].forEach((canvas) => {
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  });
}

function updateControls(hasVideo) {
  els.timeline.disabled = !hasVideo;
  els.playBtn.disabled = !hasVideo;
  els.scanBtn.disabled = !hasVideo || state.isScanningLibrary || (state.mode === "detect" && !state.detector);
  els.renderBtn.disabled = !hasVideo || state.isScanningLibrary || state.isRendering;
  els.clearFacesBtn.disabled = state.protected.length === 0 && state.candidates.length === 0;
}

function setMode(mode) {
  state.mode = mode;
  els.stage.classList.toggle("manual", mode === "manual");
  els.detectModeBtn.classList.toggle("active", mode === "detect");
  els.manualModeBtn.classList.toggle("active", mode === "manual");
  els.selectionHint.textContent =
    mode === "detect"
      ? "얼굴을 찾은 뒤 박스를 클릭하면 보호 목록에 추가됩니다."
      : "얼굴 영역을 드래그하면 보호 목록에 추가됩니다.";
  updateControls(Boolean(els.video.duration));
  drawOverlay();
}

async function scanCurrentFrame() {
  if (!els.video.duration || state.isRendering) return;
  resizeCanvases();

  if (!state.detector || state.mode === "manual") {
    state.faces = nearestAnnotations(els.video.currentTime);
    drawOverlay();
    return;
  }

  setStatus("현재 프레임에서 얼굴을 찾는 중입니다.");
  try {
    const faces = await detectFaces(els.video);
    state.faces = faces.map((face, index) => ({
      id: `face-${Date.now()}-${index}`,
      box: face.box,
      kind: "detected",
    }));
    setStatus(`${state.faces.length}개의 얼굴을 찾았습니다.`);
  } catch (error) {
    console.warn(error);
    setDetectorState("warn", "직접 지정 모드");
    setMode("manual");
    setStatus("이 브라우저에서는 자동 감지가 제한됩니다.");
  }
  drawOverlay();
}

async function detectFaces(source) {
  if (!state.detector) return [];

  if (state.detectorKind === "native") {
    const faces = await state.detector.detect(source);
    return faces.map((face) => ({ box: normalizeBox(face.boundingBox) }));
  }

  const result = state.detector.detectForVideo(source, Math.round(performance.now()));
  return (result.detections || []).map((detection) => ({
    box: normalizeMediaPipeBox(detection.boundingBox),
  }));
}

async function scanVideoLibrary() {
  if (!els.video.duration || !state.detector) {
    els.faceBoard.classList.remove("hidden");
    renderCandidateBoard();
    return;
  }

  state.isScanningLibrary = true;
  updateControls(true);
  state.candidates = [];
  state.protected = [];
  renderProtectedFaces();
  renderCandidateBoard();
  setStatus("업로드한 영상에서 얼굴을 훑는 중입니다.");

  const duration = els.video.duration;
  const samples = Math.min(40, Math.max(8, Math.ceil(duration * 1.25)));
  const originalTime = els.video.currentTime || 0;

  try {
    for (let index = 0; index < samples; index += 1) {
      const time = samples === 1 ? 0 : (duration * index) / (samples - 1);
      await seekVideo(Math.min(duration - 0.05, Math.max(0, time)));
      captureFrame();
      const faces = await detectFaces(els.video);
      faces.forEach((face) => addCandidateFace(face.box, els.video.currentTime));
      els.renderProgress.value = ((index + 1) / samples) * 100;
      setStatus(`${index + 1}/${samples} 지점 얼굴 스캔 중`);
      await wait(25);
    }
  } catch (error) {
    console.warn(error);
    setStatus("자동 스캔 중 일부 프레임을 건너뛰었습니다.");
  } finally {
    await seekVideo(Math.min(originalTime, duration));
    state.isScanningLibrary = false;
    els.renderProgress.value = 0;
    els.faceBoard.classList.remove("hidden");
    syncProtectedFromCandidates();
    renderCandidateBoard();
    updateControls(true);
    setStatus(`${state.candidates.length}개의 얼굴 후보를 찾았습니다.`);
  }
}

function addCandidateFace(box, time) {
  const safeBox = clampBox(box);
  if (safeBox.width < 16 || safeBox.height < 16) return;

  const signature = makeSignature(renderCtx, safeBox);
  const existing = state.candidates.find((candidate) => similarity(candidate.signature, signature) > 0.86);
  if (existing) {
    existing.seen += 1;
    existing.samples.push({ time, box: { ...safeBox } });
    if (safeBox.width * safeBox.height > existing.box.width * existing.box.height) {
      existing.box = { ...safeBox };
      existing.time = time;
      existing.signature = signature;
      existing.crop = cropFace(safeBox);
    }
    return;
  }

  state.candidates.push({
    id: crypto.randomUUID(),
    name: `얼굴 ${state.candidates.length + 1}`,
    category: "mosaic",
    time,
    box: { ...safeBox },
    signature,
    crop: cropFace(safeBox),
    seen: 1,
    samples: [{ time, box: { ...safeBox } }],
  });
}

function normalizeBox(box) {
  return {
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
  };
}

function normalizeMediaPipeBox(box) {
  return {
    x: box.originX ?? box.origin_x ?? box.x ?? 0,
    y: box.originY ?? box.origin_y ?? box.y ?? 0,
    width: box.width ?? 0,
    height: box.height ?? 0,
  };
}

function onOverlayClick(event) {
  if (state.mode !== "detect" || state.isRendering) return;
  const point = eventToVideoPoint(event);
  const face = state.faces.find(({ box }) => pointInBox(point, box));
  if (!face) return;
  addProtectedFace(face.box);
}

function onPointerDown(event) {
  if (state.mode !== "manual" || state.isRendering || !els.video.duration) return;
  state.manualStart = eventToVideoPoint(event);
  state.manualDraft = { x: state.manualStart.x, y: state.manualStart.y, width: 0, height: 0 };
  els.overlay.setPointerCapture(event.pointerId);
}

function onPointerMove(event) {
  if (!state.manualStart || state.mode !== "manual") return;
  const point = eventToVideoPoint(event);
  state.manualDraft = boxFromPoints(state.manualStart, point);
  drawOverlay();
}

function onPointerUp(event) {
  if (!state.manualStart || state.mode !== "manual") return;
  const point = eventToVideoPoint(event);
  const box = clampBox(boxFromPoints(state.manualStart, point));
  state.manualStart = null;
  state.manualDraft = null;

  if (box.width < 18 || box.height < 18) {
    drawOverlay();
    return;
  }

  state.faces = [{ id: `manual-${Date.now()}`, box, kind: "manual" }];
  state.annotations.push({ time: els.video.currentTime, box, category: "mosaic" });
  addMosaicCandidate(box);
  drawOverlay();
}

function addMosaicCandidate(box) {
  captureFrame();
  const safeBox = clampBox(box);
  const signature = makeSignature(renderCtx, safeBox);
  state.candidates.push({
    id: crypto.randomUUID(),
    name: `얼굴 ${state.candidates.length + 1}`,
    category: "mosaic",
    time: els.video.currentTime,
    box: { ...safeBox },
    signature,
    crop: cropFace(safeBox),
    seen: 1,
    samples: [{ time: els.video.currentTime, box: { ...safeBox } }],
  });
  syncProtectedFromCandidates();
  renderCandidateBoard();
  updateControls(true);
  setStatus("직접 지정한 영역을 모자이크 대상으로 추가했습니다.");
}

function addProtectedFace(box) {
  captureFrame();
  const signature = makeSignature(renderCtx, box);
  const crop = cropFace(box);
  const existing = state.candidates.find((candidate) => similarity(candidate.signature, signature) > 0.86);
  if (existing) {
    existing.category = "protect";
    syncProtectedFromCandidates();
    renderCandidateBoard();
    renderProtectedFaces();
    updateControls(true);
    setStatus(`${existing.name}을 보호 목록으로 옮겼습니다.`);
    return;
  }

  const protectedFace = {
    id: crypto.randomUUID(),
    name: `얼굴 ${state.protected.length + 1}`,
    category: "protect",
    time: els.video.currentTime,
    box: { ...box },
    signature,
    crop,
    seen: 1,
    samples: [{ time: els.video.currentTime, box: { ...box } }],
  };
  state.candidates.push(protectedFace);
  syncProtectedFromCandidates();
  renderCandidateBoard();
  renderProtectedFaces();
  updateControls(true);
  setStatus(`${protectedFace.name}을 보호 목록에 추가했습니다.`);
}

function clearProtectedFaces() {
  state.protected = [];
  state.candidates = [];
  state.annotations = [];
  state.faces = [];
  renderProtectedFaces();
  renderCandidateBoard();
  updateControls(Boolean(els.video.duration));
  drawOverlay();
}

function syncProtectedFromCandidates() {
  state.protected = state.candidates.filter((candidate) => candidate.category === "protect");
  renderProtectedFaces();
}

function renderProtectedFaces() {
  els.faceCount.textContent = `${state.protected.length}명`;
  els.protectedFaces.classList.toggle("empty", state.protected.length === 0);

  if (state.protected.length === 0) {
    els.protectedFaces.innerHTML = "<span>타임바에서 얼굴을 선택하세요</span>";
    return;
  }

  els.protectedFaces.innerHTML = "";
  state.protected.forEach((face) => {
    const item = document.createElement("div");
    item.className = "face-item";
    item.innerHTML = `
      <img src="${face.crop}" alt="${face.name}" />
      <div>
        <div class="face-name">${face.name}</div>
        <div class="face-meta">${formatTime(face.time)} 선택</div>
      </div>
      <button class="remove-face" type="button" aria-label="${face.name} 삭제">×</button>
    `;
    item.querySelector("button").addEventListener("click", () => {
      const candidate = state.candidates.find((entry) => entry.id === face.id);
      if (candidate) candidate.category = "mosaic";
      syncProtectedFromCandidates();
      renderCandidateBoard();
      renderProtectedFaces();
      updateControls(Boolean(els.video.duration));
    });
    els.protectedFaces.appendChild(item);
  });
}

function renderCandidateBoard() {
  const mosaic = state.candidates.filter((face) => face.category !== "protect");
  const protect = state.candidates.filter((face) => face.category === "protect");
  els.scanSummary.textContent = `${state.candidates.length}명 감지`;
  els.mosaicFaces.innerHTML = "";
  els.protectedFacesBoard.innerHTML = "";
  renderCandidateCards(els.mosaicFaces, mosaic);
  renderCandidateCards(els.protectedFacesBoard, protect);

  if (state.candidates.length === 0) {
    els.mosaicFaces.innerHTML = `<div class="empty-drop">감지된 얼굴이 여기에 표시됩니다</div>`;
    els.protectedFacesBoard.innerHTML = `<div class="empty-drop">보호할 얼굴을 이쪽으로 드래그</div>`;
  }
}

function renderCandidateCards(container, faces) {
  faces.forEach((face) => {
    const card = document.createElement("button");
    card.className = "candidate-card";
    card.type = "button";
    card.draggable = true;
    card.dataset.id = face.id;
    card.innerHTML = `
      <img src="${face.crop}" alt="${face.name}" />
      <span>
        <strong>${face.name}</strong>
        <span>${formatTime(face.time)} · ${face.seen}회</span>
      </span>
    `;
    card.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", face.id);
      event.dataTransfer.effectAllowed = "move";
    });
    card.addEventListener("click", async () => {
      await seekVideo(face.time);
      state.faces = [{ id: face.id, box: face.box, kind: "detected" }];
      drawOverlay();
    });
    container.appendChild(card);
  });
}

function onDragOver(event) {
  event.preventDefault();
  event.currentTarget.classList.add("drag-over");
}

function onDragLeave(event) {
  event.currentTarget.classList.remove("drag-over");
}

function onDropFace(event) {
  event.preventDefault();
  const zone = event.currentTarget;
  zone.classList.remove("drag-over");
  const id = event.dataTransfer.getData("text/plain");
  const candidate = state.candidates.find((face) => face.id === id);
  if (!candidate) return;
  candidate.category = zone.dataset.category;
  syncProtectedFromCandidates();
  renderCandidateBoard();
  updateControls(Boolean(els.video.duration));
  setStatus(
    candidate.category === "protect"
      ? `${candidate.name}은 모자이크에서 제외합니다.`
      : `${candidate.name}은 모자이크 대상입니다.`,
  );
}

async function renderMosaicVideo() {
  if (!els.video.duration || state.isRendering) return;
  state.isRendering = true;
  updateControls(true);
  els.downloadLink.classList.add("hidden");
  els.downloadLink.removeAttribute("href");
  els.renderProgress.value = 0;
  els.video.pause();
  await seekVideo(0);

  const fps = 12;
  const stream = els.renderCanvas.captureStream(fps);
  const [canvasTrack] = stream.getVideoTracks();
  const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";
  const recorder = new MediaRecorder(stream, { mimeType });
  const chunks = [];
  const stopped = new Promise((resolve) => {
    recorder.addEventListener(
      "stop",
      () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        if (blob.size === 0) {
          setStatus("다운로드 파일을 만들지 못했습니다. 다시 생성해 주세요.");
          resolve();
          return;
        }
        const url = URL.createObjectURL(blob);
        els.downloadLink.href = url;
        els.downloadLink.download = makeDownloadName();
        els.downloadLink.classList.remove("hidden");
        setStatus("모자이크 영상이 준비되었습니다. 다운로드 버튼을 누르세요.");
        resolve();
      },
      { once: true },
    );
  });

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  });

  els.renderCanvas.style.display = "block";
  els.video.style.opacity = "0";
  els.comparePanel.classList.remove("hidden");
  recorder.start();
  setStatus("모자이크를 생성하는 중입니다.");

  try {
    await processFramesBySeek(fps, canvasTrack);
    recorder.requestData?.();
    await wait(120);
    if (recorder.state !== "inactive") {
      recorder.stop();
      await stopped;
    }
  } catch (error) {
    console.error(error);
    recorder.state !== "inactive" && recorder.stop();
    setStatus("생성 중 문제가 발생했습니다. 다른 영상이나 브라우저로 다시 시도해 주세요.");
  } finally {
    els.video.pause();
    els.video.style.opacity = "1";
    els.renderCanvas.style.display = "none";
    state.isRendering = false;
    updateControls(true);
    await seekVideo(0);
  }
}

function makeDownloadName() {
  const base = els.fileName.textContent.replace(/\.[^.]+$/, "") || "movie-mosaic";
  return `${base}-mosaic.webm`;
}

async function processFramesBySeek(fps, canvasTrack) {
  const duration = els.video.duration;
  const totalFrames = Math.max(1, Math.ceil(duration * fps));
  const tracker = createFaceTracker();

  for (let frame = 0; frame <= totalFrames; frame += 1) {
    const time = Math.min(duration - 0.02, frame / fps);
    await seekVideo(Math.max(0, time));
    captureFrame();
    updateOriginalPreview();

    let activeFaces = [];
    try {
      activeFaces = state.detector ? await detectFaces(els.video) : nearestAnnotations(time);
    } catch {
      activeFaces = nearestAnnotations(time);
    }

    const trackedFaces = tracker.update(time, activeFaces);
    applyMosaic(time, trackedFaces);
    updateMosaicPreview();
    canvasTrack?.requestFrame?.();

    els.renderProgress.value = Math.min(100, (frame / totalFrames) * 100);
    els.statusText.textContent = `${Math.round(els.renderProgress.value)}% 생성 중`;
    await wait(1000 / fps);
  }
}

function applyMosaic(time, faces) {
  const pixelSize = Number(els.pixelSize.value);
  const boxes = mergeMosaicBoxes(time, faces);
  boxes.forEach(({ box, force }) => {
    const signature = makeSignature(renderCtx, box);
    if (!force && isProtected(signature)) return;
    mosaicBox(box, pixelSize);
  });
}

function createFaceTracker() {
  const tracks = [];

  return {
    update(time, detections) {
      const normalized = detections.map(({ box }) => ({
        box: clampBox(box),
        signature: makeSignature(renderCtx, clampBox(box)),
      }));

      tracks.forEach((track) => {
        track.matched = false;
        track.predictedBox = predictTrackBox(track, time);
      });

      normalized.forEach((detection) => {
        const track = bestTrackForDetection(tracks, detection);
        if (track) {
          const dt = Math.max(1 / 30, time - track.lastUpdate);
          const previousBox = track.box;
          const predicted = track.predictedBox || track.box;
          track.box = smoothBox(predicted, detection.box, TRACK_DETECTION_BLEND);
          const measuredVelocity = {
            x: (track.box.x - previousBox.x) / dt,
            y: (track.box.y - previousBox.y) / dt,
            width: (track.box.width - previousBox.width) / dt,
            height: (track.box.height - previousBox.height) / dt,
          };
          track.velocity = blendVelocity(track.velocity, measuredVelocity, TRACK_VELOCITY_BLEND);
          track.signature = blendSignature(track.signature, detection.signature, 0.35);
          track.lastSeen = time;
          track.lastUpdate = time;
          track.matched = true;
        } else {
          tracks.push({
            id: crypto.randomUUID(),
            box: detection.box,
            velocity: { x: 0, y: 0, width: 0, height: 0 },
            signature: detection.signature,
            lastSeen: time,
            lastUpdate: time,
            matched: true,
          });
        }
      });

      for (let index = tracks.length - 1; index >= 0; index -= 1) {
        const track = tracks[index];
        const age = time - track.lastSeen;
        if (age > TRACK_HOLD_SECONDS) {
          tracks.splice(index, 1);
        } else if (!track.matched) {
          track.box = track.predictedBox || track.box;
          track.velocity = scaleVelocity(track.velocity, TRACK_MISSED_DAMPING);
          track.lastUpdate = time;
        }
      }

      return tracks.map((track) => ({ box: track.box, trackId: track.id }));
    },
  };
}

function bestTrackForDetection(tracks, detection) {
  let best = null;
  let bestScore = 0;
  tracks.forEach((track) => {
    if (track.matched) return;
    const predicted = track.predictedBox || track.box;
    const overlap = boxIou(predicted, detection.box);
    const visual = similarity(track.signature, detection.signature);
    const center = centerDistanceScore(predicted, detection.box);
    const score = overlap * 0.55 + visual * 0.25 + center * 0.2;
    if (
      score > bestScore &&
      (overlap >= TRACK_IOU_THRESHOLD || visual >= TRACK_SIGNATURE_THRESHOLD || center > 0.74)
    ) {
      best = track;
      bestScore = score;
    }
  });
  return best;
}

function smoothBox(a, b, amount) {
  return interpolateBox(a, b, amount);
}

function predictTrackBox(track, time) {
  const dt = Math.max(0, Math.min(0.25, time - track.lastUpdate));
  return clampBox({
    x: track.box.x + track.velocity.x * dt,
    y: track.box.y + track.velocity.y * dt,
    width: track.box.width + track.velocity.width * dt,
    height: track.box.height + track.velocity.height * dt,
  });
}

function blendVelocity(a, b, amount) {
  return {
    x: a.x * (1 - amount) + b.x * amount,
    y: a.y * (1 - amount) + b.y * amount,
    width: a.width * (1 - amount) + b.width * amount,
    height: a.height * (1 - amount) + b.height * amount,
  };
}

function scaleVelocity(velocity, amount) {
  return {
    x: velocity.x * amount,
    y: velocity.y * amount,
    width: velocity.width * amount,
    height: velocity.height * amount,
  };
}

function blendSignature(a, b, amount) {
  return a.map((value, index) => value * (1 - amount) + b[index] * amount);
}

function mergeMosaicBoxes(time, detectedFaces) {
  const boxes = [];

  detectedFaces.forEach(({ box }) => {
    const signature = makeSignature(renderCtx, box);
    if (isProtected(signature)) return;
    boxes.push({ box, force: false });
  });

  candidateBoxesAt(time, "mosaic").forEach((box) => {
    boxes.push({ box, force: true });
  });

  return dedupeBoxes(boxes);
}

function candidateBoxesAt(time, category) {
  return state.candidates
    .filter((candidate) => (category === "mosaic" ? candidate.category !== "protect" : candidate.category === category))
    .map((candidate) => nearestSampleBox(candidate, time))
    .filter(Boolean);
}

function nearestSampleBox(candidate, time) {
  const samples = candidate.samples || [{ time: candidate.time, box: candidate.box }];
  const sorted = [...samples].sort((a, b) => a.time - b.time);
  const before = [...sorted].reverse().find((sample) => sample.time <= time);
  const after = sorted.find((sample) => sample.time >= time);

  if (before && after) {
    const gap = after.time - before.time;
    if (gap === 0) return before.box;
    if (gap <= SAMPLE_INTERPOLATE_SECONDS) {
      return interpolateBox(before.box, after.box, (time - before.time) / gap);
    }
  }

  let closest = sorted[0];
  sorted.forEach((sample) => {
    if (Math.abs(sample.time - time) < Math.abs(closest.time - time)) closest = sample;
  });

  if (!closest || Math.abs(closest.time - time) > SAMPLE_HOLD_SECONDS) return null;
  return closest.box;
}

function interpolateBox(a, b, amount) {
  const t = clamp(amount, 0, 1);
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    width: a.width + (b.width - a.width) * t,
    height: a.height + (b.height - a.height) * t,
  };
}

function dedupeBoxes(entries) {
  const result = [];
  entries.forEach((entry) => {
    const duplicate = result.some(
      (kept) => boxOverlap(kept.box, entry.box) > 0.42 || centerDistanceScore(kept.box, entry.box) > 0.82,
    );
    if (!duplicate) result.push(entry);
  });
  return result;
}

function boxOverlap(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  return intersection / Math.max(1, Math.min(areaA, areaB));
}

function boxIou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - intersection;
  return intersection / Math.max(1, union);
}

function centerDistanceScore(a, b) {
  const ax = a.x + a.width / 2;
  const ay = a.y + a.height / 2;
  const bx = b.x + b.width / 2;
  const by = b.y + b.height / 2;
  const distance = Math.hypot(ax - bx, ay - by);
  const scale = Math.max(1, Math.max(a.width, a.height, b.width, b.height));
  return Math.max(0, 1 - distance / scale);
}

function isProtected(signature) {
  const threshold = Number(els.matchThreshold.value) / 100;
  return state.protected.some((face) => similarity(signature, face.signature) >= threshold);
}

function mosaicBox(box, pixelSize) {
  const padded = padBox(box, 0.12);
  const width = Math.max(1, Math.floor(padded.width / pixelSize));
  const height = Math.max(1, Math.floor(padded.height / pixelSize));
  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const ctx = scratch.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    els.renderCanvas,
    padded.x,
    padded.y,
    padded.width,
    padded.height,
    0,
    0,
    width,
    height,
  );
  renderCtx.save();
  clipFaceShape(renderCtx, padded);
  renderCtx.imageSmoothingEnabled = false;
  renderCtx.drawImage(scratch, 0, 0, width, height, padded.x, padded.y, padded.width, padded.height);
  renderCtx.imageSmoothingEnabled = true;
  renderCtx.restore();
}

function clipFaceShape(ctx, box) {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const rx = box.width * 0.48;
  const ry = box.height * 0.54;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.clip();
}

function captureFrame() {
  resizeCanvases();
  renderCtx.drawImage(els.video, 0, 0, els.renderCanvas.width, els.renderCanvas.height);
}

function updateOriginalPreview() {
  resizePreviewCanvases();
  originalPreviewCtx.drawImage(els.video, 0, 0, els.originalPreview.width, els.originalPreview.height);
}

function updateMosaicPreview() {
  resizePreviewCanvases();
  mosaicPreviewCtx.drawImage(
    els.renderCanvas,
    0,
    0,
    els.mosaicPreview.width,
    els.mosaicPreview.height,
  );
}

function cropFace(box) {
  const padded = padBox(box, 0.12);
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(
    els.renderCanvas,
    padded.x,
    padded.y,
    padded.width,
    padded.height,
    0,
    0,
    canvas.width,
    canvas.height,
  );
  return canvas.toDataURL("image/jpeg", 0.82);
}

function makeSignature(ctx, box) {
  const padded = padBox(box, 0.08);
  const cells = 4;
  const values = [];
  for (let y = 0; y < cells; y += 1) {
    for (let x = 0; x < cells; x += 1) {
      const sx = Math.floor(padded.x + (padded.width * x) / cells);
      const sy = Math.floor(padded.y + (padded.height * y) / cells);
      const sw = Math.max(1, Math.floor(padded.width / cells));
      const sh = Math.max(1, Math.floor(padded.height / cells));
      const data = ctx.getImageData(sx, sy, sw, sh).data;
      let r = 0;
      let g = 0;
      let b = 0;
      const count = data.length / 4;
      for (let i = 0; i < data.length; i += 4) {
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
      }
      values.push(r / count / 255, g / count / 255, b / count / 255);
    }
  }
  return values;
}

function similarity(a, b) {
  if (!a.length || a.length !== b.length) return 0;
  let distance = 0;
  for (let i = 0; i < a.length; i += 1) {
    distance += Math.abs(a[i] - b[i]);
  }
  return Math.max(0, 1 - distance / a.length);
}

function nearestAnnotations(time) {
  if (state.annotations.length === 0) return [];
  return state.annotations
    .filter((entry) => Math.abs(entry.time - time) < 1.2)
    .map((entry, index) => ({ id: `annotation-${index}`, box: entry.box, kind: "manual" }));
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  state.faces.forEach((face) => drawBox(face.box, face.kind === "manual" ? "#ffcc5c" : "#28c2a0"));
  state.protected.forEach((face) => {
    if (Math.abs(face.time - els.video.currentTime) < 0.08) drawBox(face.box, "#ffffff", face.name);
  });
  if (state.manualDraft) drawBox(state.manualDraft, "#ffcc5c");
}

function drawBox(box, color, label = "") {
  overlayCtx.save();
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = Math.max(3, els.overlay.width / 420);
  overlayCtx.fillStyle = "rgba(0, 0, 0, 0.32)";
  overlayCtx.strokeRect(box.x, box.y, box.width, box.height);
  if (label) {
    overlayCtx.font = "700 18px sans-serif";
    const textWidth = overlayCtx.measureText(label).width + 16;
    overlayCtx.fillRect(box.x, Math.max(0, box.y - 30), textWidth, 26);
    overlayCtx.fillStyle = color;
    overlayCtx.fillText(label, box.x + 8, Math.max(19, box.y - 10));
  }
  overlayCtx.restore();
}

function eventToVideoPoint(event) {
  const rect = els.overlay.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * els.overlay.width,
    y: ((event.clientY - rect.top) / rect.height) * els.overlay.height,
  };
}

function pointInBox(point, box) {
  return (
    point.x >= box.x &&
    point.x <= box.x + box.width &&
    point.y >= box.y &&
    point.y <= box.y + box.height
  );
}

function boxFromPoints(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

function clampBox(box) {
  const x = clamp(box.x, 0, els.overlay.width);
  const y = clamp(box.y, 0, els.overlay.height);
  return {
    x,
    y,
    width: clamp(box.width, 0, els.overlay.width - x),
    height: clamp(box.height, 0, els.overlay.height - y),
  };
}

function padBox(box, ratio) {
  const padX = box.width * ratio;
  const padY = box.height * ratio;
  return clampBox({
    x: box.x - padX,
    y: box.y - padY,
    width: box.width + padX * 2,
    height: box.height + padY * 2,
  });
}

function syncTimeline() {
  els.timeline.value = String(els.video.currentTime || 0);
  els.currentTime.textContent = formatTime(els.video.currentTime || 0);
  drawOverlay();
}

function updateSliderLabels() {
  els.pixelSizeValue.textContent = els.pixelSize.value;
  els.matchThresholdValue.textContent = els.matchThreshold.value;
}

function setDetectorState(kind, text) {
  els.detectorState.className = `state-dot ${kind}`;
  els.detectorState.textContent = text;
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function waitForEvent(target, eventName) {
  return new Promise((resolve) => target.addEventListener(eventName, resolve, { once: true }));
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function seekVideo(time) {
  return new Promise((resolve) => {
    const target = clamp(time, 0, els.video.duration || 0);
    if (Math.abs(els.video.currentTime - target) < 0.01) {
      resolve();
      return;
    }
    els.video.addEventListener("seeked", resolve, { once: true });
    els.video.currentTime = target;
  });
}

function formatTime(seconds = 0) {
  if (!Number.isFinite(seconds)) return "00:00.0";
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
