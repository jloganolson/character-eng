import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import loadMujoco from "@mujoco/mujoco";

const MODEL_CACHE = new Map();
let mujocoPromise = null;
let currentViewer = null;
let latestState = null;
let inflight = false;
let currentTrajectoryKey = "";

const heroHeadline = document.getElementById("hero-headline");
const heroSummary = document.getElementById("hero-summary");
const heroBodyMode = document.getElementById("hero-body-mode");
const heroBodySummary = document.getElementById("hero-body-summary");
const heroEyes = document.getElementById("hero-eyes");
const heroEyesDetail = document.getElementById("hero-eyes-detail");
const heroDispatch = document.getElementById("hero-dispatch");
const heroDispatchDetail = document.getElementById("hero-dispatch-detail");
const viewerEngine = document.getElementById("viewer-engine");
const viewerCanvas = document.getElementById("viewer-canvas");
const viewerStatus = document.getElementById("viewer-status");
const lastUpdated = document.getElementById("last-updated");
const viewportTitle = document.getElementById("viewport-title");
const viewportCopy = document.getElementById("viewport-copy");
const clipKey = document.getElementById("clip-key");
const clipDetail = document.getElementById("clip-detail");
const keepaliveState = document.getElementById("keepalive-state");
const keepaliveDetail = document.getElementById("keepalive-detail");
const faceProgram = document.getElementById("face-program");
const faceDetail = document.getElementById("face-detail");
const interactionState = document.getElementById("interaction-state");
const interactionDetail = document.getElementById("interaction-detail");
const actionStatus = document.getElementById("action-status");
const dispatchAuthority = document.getElementById("dispatch-authority");
const dispatchMode = document.getElementById("dispatch-mode");
const dispatchPreview = document.getElementById("dispatch-preview");
const dispatchHardware = document.getElementById("dispatch-hardware");
const dispatchReady = document.getElementById("dispatch-ready");
const rawState = document.getElementById("raw-state");
const buttons = Array.from(document.querySelectorAll("button[data-action]"));

function setStatus(message) {
  if (viewerStatus) viewerStatus.textContent = message;
}

function formatAge(timestamp) {
  if (!timestamp) return "never";
  const age = Math.max(0, Date.now() / 1000 - timestamp);
  return `${age.toFixed(1)}s ago`;
}

function statusTone(state) {
  if (state === "ok" || state === "active") return "status-ok";
  if (state === "error") return "status-err";
  return "status-warn";
}

function setStatusClass(element, state) {
  if (!element) return;
  element.classList.remove("status-ok", "status-warn", "status-err");
  element.classList.add(statusTone(state));
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function getMujoco() {
  if (window.location.hostname === "0.0.0.0") {
    const canonicalUrl = new URL(window.location.href);
    canonicalUrl.hostname = "localhost";
    window.location.replace(canonicalUrl.toString());
    throw new Error("Redirecting to localhost so MuJoCo can use SharedArrayBuffer.");
  }

  if (!window.crossOriginIsolated) {
    throw new Error("MuJoCo requires a cross-origin isolated page. Open this preview on localhost.");
  }

  if (!mujocoPromise) {
    mujocoPromise = loadMujoco({
      locateFile: (file) => `/node_modules/@mujoco/mujoco/${file}`,
    });
  }
  return mujocoPromise;
}

function ensureFsDirectories(mujoco, fullPath) {
  const parts = fullPath.split("/").filter(Boolean);
  let current = "";
  for (let i = 0; i < parts.length - 1; i += 1) {
    current += `/${parts[i]}`;
    const stat = mujoco.FS.analyzePath(current);
    if (!stat.exists) mujoco.FS.mkdir(current);
  }
}

async function loadModelBundle(mujoco, robot) {
  if (MODEL_CACHE.has(robot)) return MODEL_CACHE.get(robot);

  setStatus("Loading MuJoCo model files...");
  const manifest = await fetchJson(`/robot-preview/mujoco/model?robot=${encodeURIComponent(robot)}`);
  const root = `/models/${robot}`;
  mujoco.FS.mkdirTree(root);

  for (let i = 0; i < manifest.files.length; i += 1) {
    const relPath = manifest.files[i].path;
    const fsPath = `${root}/${relPath}`;
    if (!mujoco.FS.analyzePath(fsPath).exists) {
      setStatus(`Loading model assets (${i + 1}/${manifest.files.length})...`);
      const response = await fetch(`/robot-preview/mujoco/file?path=${encodeURIComponent(relPath)}`, {cache: "force-cache"});
      if (!response.ok) throw new Error(`Failed to load model asset: ${relPath}`);
      ensureFsDirectories(mujoco, fsPath);
      mujoco.FS.writeFile(fsPath, new Uint8Array(await response.arrayBuffer()));
    }
  }

  setStatus("Compiling MuJoCo model...");
  const model = mujoco.MjModel.from_xml_path(`${root}/${manifest.entry_path}`);
  const bundle = {model, root, manifest};
  MODEL_CACHE.set(robot, bundle);
  return bundle;
}

function createGeometryFromMesh(model, meshId) {
  const meshInfo = model.mesh(meshId);
  const vertStart = Number(meshInfo.vertadr);
  const vertCount = Number(meshInfo.vertnum);
  const faceStart = Number(meshInfo.faceadr);
  const faceCount = Number(meshInfo.facenum);
  const vertices = model.mesh_vert.slice(vertStart * 3, (vertStart + vertCount) * 3);
  const faces = model.mesh_face.slice(faceStart * 3, (faceStart + faceCount) * 3);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(Array.from(vertices), 3));
  geometry.setIndex(Array.from(faces));
  geometry.computeVertexNormals();
  return geometry;
}

function createVisualScene(mujoco, model, data, canvas) {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1016);

  const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(2.1, -2.4, 1.5);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.55);

  scene.add(new THREE.AmbientLight(0xffffff, 1.15));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
  keyLight.position.set(4, -3, 6);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x7de2d1, 0.42);
  fillLight.position.set(-3, 2, 4);
  scene.add(fillLight);
  const floor = new THREE.GridHelper(4, 20, 0x2f4255, 0x17222f);
  floor.rotation.x = Math.PI / 2;
  scene.add(floor);

  const visualMeshes = [];
  for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
    const geom = model.geom(geomId);
    if (Number(geom.group) !== 1) continue;
    if (Number(geom.type) !== mujoco.mjtGeom.mjGEOM_MESH.value) continue;
    const meshId = Number(geom.dataid);
    if (meshId < 0) continue;
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xc8d3df),
      metalness: 0.12,
      roughness: 0.76,
    });
    const mesh = new THREE.Mesh(createGeometryFromMesh(model, meshId), material);
    mesh.matrixAutoUpdate = false;
    scene.add(mesh);
    visualMeshes.push({geomId, mesh});
  }

  const tempMatrix = new THREE.Matrix4();
  const tempQuat = new THREE.Quaternion();

  function resize() {
    const width = canvas.clientWidth || canvas.parentElement?.clientWidth || 640;
    const height = canvas.clientHeight || canvas.parentElement?.clientHeight || 480;
    if (canvas.width !== width || canvas.height !== height) {
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
  }

  function syncGeoms() {
    for (const {geomId, mesh} of visualMeshes) {
      const posOffset = geomId * 3;
      const matOffset = geomId * 9;
      mesh.position.set(
        data.geom_xpos[posOffset],
        data.geom_xpos[posOffset + 1],
        data.geom_xpos[posOffset + 2],
      );
      tempMatrix.set(
        data.geom_xmat[matOffset],
        data.geom_xmat[matOffset + 1],
        data.geom_xmat[matOffset + 2],
        0,
        data.geom_xmat[matOffset + 3],
        data.geom_xmat[matOffset + 4],
        data.geom_xmat[matOffset + 5],
        0,
        data.geom_xmat[matOffset + 6],
        data.geom_xmat[matOffset + 7],
        data.geom_xmat[matOffset + 8],
        0,
        0,
        0,
        0,
        1,
      );
      tempQuat.setFromRotationMatrix(tempMatrix);
      mesh.quaternion.copy(tempQuat);
      mesh.updateMatrix();
    }
  }

  return {
    renderer,
    scene,
    camera,
    controls,
    resize,
    syncGeoms,
    dispose() {
      controls.dispose();
      renderer.dispose();
      for (const {mesh} of visualMeshes) {
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
    },
  };
}

function destroyViewer() {
  if (!currentViewer) return;
  cancelAnimationFrame(currentViewer.rafId);
  currentViewer.resizeObserver?.disconnect();
  currentViewer.three.dispose();
  currentViewer.data.delete();
  currentViewer = null;
}

function buildJointAddressMap(mujoco, model, jointNames) {
  return jointNames.map((name) => {
    const jointId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
    if (jointId < 0) throw new Error(`Unknown MuJoCo joint: ${name}`);
    return Number(model.jnt_qposadr[jointId]);
  });
}

function applyFrame(viewer, frameIndex) {
  const {mujoco, model, data, trajectory, jointQposAdr} = viewer;
  const qpos = data.qpos;
  qpos[0] = trajectory.root_pos[0];
  qpos[1] = trajectory.root_pos[1];
  qpos[2] = trajectory.root_pos[2];
  qpos[3] = trajectory.root_quat_wxyz[0];
  qpos[4] = trajectory.root_quat_wxyz[1];
  qpos[5] = trajectory.root_quat_wxyz[2];
  qpos[6] = trajectory.root_quat_wxyz[3];

  const pose = trajectory.frames[frameIndex] || [];
  for (let i = 0; i < jointQposAdr.length; i += 1) {
    qpos[jointQposAdr[i]] = pose[i] ?? 0;
  }

  mujoco.mj_forward(model, data);
  viewer.three.syncGeoms();
}

async function loadTrajectory(payload) {
  const body = payload.body || {};
  const clip = body.active_clip || {};
  return fetchJson(
    `/robot-preview/trajectory?mode=${encodeURIComponent(body.mode || "")}&clip=${encodeURIComponent(clip.key || "")}`,
  );
}

async function ensureViewer(payload) {
  const trajectoryKey = [
    payload.body?.mode || "",
    payload.body?.active_clip?.key || "",
    payload.wave?.started_at || 0,
    payload.fistbump?.updated_at || 0,
    payload.fistbump?.state || "",
  ].join("|");

  if (currentViewer && trajectoryKey === currentTrajectoryKey) return;
  currentTrajectoryKey = trajectoryKey;

  const mujoco = await getMujoco();
  const [bundle, trajectory] = await Promise.all([
    loadModelBundle(mujoco, "booster_k1"),
    loadTrajectory(payload),
  ]);

  destroyViewer();
  const data = new mujoco.MjData(bundle.model);
  const three = createVisualScene(mujoco, bundle.model, data, viewerCanvas);
  const jointQposAdr = buildJointAddressMap(mujoco, bundle.model, trajectory.joint_names || []);
  currentViewer = {
    mujoco,
    model: bundle.model,
    data,
    three,
    trajectory,
    jointQposAdr,
    startedAtMs: performance.now(),
    rafId: 0,
    resizeObserver: new ResizeObserver(() => three.resize()),
  };
  currentViewer.resizeObserver.observe(viewerCanvas);

  const tick = () => {
    if (!currentViewer) return;
    const frames = currentViewer.trajectory.frames || [];
    if (!frames.length) return;
    const elapsed = (performance.now() - currentViewer.startedAtMs) / 1000;
    const frameFloat = elapsed * currentViewer.trajectory.fps;
    let frameIndex = Math.floor(frameFloat);
    if (currentViewer.trajectory.loop) {
      frameIndex %= frames.length;
    } else if (currentViewer.trajectory.hold_last_frame) {
      frameIndex = Math.min(frames.length - 1, frameIndex);
    } else if (frameIndex >= frames.length) {
      frameIndex = frames.length - 1;
    }
    applyFrame(currentViewer, Math.max(0, frameIndex));
    currentViewer.three.resize();
    currentViewer.three.controls.update();
    currentViewer.three.renderer.render(currentViewer.three.scene, currentViewer.three.camera);
    currentViewer.rafId = requestAnimationFrame(tick);
  };

  setStatus(trajectory.summary || "Trajectory loaded.");
  tick();
}

function render(payload) {
  latestState = payload;
  const body = payload.body || {};
  const clip = body.active_clip || {};
  const keepalive = payload.keepalive || {};
  const face = payload.face || {};
  const interaction = payload.interaction || {};
  const eyes = payload.rust_eyes || {};
  const dispatch = payload.dispatch || {};

  heroHeadline.textContent = payload.headline || "Idle";
  heroSummary.textContent = payload.source ? `Source: ${payload.source}` : "Preview state loaded.";
  heroBodyMode.textContent = body.mode || "idle_keepalive";
  heroBodySummary.textContent = body.summary || "No active body program.";
  heroEyes.textContent = eyes.state || "disabled";
  heroEyesDetail.textContent = eyes.expression_key || eyes.message || "rust-eyes bridge is optional.";
  heroDispatch.textContent = dispatch.mode || "preview_only";
  heroDispatchDetail.textContent = dispatch.summary || "";
  setStatusClass(heroEyes, eyes.state || "disabled");

  viewerEngine.textContent = dispatch.preview_transport || "browser_mujoco_pending";
  lastUpdated.textContent = formatAge(Number(payload.updated_at || 0));
  viewportTitle.textContent = clip.label ? `${clip.label} Program` : "MuJoCo Viewer Mount";
  viewportCopy.textContent = body.summary || "Waiting for compiled preview state.";

  clipKey.textContent = clip.key || "idle.keepalive";
  clipDetail.textContent = clip.playback
    ? `${clip.playback}${clip.seconds_left ? ` · ${Number(clip.seconds_left).toFixed(1)}s left` : ""}`
    : "Looped keepalive program.";
  keepaliveState.textContent = keepalive.state || "inactive";
  keepaliveDetail.textContent = `${keepalive.preset || "idle_attentive_v1"} · ${keepalive.summary || ""}`;
  faceProgram.textContent = `${face.expression || "neutral"} · ${face.gaze || "scene"}`;
  faceDetail.textContent = `${face.gaze_type || "hold"}${eyes.expression_key ? ` · eyes ${eyes.expression_key}` : ""}`;
  interactionState.textContent = interaction.state || "idle";
  interactionDetail.textContent = interaction.note || "No active interaction.";
  setStatusClass(keepaliveState, keepalive.state || "inactive");
  setStatusClass(interactionState, interaction.active ? "active" : "disabled");

  dispatchAuthority.textContent = dispatch.authority || "character_eng";
  dispatchMode.textContent = dispatch.mode || "preview_only";
  dispatchPreview.textContent = dispatch.preview_transport || "browser_mujoco_pending";
  dispatchHardware.textContent = dispatch.hardware_transport || "booster_sdk_pending";
  dispatchReady.textContent = String(!!dispatch.ready_for_hardware);
  rawState.textContent = JSON.stringify(payload, null, 2);
}

async function fetchState() {
  if (inflight) return;
  inflight = true;
  try {
    const payload = await fetchJson("/robot-preview/state", {cache: "no-store"});
    render(payload);
    await ensureViewer(payload);
  } catch (error) {
    actionStatus.textContent = `Preview state fetch failed: ${error.message || error}`;
    setStatus(`Viewer unavailable: ${error.message || error}`);
  } finally {
    inflight = false;
  }
}

async function postAction(action) {
  const sessionId = (((latestState || {}).interaction || {}).session_id || "").trim();
  buttons.forEach((button) => {
    button.disabled = true;
  });
  try {
    const payload = await fetchJson("/robot-preview/action", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        action,
        session_id: sessionId,
        source: "robot_preview",
      }),
    });
    render(payload);
    await ensureViewer(payload);
    actionStatus.textContent = `Action applied: ${action}`;
  } catch (error) {
    actionStatus.textContent = `Action failed (${action}): ${error.message || error}`;
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

buttons.forEach((button) => {
  button.addEventListener("click", () => postAction(button.dataset.action || "refresh"));
});

fetchState();
window.setInterval(fetchState, 800);
window.addEventListener("beforeunload", () => destroyViewer());
