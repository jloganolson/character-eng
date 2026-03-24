"""Vision service for character-eng — Flask + MJPEG streaming.

Provides camera capture, face tracking (InsightFace), person tracking (SAM3 + ReID),
and VLM questioning (vLLM). Exposes structured JSON endpoints for character-eng polling.
"""

import argparse
import base64
import glob
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
except ImportError:
    sys.exit(
        "ERROR: Vision service dependencies not found.\n"
        "Run from the vision directory: cd services/vision && uv run python app.py\n"
        "Or use: uv run --project services/vision python app.py"
    )

from flask import Flask, Response, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from character_eng.vision.vlm import VLMTaskSpec

# ---------------------------------------------------------------------------
# Config (overridden by CLI args in __main__)
# ---------------------------------------------------------------------------
PORT = 7860
DEVICE = "cuda"
DTYPE = torch.bfloat16

_vllm_client = None
_vllm_model_id = None  # set from --vllm-model or auto-detected
_vllm_image_max_side = 384
_camera_names: dict[str, str] = {}  # e.g. {"0": "Front", "2": "Arm"}
_capture_resolution: tuple[int, int] | None = None  # (width, height)

# ---------------------------------------------------------------------------
# Webcam capture (background thread)
# ---------------------------------------------------------------------------
class WebcamCapture:
    def __init__(self, device: str = "/dev/video0"):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._device = device
        self._resolution: tuple[int, int] | None = None
        self._external_only = False

    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            print(f"Warning: Cannot open {self._device}")
            return
        if self._resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get_frame(self) -> np.ndarray | None:
        """Return latest frame in BGR (for cv2.imencode)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_rgb(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)

    def switch_device(self, path: str) -> None:
        self.release()
        self._device = path
        self.start()

    def set_resolution(self, width: int, height: int) -> None:
        self._resolution = (width, height)
        if self._cap is not None and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def inject(self, frame: np.ndarray) -> None:
        """Inject a frame from an external source (browser camera)."""
        with self._lock:
            self._frame = frame

    def set_input_mode(self, mode: str) -> None:
        with self._lock:
            self._external_only = str(mode or "camera").lower() == "external"

    def input_mode(self) -> str:
        with self._lock:
            return "external" if self._external_only else "camera"

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._frame = None

    def _loop(self) -> None:
        while self._running and self._cap is not None:
            if self._external_only:
                time.sleep(0.01)
                continue
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame  # BGR
            else:
                time.sleep(0.01)


_sam3_model = None
_sam3_proc = None
_sam3_status = "not_loaded"  # "not_loaded" | "loading" | "ready" | "error"
_sam3_error = ""

def _load_sam3():
    global _sam3_model, _sam3_proc, _sam3_status, _sam3_error
    if _sam3_model is not None:
        return
    if _sam3_status == "loading":
        while _sam3_status == "loading":
            time.sleep(0.2)
        if _sam3_model is not None:
            return
        raise RuntimeError(f"SAM3 failed to load: {_sam3_error}")
    _sam3_status = "loading"
    try:
        print("Loading SAM3 …", flush=True)
        try:
            from transformers import Sam3Processor as S3P, Sam3Model as S3M

            _sam3_proc = S3P.from_pretrained("facebook/sam3")
            _sam3_model = S3M.from_pretrained(
                "facebook/sam3", torch_dtype=DTYPE,
            ).to(DEVICE)
            _sam3_model.eval()
            print("SAM3 backend: transformers", flush=True)
        except (ImportError, AttributeError):
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor

            _sam3_model = build_sam3_image_model(device=DEVICE, eval_mode=True)
            if DTYPE is not None:
                _sam3_model = _sam3_model.to(dtype=DTYPE)
            _sam3_model.eval()
            _sam3_proc = NativeSam3Processor(_sam3_model, device=DEVICE)
            print("SAM3 backend: native sam3 package", flush=True)
        _sam3_status = "ready"
        print("SAM3 ready.", flush=True)
    except Exception as e:
        _sam3_status = "error"
        _sam3_error = str(e)
        print(f"SAM3 load error: {e}", flush=True)
        raise


# ---------------------------------------------------------------------------
# VLM inference (vLLM only)
# ---------------------------------------------------------------------------
_slots: dict[str, dict] = {}  # slot_id -> {question, response, timing, done, loop, stop}
_slot_lock = threading.Lock()
_next_slot_id = 0

# --- Constant/ephemeral VLM questions (set via /set_questions API) ---
_constant_questions: list[dict] = []
_ephemeral_questions: list[dict] = []
_questions_lock = threading.Lock()
_question_answers: dict[str, dict] = {}  # task_id -> {answer, elapsed, slot_type, timestamp}
_question_answer_seq = 0

# --- Constant/ephemeral SAM3 targets (set via /set_sam_targets API) ---
_constant_sam_targets: list[str] = ["person"]
_ephemeral_sam_targets: list[str] = []
_sam_targets_lock = threading.Lock()

# --- Managed question worker (round-robin over constant + ephemeral) ---
_managed_worker_running = False
_managed_worker_thread: threading.Thread | None = None
_snapshot_seq = 0

def run_vlm_vllm(frame_rgb: np.ndarray, question: str, max_tokens: int) -> tuple[str, float]:
    """Run VLM inference via vLLM OpenAI-compatible API. Returns (answer, elapsed)."""
    pil_img = Image.fromarray(frame_rgb)
    t0 = time.perf_counter()

    # encode image as data URL for vLLM
    w, h = pil_img.size
    max_side = _vllm_image_max_side
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    resp = _vllm_client.chat.completions.create(
        model=_vllm_model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": question},
        ]}],
    )

    answer = resp.choices[0].message.content.strip()
    elapsed = time.perf_counter() - t0
    return answer, elapsed


# ---------------------------------------------------------------------------
# Face tracking (optional — graceful if insightface not installed)
# ---------------------------------------------------------------------------
ft = None  # type: ignore

# ---------------------------------------------------------------------------
# Person tracking (experimental — SAM3 + ReID)
# ---------------------------------------------------------------------------
pt = None  # type: ignore

# ---------------------------------------------------------------------------
# V4L2 helpers
# ---------------------------------------------------------------------------
def _v4l2_device_name(dev_path: str) -> str:
    """Read hardware name via v4l2-ctl (returns '' on failure)."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--device", dev_path, "--info"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode()
        for line in out.splitlines():
            if "Card type" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _camera_label(dev_path: str) -> str:
    """Return best available label: user name > V4L2 name > path."""
    # Check user-provided names (keyed by index or full path)
    for key in [dev_path, dev_path.replace("/dev/video", "")]:
        if key in _camera_names:
            return _camera_names[key]
    v4l2 = _v4l2_device_name(dev_path)
    return v4l2 if v4l2 else dev_path


# ---------------------------------------------------------------------------
# Memory status helper
# ---------------------------------------------------------------------------
def _memory_status() -> dict:
    """Return GPU/CPU memory usage info."""
    info: dict = {}
    # GPU (works for both discrete and Jetson unified memory)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        info["gpu"] = {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "total_mb": round(total, 1),
            "free_mb": round(total - allocated, 1),
        }
    # CPU / system memory (especially useful for Jetson unified memory)
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])  # kB
        total_mb = meminfo.get("MemTotal", 0) / 1024
        free_mb = meminfo.get("MemAvailable", 0) / 1024
        info["cpu"] = {
            "total_mb": round(total_mb, 1),
            "free_mb": round(free_mb, 1),
            "used_mb": round(total_mb - free_mb, 1),
        }
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
cam = WebcamCapture()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    # Model display name for status bar
    model_display = _vllm_model_id or "none"
    if "/" in model_display:
        model_display = model_display.split("/")[-1]

    # Show/hide VLM form based on whether vLLM is configured
    vlm_enabled = _vllm_client is not None

    return f"""<!DOCTYPE html>
<html><head><title>VLM Webcam QA</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 12px; }}
  .status-bar {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 8px 12px; background: #1a1a2e; border-radius: 6px; margin-bottom: 8px; }}
  .status-chip {{ display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 4px; font-size: 0.8em; font-weight: 500; color: #ccc; background: #2a2a3e; cursor: pointer; user-select: none; border: 1px solid transparent; }}
  .status-chip:hover {{ border-color: #555; }}
  .status-chip.no-toggle {{ cursor: default; }}
  .status-chip.no-toggle:hover {{ border-color: transparent; }}
  .status-chip .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .dot-off {{ background: #555; }}
  .dot-loading {{ background: #f59e0b; animation: pulse 1s infinite; }}
  .dot-ready {{ background: #22c55e; }}
  .dot-error {{ background: #ef4444; }}
  .dot-unavail {{ background: #555; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
  .status-chip .err {{ color: #f87171; font-weight: normal; font-size: 0.85em; }}
  .mem-bar {{ padding: 4px 12px; background: #1a1a2e; border-radius: 6px; margin-bottom: 12px; font-size: 0.75em; color: #aaa; display: flex; gap: 16px; align-items: center; }}
  .mem-bar .mem-fill {{ height: 6px; border-radius: 3px; transition: width 0.5s; }}
  .mem-bar .mem-track {{ flex: 1; max-width: 200px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }}
  .controls {{ margin: 8px 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ margin-right: 0; }}
  #feed {{ display: block; background: #000; border-radius: 4px; }}
  #response {{ background: #f0f0f0; padding: 10px; min-height: 60px; white-space: pre-wrap; border-radius: 4px; }}
  .det-panel {{ background: #f8f8f8; padding: 6px 10px; margin-top: 4px; font-size: 0.9em; border-radius: 4px; }}
  .btn-sm {{ padding: 2px 8px; font-size: 0.85em; cursor: pointer; }}
</style>
</head>
<body>
<h2 style="margin-top:0;">VLM Webcam QA</h2>

<div class="status-bar" id="status_bar">
  <span class="status-chip no-toggle"><span class="dot dot-off" id="dot_vllm"></span> vLLM (<span id="model_name">{model_display}</span>) <span id="lbl_vllm"></span></span>
  <span class="status-chip" onclick="toggleTracker('sam3')" title="Click to toggle SAM3"><span class="dot dot-off" id="dot_sam3"></span> SAM3 <span id="lbl_sam3"></span></span>
  <span class="status-chip" onclick="toggleTracker('face')" title="Click to toggle InsightFace"><span class="dot dot-off" id="dot_face"></span> InsightFace <span id="lbl_face"></span></span>
  <span class="status-chip" onclick="toggleTracker('person')" title="Click to toggle Person tracker"><span class="dot dot-off" id="dot_reid"></span> ReID (ResNet50) <span id="lbl_reid"></span></span>
</div>

<div class="mem-bar" id="mem_bar">
  <span id="mem_gpu_label">GPU: --</span>
  <span class="mem-track"><span class="mem-fill" id="mem_gpu_fill" style="width:0%;background:#22c55e;"></span></span>
  <span id="mem_cpu_label">RAM: --</span>
  <span class="mem-track"><span class="mem-fill" id="mem_cpu_fill" style="width:0%;background:#3b82f6;"></span></span>
</div>

<div class="controls">
  <label>Camera:
    <select id="cam_sel" onchange="switchCam(this.value)"></select>
  </label>
  <button class="btn-sm" onclick="refreshCameras()">Refresh</button>
  <label>Resolution:
    <select id="res_sel" onchange="setResolution(this.value)">
      <option value="">Default</option>
      <option value="640x480">640x480</option>
      <option value="1280x720">1280x720</option>
      <option value="1920x1080">1920x1080</option>
      <option value="320x240">320x240</option>
    </select>
  </label>
</div>

<img id="feed" src="/video_feed" width="640" height="480">

<br>
<div id="vlm_form" style="display:{'block' if vlm_enabled else 'none'};">
<div id="slots_container"></div>
<button onclick="addSlot()" style="margin:6px 0;">+ Add Question</button>
</div>

<div class="controls">
  <label>SAM3 extra prompts:
    <input type="text" id="sam3_prompt" value="" size="30" placeholder="e.g. laptop, cup, phone">
  </label>
  <label><input type="checkbox" id="inject_ctx" checked> Inject tracker context into VLM prompt</label>
</div>

<div class="det-panel">
  <strong>Faces (InsightFace):</strong> <span id="face_info">--</span>
</div>
<div class="det-panel">
  <strong>Persons (SAM3+ReID):</strong> <span id="person_info">--</span>
</div>


<script>
// --- Tracker toggle state (for clickable chips) ---
var _trackerState = {{face: false, person: false}};

function toggleTracker(which) {{
  if (which === "sam3") {{
    // SAM3 preload toggle
    fetch("/preload_sam3", {{method:"POST"}});
    return;
  }}
  var endpoint = which === "face" ? "/toggle_face_tracking" : "/toggle_person_tracking";
  var current = _trackerState[which];
  fetch(endpoint, {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{enabled: !current}})}})
    .then(r=>r.json()).then(function(d) {{
      if (d.enabled !== undefined) _trackerState[which] = d.enabled;
    }});
}}

// --- Status bar ---
function _setDot(id, state) {{
  document.getElementById(id).className = "dot dot-" + state;
}}
function _pollStatus() {{
  fetch("/model_status").then(r=>r.json()).then(function(d) {{
    _setDot("dot_vllm", d.vllm === "ready" ? "ready" : d.vllm === "error" ? "error" : "off");
    document.getElementById("lbl_vllm").textContent = d.vllm === "ready" ? "" : d.vllm === "off" ? "(disabled)" : "(down)";
    if (d.vllm_model) document.getElementById("model_name").textContent = d.vllm_model;

    var s3 = d.sam3;
    _setDot("dot_sam3", s3.status === "ready" ? "ready" : s3.status === "loading" ? "loading" : s3.status === "error" ? "error" : "off");
    document.getElementById("lbl_sam3").innerHTML = s3.status === "loading" ? "(loading...)" : s3.status === "error" ? '<span class="err">(' + s3.error.substring(0,50) + ')</span>' : "";

    var fc = d.face;
    _setDot("dot_face", fc.status === "ready" ? "ready" : fc.status === "loading" ? "loading" : fc.status === "error" ? "error" : fc.status === "unavailable" ? "unavail" : "off");
    document.getElementById("lbl_face").innerHTML = fc.status === "loading" ? "(loading...)" : fc.status === "error" ? '<span class="err">(' + fc.error.substring(0,50) + ')</span>' : "";
    _trackerState.face = fc.enabled || false;

    var pt = d.person;
    _setDot("dot_reid", pt.status === "ready" ? "ready" : pt.status === "loading" ? "loading" : pt.status === "error" ? "error" : pt.status === "unavailable" ? "unavail" : "off");
    document.getElementById("lbl_reid").innerHTML = pt.status === "loading" ? "(loading...)" : pt.status === "error" ? '<span class="err">(' + pt.error.substring(0,50) + ')</span>' : "";
    _trackerState.person = pt.enabled || false;
  }}).catch(function(){{}});
}}
setInterval(_pollStatus, 2000);
_pollStatus();

// --- Memory bar ---
function _pollMemory() {{
  fetch("/memory_status").then(r=>r.json()).then(function(d) {{
    if (d.gpu) {{
      var pct = ((d.gpu.allocated_mb / d.gpu.total_mb) * 100).toFixed(0);
      document.getElementById("mem_gpu_label").textContent = "GPU: " + d.gpu.allocated_mb.toFixed(0) + " / " + d.gpu.total_mb.toFixed(0) + " MB";
      var fill = document.getElementById("mem_gpu_fill");
      fill.style.width = pct + "%";
      fill.style.background = pct > 85 ? "#ef4444" : pct > 65 ? "#f59e0b" : "#22c55e";
    }}
    if (d.cpu) {{
      var pct = ((d.cpu.used_mb / d.cpu.total_mb) * 100).toFixed(0);
      document.getElementById("mem_cpu_label").textContent = "RAM: " + d.cpu.used_mb.toFixed(0) + " / " + d.cpu.total_mb.toFixed(0) + " MB";
      var fill = document.getElementById("mem_cpu_fill");
      fill.style.width = pct + "%";
      fill.style.background = pct > 85 ? "#ef4444" : pct > 65 ? "#f59e0b" : "#3b82f6";
    }}
  }}).catch(function(){{}});
}}
setInterval(_pollMemory, 3000);
_pollMemory();

// --- Data pollers (always on) ---
setInterval(function() {{
  fetch("/face_data").then(r=>r.json()).then(function(d) {{
    var info = document.getElementById("face_info");
    if (!d.faces || d.faces.length === 0) {{ info.textContent = "none"; }}
    else {{ info.textContent = d.faces.map(function(f) {{ return f.identity + " (~" + f.age + f.gender + ", " + f.gaze_direction + ")"; }}).join("; "); }}
  }});
}}, 500);
setInterval(function() {{
  fetch("/person_data").then(r=>r.json()).then(function(d) {{
    var info = document.getElementById("person_info");
    if (!d.faces || d.faces.length === 0) {{ info.textContent = "none"; }}
    else {{ info.textContent = d.faces.map(function(f) {{ return f.identity + " (" + (f.confidence ? (f.confidence*100).toFixed(0) + "%%" : "") + ")"; }}).join("; "); }}
  }});
}}, 500);

// --- Camera ---
function switchCam(dev) {{
  fetch("/switch_camera", {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{device:dev}})}});
}}
function refreshCameras() {{
  fetch("/cameras").then(r=>r.json()).then(function(d) {{
    var sel = document.getElementById("cam_sel");
    var cur = sel.value;
    sel.innerHTML = "";
    d.cameras.forEach(function(c) {{
      var opt = document.createElement("option");
      opt.value = c.path;
      opt.textContent = c.label + " (" + c.path + ")";
      if (c.active) {{ opt.selected = true; opt.style.fontWeight = "bold"; }}
      sel.appendChild(opt);
    }});
    if (!sel.value && d.cameras.length) sel.value = d.cameras[0].path;
  }});
}}
// initial camera load
refreshCameras();

// --- Resolution ---
function setResolution(val) {{
  if (!val) return;
  var parts = val.split("x");
  fetch("/set_resolution", {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{width:parseInt(parts[0]), height:parseInt(parts[1])}})}});
}}

// --- SAM3 prompts ---
var _sam3Timer = null;
function _pushSam3() {{
  fetch("/set_sam3_prompts", {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{prompts: document.getElementById("sam3_prompt").value}})}});
}}
document.addEventListener("DOMContentLoaded", function() {{
  var el = document.getElementById("sam3_prompt");
  el.addEventListener("input", function() {{ clearTimeout(_sam3Timer); _sam3Timer = setTimeout(_pushSam3, 400); }});
  el.addEventListener("change", _pushSam3);
}});

// --- Multi-slot VLM Questions ---
var _slotCounter = 0;
var _activePolls = {{}};

function addSlot() {{
  var localId = _slotCounter++;
  var container = document.getElementById("slots_container");
  var div = document.createElement("div");
  div.id = "slot_" + localId;
  div.style.cssText = "border:1px solid #ccc; border-radius:6px; padding:8px; margin:6px 0; background:#fafafa;";
  div.innerHTML = '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">' +
    '<input type="text" class="slot-q" value="What&#39;s going on?" size="45" style="flex:1;min-width:200px;">' +
    '<label>Tokens:<input type="number" class="slot-tok" value="128" min="32" max="512" step="16" style="width:5em;"></label>' +
    '<label><input type="checkbox" class="slot-loop"> Loop</label>' +
    '<button class="slot-ask" onclick="slotAsk(' + localId + ')">Ask</button>' +
    '<button class="slot-stop" onclick="slotStop(' + localId + ')" style="display:none;">Stop</button>' +
    '<button onclick="removeSlot(' + localId + ')" style="font-size:0.85em;color:#999;">✕</button>' +
    '</div>' +
    '<div class="slot-result" style="margin-top:4px;">' +
    '<pre class="slot-resp" style="background:#f0f0f0;padding:6px;min-height:1.5em;white-space:pre-wrap;border-radius:4px;margin:4px 0 0 0;font-size:0.9em;"></pre>' +
    '<small class="slot-timing" style="color:#666;"></small>' +
    '</div>';
  container.appendChild(div);
}}

function removeSlot(localId) {{
  var div = document.getElementById("slot_" + localId);
  if (!div) return;
  var serverId = div.dataset.serverId;
  if (serverId) fetch("/remove_slot/" + serverId, {{method:"POST"}});
  if (_activePolls[localId]) clearTimeout(_activePolls[localId]);
  div.remove();
}}

function slotAsk(localId) {{
  var div = document.getElementById("slot_" + localId);
  if (!div) return;
  var q = div.querySelector(".slot-q").value;
  var tok = parseInt(div.querySelector(".slot-tok").value);
  var loop = div.querySelector(".slot-loop").checked;
  var askBtn = div.querySelector(".slot-ask");
  var stopBtn = div.querySelector(".slot-stop");
  var resp = div.querySelector(".slot-resp");
  var timing = div.querySelector(".slot-timing");

  askBtn.disabled = true; askBtn.textContent = "...";
  resp.textContent = "";
  timing.textContent = "";

  fetch("/ask", {{method:"POST", headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{question:q, max_tokens:tok, loop:loop, inject_context:document.getElementById("inject_ctx").checked}})
  }}).then(r=>r.json()).then(function(d) {{
    if (d.error) {{ resp.textContent = d.error; askBtn.disabled=false; askBtn.textContent="Ask"; return; }}
    div.dataset.serverId = d.slot_id;
    if (loop) {{ stopBtn.style.display = "inline"; }}
    _pollSlot(localId, d.slot_id);
  }});
}}

function slotStop(localId) {{
  var div = document.getElementById("slot_" + localId);
  if (!div) return;
  var serverId = div.dataset.serverId;
  if (serverId) fetch("/stop_slot/" + serverId, {{method:"POST"}});
  div.querySelector(".slot-stop").style.display = "none";
}}

function _pollSlot(localId, serverId) {{
  fetch("/result/" + serverId).then(r=>r.json()).then(function(d) {{
    var div = document.getElementById("slot_" + localId);
    if (!div) return;
    var resp = div.querySelector(".slot-resp");
    var timing = div.querySelector(".slot-timing");
    resp.textContent = d.response;
    var parts = [];
    if (d.duration_since_asked) parts.push(d.duration_since_asked);
    if (d.loop && d.running) parts.push("looping");
    timing.textContent = parts.join(" | ");
    if (!d.running) {{
      div.querySelector(".slot-ask").disabled = false;
      div.querySelector(".slot-ask").textContent = "Ask";
      div.querySelector(".slot-stop").style.display = "none";
      delete _activePolls[localId];
    }} else {{
      _activePolls[localId] = setTimeout(function(){{ _pollSlot(localId, serverId); }}, 200);
    }}
  }});
}}

// Add one slot by default
addSlot();
</script>
</body></html>"""


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = cam.get_frame()
            if frame is not None:
                if pt is not None and pt.enabled:
                    frame = pt.annotate_frame(frame)
                if ft is not None and ft.enabled:
                    frame = ft.annotate_frame(frame)
                ok, jpeg = cv2.imencode(".jpg", frame)
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpeg.tobytes() + b"\r\n")
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/switch_camera", methods=["POST"])
def switch_camera():
    data = request.get_json()
    cam.switch_device(data["device"])
    return json.dumps({"ok": True})


@app.route("/cameras")
def cameras():
    """Return available cameras with paths, V4L2 names, user labels, and active flag."""
    devices = sorted(glob.glob("/dev/video*")) or ["/dev/video0"]
    result = []
    for d in devices:
        result.append({
            "path": d,
            "v4l2_name": _v4l2_device_name(d),
            "label": _camera_label(d),
            "active": d == cam._device,
        })
    return json.dumps({"cameras": result})


@app.route("/set_resolution", methods=["POST"])
def set_resolution():
    data = request.get_json()
    w = data.get("width", 640)
    h = data.get("height", 480)
    cam.set_resolution(w, h)
    return json.dumps({"ok": True, "width": w, "height": h})


@app.route("/sam3_status")
def sam3_status():
    return json.dumps({"status": _sam3_status, "error": _sam3_error})


@app.route("/preload_sam3", methods=["POST"])
def preload_sam3():
    if _sam3_status in ("ready", "loading"):
        return json.dumps({"status": _sam3_status})
    threading.Thread(target=_load_sam3, daemon=True).start()
    return json.dumps({"status": "loading"})


@app.route("/model_status")
def model_status():
    """Return status of all models for the status bar."""
    # vLLM
    vllm_status = "off"
    if _vllm_client is not None:
        try:
            _vllm_client.models.list()
            vllm_status = "ready"
        except Exception:
            vllm_status = "error"

    # Model display name
    vllm_model_display = _vllm_model_id or "none"
    if "/" in vllm_model_display:
        vllm_model_display = vllm_model_display.split("/")[-1]

    # SAM3
    sam3 = {"status": _sam3_status, "error": _sam3_error}

    # InsightFace
    if ft is not None:
        face = {"status": ft.status, "error": ft.error, "enabled": ft.enabled}
    else:
        face = {"status": "unavailable", "error": "", "enabled": False}

    # Person tracker (SAM3 + ResNet50 ReID)
    if pt is not None:
        person = {"status": pt.status, "error": pt.error, "enabled": pt.enabled}
    else:
        person = {"status": "unavailable", "error": "", "enabled": False}

    return json.dumps({
        "vllm": vllm_status,
        "vllm_model": vllm_model_display,
        "sam3": sam3,
        "face": face,
        "person": person,
    })


@app.route("/memory_status")
def memory_status_route():
    return json.dumps(_memory_status())


@app.route("/face_data")
def face_data():
    if ft is None:
        return json.dumps({"status": "unavailable", "enabled": False, "faces": [], "roster": []})
    return json.dumps({
        "status": ft.status,
        "enabled": ft.enabled,
        "error": ft.error,
        "faces": ft.get_faces(),
        "roster": ft.identity_roster,
    })


@app.route("/toggle_face_tracking", methods=["POST"])
def toggle_face_tracking():
    if ft is None:
        return json.dumps({"error": "insightface not available"})
    data = request.get_json()
    if data.get("enabled"):
        ft.enable()
    else:
        ft.disable()
    return json.dumps({"ok": True, "status": ft.status, "enabled": ft.enabled})


@app.route("/toggle_person_tracking", methods=["POST"])
def toggle_person_tracking():
    if pt is None:
        return json.dumps({"error": "person tracker not available"})
    data = request.get_json()
    if data.get("enabled"):
        if _sam3_status not in ("ready", "loading"):
            threading.Thread(target=_load_sam3, daemon=True).start()
        pt.enable()
    else:
        pt.disable()
    return json.dumps({"ok": True, "status": pt.status, "enabled": pt.enabled})


@app.route("/set_sam3_prompts", methods=["POST"])
def set_sam3_prompts():
    data = request.get_json()
    prompts = data.get("prompts", "")
    if pt is not None:
        pt.set_sam3_prompts(prompts)
    return json.dumps({"ok": True, "prompts": prompts})


@app.route("/person_data")
def person_data():
    if pt is None:
        return json.dumps({"status": "unavailable", "enabled": False, "faces": [], "roster": []})
    return json.dumps({
        "status": pt.status,
        "enabled": pt.enabled,
        "error": pt.error,
        "faces": pt.get_faces(),
        "roster": pt.identity_roster,
    })


def _normalize_question_spec(raw, *, slot_type: str, index: int) -> dict:
    spec = VLMTaskSpec.from_payload(raw, default_id=f"{slot_type}_{index}")
    payload = spec.to_payload()
    payload["slot_type"] = slot_type
    return payload


def _current_person_targets() -> list[dict]:
    if pt is not None and pt.enabled:
        try:
            return list(pt.get_faces())
        except Exception:
            return []
    return []


def _crop_frame_to_bbox(frame_rgb: np.ndarray, bbox: tuple[int, int, int, int], margin_ratio: float = 0.15) -> np.ndarray:
    x, y, w, h = bbox
    height, width = frame_rgb.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    cropped = frame_rgb[y1:y2, x1:x2]
    return cropped if cropped.size else frame_rgb


def _prepare_vlm_frame(frame_rgb: np.ndarray, spec: dict) -> tuple[np.ndarray | None, dict]:
    target = str(spec.get("target", "scene") or "scene").strip().lower()
    if target != "nearest_person":
        return frame_rgb, {"target": "scene", "target_bbox": None, "target_identity": ""}

    people = _current_person_targets()
    if not people:
        return None, {"target": "nearest_person", "target_bbox": None, "target_identity": ""}

    person = people[0]
    bbox = person.get("bbox")
    bbox_tuple = tuple(bbox) if bbox else None
    if bbox_tuple and len(bbox_tuple) == 4:
        crop = _crop_frame_to_bbox(frame_rgb, bbox_tuple)
    else:
        crop = frame_rgb
    return crop, {
        "target": "nearest_person",
        "target_bbox": list(bbox_tuple) if bbox_tuple else None,
        "target_identity": str(person.get("identity", "")).strip(),
    }


def _build_question(raw_question: str, inject_context: bool) -> str:
    """Optionally prepend tracker context to question."""
    if not inject_context:
        return raw_question
    ctx_parts = []
    if ft is not None and ft.enabled:
        c = ft.build_context_string()
        if c:
            ctx_parts.append(c)
    if pt is not None and pt.enabled:
        c = pt.build_context_string()
        if c:
            ctx_parts.append(c)
    if ctx_parts:
        return " ".join(ctx_parts) + " " + raw_question
    return raw_question


def _slot_worker(slot_id: str):
    """Worker thread for a question slot. Runs once or loops."""
    slot = _slots.get(slot_id)
    if not slot:
        return
    while True:
        if slot["stop"]:
            break
        frame_rgb = cam.get_frame_rgb()
        if frame_rgb is None:
            slot["response"] = "(no frame)"
            slot["timing"] = ""
            slot["done"] = True
            break
        question = _build_question(slot["question"], slot["inject_context"])
        slot["asked_at"] = time.time()
        slot["done"] = False
        try:
            answer, elapsed = run_vlm_vllm(frame_rgb, question, slot["max_tokens"])
            slot["response"] = answer
            slot["timing"] = f"{elapsed:.2f}s"
            slot["duration_since_asked"] = f"{elapsed:.2f}s"
        except Exception as e:
            slot["response"] = f"(error: {e})"
            slot["timing"] = ""
        slot["done"] = True
        if not slot["loop"] or slot["stop"]:
            break
    slot["running"] = False


@app.route("/ask", methods=["POST"])
def ask():
    global _next_slot_id
    if _vllm_client is None:
        return json.dumps({"error": "VLM not configured (no --vllm-url)"})

    data = request.get_json()
    slot_id = data.get("slot_id")
    loop = data.get("loop", False)

    # If slot_id provided and already running a loop, just update it
    if slot_id and slot_id in _slots and _slots[slot_id]["running"]:
        return json.dumps({"error": "Slot already running", "slot_id": slot_id})

    # Create new slot if no id provided
    if not slot_id:
        with _slot_lock:
            slot_id = str(_next_slot_id)
            _next_slot_id += 1

    _slots[slot_id] = {
        "question": data.get("question", "What's going on?"),
        "max_tokens": data.get("max_tokens", 128),
        "inject_context": data.get("inject_context", True),
        "loop": loop,
        "stop": False,
        "running": True,
        "response": "",
        "timing": "",
        "done": False,
        "asked_at": time.time(),
        "duration_since_asked": "",
    }

    threading.Thread(target=_slot_worker, args=(slot_id,), daemon=True).start()
    return json.dumps({"ok": True, "slot_id": slot_id})


@app.route("/result/<slot_id>")
def result(slot_id):
    slot = _slots.get(slot_id)
    if not slot:
        return json.dumps({"error": "Unknown slot", "done": True})
    return json.dumps({
        "response": slot["response"],
        "timing": slot["timing"],
        "done": slot["done"],
        "loop": slot["loop"],
        "running": slot["running"],
        "duration_since_asked": slot.get("duration_since_asked", ""),
    })


@app.route("/stop_slot/<slot_id>", methods=["POST"])
def stop_slot(slot_id):
    slot = _slots.get(slot_id)
    if slot:
        slot["stop"] = True
        slot["loop"] = False
    return json.dumps({"ok": True})


@app.route("/remove_slot/<slot_id>", methods=["POST"])
def remove_slot(slot_id):
    slot = _slots.get(slot_id)
    if slot:
        slot["stop"] = True
        slot["loop"] = False
    _slots.pop(slot_id, None)
    return json.dumps({"ok": True})


# ---------------------------------------------------------------------------
# Managed question worker (round-robin constant + ephemeral)
# ---------------------------------------------------------------------------
def _managed_question_worker():
    """Background thread: cycles through constant + ephemeral questions."""
    global _managed_worker_running, _question_answer_seq
    while _managed_worker_running:
        with _questions_lock:
            questions = (
                [dict(q) for q in _constant_questions]
                + [dict(q) for q in _ephemeral_questions]
            )
        if not questions or _vllm_client is None:
            time.sleep(0.5)
            continue
        for spec in questions:
            if not _managed_worker_running:
                break
            frame_rgb = cam.get_frame_rgb()
            if frame_rgb is None:
                time.sleep(0.1)
                continue
            task_id = str(spec.get("task_id", "")).strip() or f"vlm_task_{len(questions)}"
            cadence_s = float(spec.get("cadence_s", 0.0) or 0.0)
            with _questions_lock:
                last_answer = dict(_question_answers.get(task_id, {}) or {})
            last_timestamp = float(last_answer.get("timestamp", 0.0) or 0.0)
            if cadence_s > 0.0 and (time.time() - last_timestamp) < cadence_s:
                continue
            prepared_frame, target_meta = _prepare_vlm_frame(frame_rgb, spec)
            if prepared_frame is None:
                with _questions_lock:
                    _question_answer_seq += 1
                    _question_answers[task_id] = {
                        "task_id": task_id,
                        "label": spec.get("label") or task_id,
                        "question": spec.get("question", ""),
                        "answer": "(no target person in view)",
                        "elapsed": 0.0,
                        "slot_type": spec.get("slot_type", "constant"),
                        "timestamp": time.time(),
                        "answer_id": f"vlm-{_question_answer_seq}",
                        "target": spec.get("target", "scene"),
                        "cadence_s": cadence_s,
                        "interpret_as": spec.get("interpret_as", "general"),
                        "target_bbox": target_meta.get("target_bbox"),
                        "target_identity": target_meta.get("target_identity", ""),
                    }
                continue
            question = str(spec.get("question", "")).strip()
            if target_meta.get("target") == "nearest_person":
                question = "Answer about the visible person in this crop only. " + question
            full_q = _build_question(question, inject_context=True)
            try:
                answer, elapsed = run_vlm_vllm(prepared_frame, full_q, 128)
                with _questions_lock:
                    _question_answer_seq += 1
                    _question_answers[task_id] = {
                        "task_id": task_id,
                        "label": spec.get("label") or task_id,
                        "question": spec.get("question", ""),
                        "answer": answer,
                        "elapsed": round(elapsed, 3),
                        "slot_type": spec.get("slot_type", "constant"),
                        "timestamp": time.time(),
                        "answer_id": f"vlm-{_question_answer_seq}",
                        "target": spec.get("target", "scene"),
                        "cadence_s": cadence_s,
                        "interpret_as": spec.get("interpret_as", "general"),
                        "target_bbox": target_meta.get("target_bbox"),
                        "target_identity": target_meta.get("target_identity", ""),
                    }
            except Exception as e:
                with _questions_lock:
                    _question_answer_seq += 1
                    _question_answers[task_id] = {
                        "task_id": task_id,
                        "label": spec.get("label") or task_id,
                        "question": spec.get("question", ""),
                        "answer": f"(error: {e})",
                        "elapsed": 0,
                        "slot_type": spec.get("slot_type", "constant"),
                        "timestamp": time.time(),
                        "answer_id": f"vlm-{_question_answer_seq}",
                        "target": spec.get("target", "scene"),
                        "cadence_s": cadence_s,
                        "interpret_as": spec.get("interpret_as", "general"),
                        "target_bbox": target_meta.get("target_bbox"),
                        "target_identity": target_meta.get("target_identity", ""),
                    }


def _start_managed_worker():
    global _managed_worker_running, _managed_worker_thread
    if _managed_worker_running:
        return
    _managed_worker_running = True
    _managed_worker_thread = threading.Thread(target=_managed_question_worker, daemon=True)
    _managed_worker_thread.start()


def _get_sam_prompts() -> list[str]:
    """Return combined SAM3 prompts from managed targets + manual input."""
    with _sam_targets_lock:
        prompts = list(_constant_sam_targets) + list(_ephemeral_sam_targets)
    # Deduplicate preserving order
    seen = set()
    result = []
    for p in prompts:
        normalized = _normalize_sam_prompt_item(p)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_sam_prompt_item(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("target", "label", "name", "text", "value"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return ""


# ---------------------------------------------------------------------------
# Structured API endpoints (for character-eng polling)
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return json.dumps({"status": "ok", "timestamp": time.time()})


@app.route("/inject_frame", methods=["POST"])
def inject_frame():
    """Inject a JPEG frame from browser camera (--no-camera mode)."""
    jpeg_bytes = request.data
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        cam.inject(frame)
    return json.dumps({"ok": True})


@app.route("/set_input_mode", methods=["POST"])
def set_input_mode():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode") or "camera").strip().lower()
    if mode not in {"camera", "external"}:
        return Response(json.dumps({"ok": False, "error": f"unsupported mode: {mode}"}), status=400, mimetype="application/json")
    cam.set_input_mode(mode)
    return json.dumps({"ok": True, "mode": cam.input_mode()})


@app.route("/input_mode")
def input_mode():
    return json.dumps({"mode": cam.input_mode()})


@app.route("/frame.jpg")
def frame_jpeg():
    """Return the latest frame as a single JPEG image."""
    frame = cam.get_frame()
    if frame is None:
        return Response("no frame available", status=503, mimetype="text/plain")
    annotated = request.args.get("annotated", "0") == "1"
    max_width = int(request.args.get("max_width", "0") or "0")
    if annotated:
        if pt is not None and pt.enabled:
            frame = pt.annotate_frame(frame)
        if ft is not None and ft.enabled:
            frame = ft.annotate_frame(frame)
    if max_width > 0 and frame.shape[1] > max_width:
        scale = max_width / float(frame.shape[1])
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        return Response("failed to encode frame", status=500, mimetype="text/plain")
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/snapshot")
def snapshot():
    """Return current visual state as structured JSON."""
    global _snapshot_seq
    faces = []
    face_timing = {}
    if ft is not None and ft.enabled:
        with ft._lock:
            faces = list(ft._faces)
            face_timing = dict(ft._timing)

    persons = []
    person_timing = {}
    if pt is not None and pt.enabled:
        with pt._lock:
            persons = list(pt._faces)
            person_timing = dict(pt._timing)

    # Extra SAM3 detections (non-person)
    objects = []
    if pt is not None and pt.enabled:
        with pt._lock:
            for det in pt._extra_dets:
                objects.append({
                    "label": det.get("prompt", "unknown"),
                    "bbox": det.get("bbox"),
                    "confidence": det.get("confidence", 0),
                })

    vlm_answers = []
    with _questions_lock:
        for data in _question_answers.values():
            vlm_answers.append({
                "task_id": data.get("task_id", ""),
                "label": data.get("label", ""),
                "question": data.get("question", ""),
                "answer": data["answer"],
                "elapsed": data["elapsed"],
                "slot_type": data["slot_type"],
                "timestamp": data["timestamp"],
                "answer_id": data.get("answer_id", ""),
                "target": data.get("target", "scene"),
                "cadence_s": data.get("cadence_s", 0.0),
                "interpret_as": data.get("interpret_as", "general"),
                "target_bbox": data.get("target_bbox"),
                "target_identity": data.get("target_identity", ""),
            })

    _snapshot_seq += 1
    cycle_id = f"vision-cycle-{_snapshot_seq}"
    return json.dumps({
        "faces": faces,
        "persons": persons,
        "objects": objects,
        "vlm_answers": vlm_answers,
        "cycle_id": cycle_id,
        "timestamp": time.time(),
        "trace": {
            "cycle_id": cycle_id,
            "face_tracking": {
                "timing": face_timing,
                "faces": faces,
            },
            "sam3_detection": {
                "timing": {
                    "sam3": person_timing.get("sam3", 0),
                    "total": person_timing.get("total", 0),
                    "fps": person_timing.get("fps", 0),
                    "prompts": person_timing.get("prompts", ""),
                    "tracking": person_timing.get("tracking", ""),
                },
                "persons": persons,
                "objects": objects,
            },
            "reid_tracking": {
                "timing": {
                    "reid": person_timing.get("reid", 0),
                    "total": person_timing.get("total", 0),
                    "fps": person_timing.get("fps", 0),
                    "tracking": person_timing.get("tracking", ""),
                },
                "persons": persons,
            },
            "vlm_answers": vlm_answers,
        },
    })


@app.route("/set_questions", methods=["POST"])
def set_questions():
    """Set constant and ephemeral VLM questions."""
    data = request.get_json()
    with _questions_lock:
        global _constant_questions, _ephemeral_questions
        _constant_questions = [
            _normalize_question_spec(item, slot_type="constant", index=index)
            for index, item in enumerate(data.get("constant", []), start=1)
        ]
        _ephemeral_questions = [
            _normalize_question_spec(item, slot_type="ephemeral", index=index)
            for index, item in enumerate(data.get("ephemeral", []), start=1)
        ]
        # Clear stale answers for removed questions
        active = {
            str(item.get("task_id", "")).strip()
            for item in (_constant_questions + _ephemeral_questions)
            if str(item.get("task_id", "")).strip()
        }
        stale = [k for k in _question_answers if k not in active]
        for k in stale:
            del _question_answers[k]
    _start_managed_worker()
    return json.dumps({"ok": True, "constant": _constant_questions, "ephemeral": _ephemeral_questions})


@app.route("/set_sam_targets", methods=["POST"])
def set_sam_targets():
    """Set constant and ephemeral SAM3 targets."""
    data = request.get_json()
    with _sam_targets_lock:
        global _constant_sam_targets, _ephemeral_sam_targets
        _constant_sam_targets = [
            prompt
            for prompt in (_normalize_sam_prompt_item(item) for item in data.get("constant", ["person"]))
            if prompt
        ] or ["person"]
        _ephemeral_sam_targets = [
            prompt
            for prompt in (_normalize_sam_prompt_item(item) for item in data.get("ephemeral", []))
            if prompt
        ]
    # Update person tracker's extra prompts
    if pt is not None:
        combined = _get_sam_prompts()
        # "person" is always first in person_tracker, so strip it and pass the rest
        extras = [p for p in combined if p.lower() != "person"]
        pt.set_sam3_prompts(", ".join(extras))
    return json.dumps({"ok": True, "constant": _constant_sam_targets, "ephemeral": _ephemeral_sam_targets})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging

    parser = argparse.ArgumentParser(description="VLM Webcam QA")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1",
        help="vLLM server base URL (empty string disables VLM)",
    )
    parser.add_argument(
        "--vllm-model", default=None,
        help="Model name for vLLM (default: auto-detect from server's /v1/models)",
    )
    parser.add_argument(
        "--vllm-image-max-side", type=int, default=384,
        help="Max image side for VLM input (default: 384)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device: cuda, cpu (default: cuda)",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="Torch dtype: bfloat16, float16, float32 (default: bfloat16)",
    )
    parser.add_argument(
        "--auto-start-trackers", action="store_true",
        help="Auto-start SAM3 + all trackers at boot (desktop mode)",
    )
    parser.add_argument(
        "--camera-names", default=None,
        help='JSON map of camera names, e.g. \'{"0": "Front", "2": "Arm"}\'',
    )
    parser.add_argument(
        "--capture-resolution", default=None,
        help="Camera capture resolution, e.g. 1280x720",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Skip camera capture (frames injected via /inject_frame)",
    )
    args = parser.parse_args()

    # Device / dtype
    DEVICE = args.device
    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    DTYPE = _dtype_map.get(args.dtype, torch.bfloat16)
    print(f"Device: {DEVICE}, dtype: {DTYPE}", flush=True)

    # VLM image max side
    _vllm_image_max_side = args.vllm_image_max_side

    # Camera names
    if args.camera_names:
        _camera_names = json.loads(args.camera_names)

    # Capture resolution
    if args.capture_resolution:
        parts = args.capture_resolution.split("x")
        _capture_resolution = (int(parts[0]), int(parts[1]))

    # vLLM client setup
    if args.vllm_url:
        from openai import OpenAI
        _vllm_client = OpenAI(base_url=args.vllm_url, api_key="dummy")
        # Auto-detect model name if not specified
        if args.vllm_model:
            _vllm_model_id = args.vllm_model
        else:
            try:
                models = _vllm_client.models.list()
                if models.data:
                    _vllm_model_id = models.data[0].id
                    print(f"Auto-detected vLLM model: {_vllm_model_id}", flush=True)
                else:
                    _vllm_model_id = "unknown"
                    print("Warning: vLLM returned no models", flush=True)
            except Exception as e:
                _vllm_model_id = args.vllm_model or "unknown"
                print(f"Warning: Could not auto-detect vLLM model: {e}", flush=True)
        print(f"Using vLLM at {args.vllm_url} with model {_vllm_model_id}", flush=True)
    else:
        print("VLM disabled (no --vllm-url). Trackers still work standalone.", flush=True)

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Start camera (skip if --no-camera, frames come via /inject_frame)
    if not args.no_camera:
        if _capture_resolution:
            cam._resolution = _capture_resolution
        cam.start()
    else:
        print("Camera disabled (--no-camera). Frames via /inject_frame.", flush=True)

    # Create tracker instances (but don't enable by default)
    try:
        from face_tracker import FaceTracker
        ft = FaceTracker(cam, device=DEVICE)
        print("Face tracker: available", flush=True)
    except ImportError:
        print("insightface not installed — face tracking unavailable.", flush=True)

    try:
        from person_tracker import PersonTracker
        pt = PersonTracker(
            cam, sam3_getter=lambda: (_sam3_model, _sam3_proc),
            face_getter=lambda: ft.get_faces() if ft is not None else [],
            device=DEVICE, dtype=DTYPE,
        )
        print("Person tracker: available", flush=True)
    except ImportError as e:
        print(f"Person tracking unavailable: {e}", flush=True)

    # Auto-start mode (backward compat for desktop)
    if args.auto_start_trackers:
        print("Auto-starting trackers...", flush=True)
        threading.Thread(target=_load_sam3, daemon=True).start()
        if ft is not None:
            ft.enable()
            print("Face tracking: starting...", flush=True)
        if pt is not None:
            pt.enable()
            print("Person tracking: starting...", flush=True)

    print(f"Dashboard at http://localhost:{args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)
