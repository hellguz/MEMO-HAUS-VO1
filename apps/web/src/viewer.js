import "./viewer.css";
import * as THREE from "three";
import { SparkRenderer, SplatMesh, SplatFileType } from "@sparkjsdev/spark";

const viewerEl      = document.getElementById("viewer");
const placeholderEl   = document.getElementById("placeholder");
const splashStatusEl  = document.getElementById("splash-status");
const splashCountEl   = document.getElementById("splash-count");
const overlayEl     = document.getElementById("transition-overlay");
const hudEl         = document.getElementById("scene-hud");
const captionNameEl = document.getElementById("caption-name");
const captionAuthEl = document.getElementById("caption-author");
const captionYearEl = document.getElementById("caption-year");
const captionStoryEl = document.getElementById("caption-story");
const captionClusterEl = document.getElementById("caption-cluster");
const authorRowEl   = document.getElementById("author-row");
const yearRowEl     = document.getElementById("year-row");
const clusterRowEl  = document.getElementById("cluster-row");
const storyRowEl    = document.getElementById("story-row");
const connectionEl  = document.getElementById("connection");
const loaderEl           = document.getElementById("processing-loader");
const timerArcEl         = document.getElementById("timer-arc");
const storyOverlayEl     = document.getElementById("story-overlay");
const overlayStoryTextEl = document.getElementById("overlay-story-text");
// sceneAudioEl kept in DOM but audio is now routed through Web Audio for spatialization
const sceneAudioEl       = document.getElementById("scene-audio");
const cursorReticleEl    = document.getElementById("cursor-reticle");
const boxesBtn       = document.getElementById("btn-boxes");
const musicBtn       = document.getElementById("btn-music");
const fullscreenBtn  = document.getElementById("btn-fullscreen");
const renderModeBtn  = document.getElementById("btn-render-mode");

// ── Debug helpers ─────────────────────────────────────────────────────────
// ?limit=N in the URL loads only the first N memories into the world instead
// of all of them, and flies the camera straight to the loaded scene instead
// of parking at the overview — useful for isolating whether a viewer problem
// is caused by scene count/size (each PLY here is ~66MB) vs. something else.
const _urlParams  = new URLSearchParams(location.search);
const DEBUG_LIMIT = _urlParams.has("limit") ? parseInt(_urlParams.get("limit"), 10) : null;
console.log("[viewer] boot", { url: location.href, debugLimit: DEBUG_LIMIT });

window.addEventListener("error", e => {
  console.error("[viewer] window error:", e.message, e.error || e);
});
window.addEventListener("unhandledrejection", e => {
  console.error("[viewer] unhandled promise rejection:", e.reason);
});

// ── Spatial audio ─────────────────────────────────────────────────────────
// Uses the Web Audio API PannerNode so volume fades with distance and the
// stereo pan tracks which direction each scene is relative to where the
// camera is pointing — moving toward a memory makes it louder and centered,
// turning your head left/right shifts it in the headphones / speakers.

let _audioCtx = null;
const _audioBufferCache = new Map(); // url → Promise<AudioBuffer>
// sceneId → { source, panner, gainNode, worldPos }
const _activeSources = new Map();

// Music on/off button (bottom-right) — on by default. Muting suspends the
// shared AudioContext rather than stopping sources, so un-muting resumes
// exactly where it left off instead of restarting every track.
let audioMuted = false;

function setMusicEnabled(enabled) {
  audioMuted = !enabled;
  if (!_audioCtx) return;
  if (audioMuted) _audioCtx.suspend().catch(() => {});
  else _audioCtx.resume().catch(() => {});
}

function getAudioCtx() {
  if (!_audioCtx) {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioMuted) _audioCtx.suspend().catch(() => {});
  }
  return _audioCtx;
}

// Resume the context on any user gesture so audio can start on mobile.
document.addEventListener("pointerdown", () => {
  if (!audioMuted && _audioCtx && _audioCtx.state === "suspended") _audioCtx.resume().catch(() => {});
}, { passive: true });

function _fetchAudioBuffer(url) {
  if (_audioBufferCache.has(url)) return _audioBufferCache.get(url);
  const p = fetch(url)
    .then(r => r.arrayBuffer())
    .then(ab => getAudioCtx().decodeAudioData(ab))
    .catch(err => { console.error("Audio decode error:", err); return null; });
  _audioBufferCache.set(url, p);
  return p;
}

function stopSceneAudio(sceneId) {
  const entry = _activeSources.get(sceneId);
  if (!entry) return;
  try { entry.source.stop(); } catch {}
  try { entry.gainNode.disconnect(); } catch {}
  try { entry.stereoPanner.disconnect(); } catch {}
  _activeSources.delete(sceneId);
}

function stopAllAudio() {
  for (const id of [..._activeSources.keys()]) stopSceneAudio(id);
  sceneAudioEl.pause();
}

async function playSceneAudio(scene, worldPos) {
  if (!scene.audio_url) return;
  stopSceneAudio(scene.id);

  const ctx = getAudioCtx();
  // Never un-suspend a context the user muted — the source below still gets
  // created/started, but a suspended context produces no sound at all, which
  // is exactly what "music off" needs to mean regardless of what else happens.
  if (!audioMuted && ctx.state === "suspended") await ctx.resume().catch(() => {});

  const buf = await _fetchAudioBuffer(scene.audio_url);
  if (!buf) return;

  // GainNode — distance-based volume, driven per-frame by updateSpatialAudio().
  const gainNode = ctx.createGain();
  gainNode.gain.value = 1.0;

  // StereoPannerNode — L/R panning driven per-frame by projecting the source
  // direction onto the camera's right vector, so it tracks gyro and look turns.
  const stereoPanner = ctx.createStereoPanner();
  stereoPanner.pan.value = 0;

  const source = ctx.createBufferSource();
  source.buffer = buf;
  source.loop   = true;
  source.connect(gainNode);
  gainNode.connect(stereoPanner);
  stereoPanner.connect(ctx.destination);
  source.start();

  _activeSources.set(scene.id, { source, gainNode, stereoPanner, worldPos: [...worldPos] });
}

// ── Per-frame spatial audio update ────────────────────────────────────────
// Updates listener orientation for L/R pan AND manually drives each source's
// gain based on distance — this is what makes audio louder when you approach
// and quieter as you walk away, for every audio type.

const AUDIO_REF_DIST = 4;   // within this distance: full volume
const AUDIO_MAX_DIST = 30;  // beyond this: essentially silent
const AUDIO_SMOOTH   = 0.18; // gentle ramp so cross-fades feel seamless

function updateSpatialAudio() {
  const cam = viewer?.camera;
  if (!_audioCtx || !cam) return;
  if (audioMuted) return; // leave it suspended — don't fight the mute button
  if (_audioCtx.state === "suspended") { _audioCtx.resume().catch(() => {}); return; }

  const { x, y, z } = cam.position;
  const m = cam.matrixWorld.elements;
  // Camera right vector (column 0 of world matrix) — the key for L/R panning.
  // dot(dir_to_source, rightVec) > 0 means source is to the right → pan right.
  const rightX = m[0], rightY = m[1], rightZ = m[2];

  const now = _audioCtx.currentTime;
  for (const { gainNode, stereoPanner, worldPos } of _activeSources.values()) {
    const dx = worldPos[0] - x;
    const dy = worldPos[1] - y;
    const dz = worldPos[2] - z;
    const dist = Math.hypot(dx, dy, dz) || 0.001;

    // Distance gain: inverse-power curve, full within REF_DIST, ~0 at MAX_DIST
    const clamped = Math.max(AUDIO_REF_DIST, Math.min(dist, AUDIO_MAX_DIST));
    gainNode.gain.setTargetAtTime(Math.pow(AUDIO_REF_DIST / clamped, 1.6), now, AUDIO_SMOOTH);

    // L/R stereo pan: project source direction onto camera right axis.
    // Responds directly to gyro yaw and look-joystick turns from mobile.
    const pan = Math.max(-1, Math.min(1,
      (dx / dist) * rightX + (dy / dist) * rightY + (dz / dist) * rightZ
    ));
    stereoPanner.pan.setTargetAtTime(pan, now, AUDIO_SMOOTH);
  }
}


const DWELL_MS        = 60_000;
const WORLD_DWELL_MS  = 60_000; // time on each memory before auto-advancing
const POLL_INTERVAL   = 4_000;
const STATUS_INTERVAL = 3_000;
const OVERLAY_FADE_MS = 480;
const ERROR_SHOW_MS   = 4_000;
const TIMER_C         = 100.53; // 2π × 16

const INIT_POS    = [0, 0, -3];   // camera spawn point
// Direction the camera faces at _yaw=0/_pitch=0 (applyFPSCamera()'s default)
// is +Z, so from INIT_POS this looks toward the origin — kept here as a note,
// not passed anywhere, since there's no separate "look-at" option to set.
const INIT_TARGET = [0, 0, 1];

let viewer        = null;
let scenes        = [];
let currentIndex  = -1;
let activeSceneId = null;
let started       = false;
let errorTimer    = null;
let resetRaf      = null;
let storyOverlayTimer = null;

// ── World of Memories ─────────────────────────────────────────────────────

let worldMode = false;
let worldScenes    = [];  // scenes currently in Memory Verse (may be a subset)
let worldLoadedOrder = []; // scene ids ordered by viewer scene index
let worldFocusedId = null; // which scene's audio is currently active
const serverPositions = new Map(); // scene_id → {x_pct, y_pct} from /api/scene-positions

const WORLD_SPACING = 55;  // fallback cell size when no extents are known yet
const SCENE_GAP     = 18;  // clear gap between scene bounding boxes, in world units
const worldPositions = new Map(); // scene id → [x, y, z]
// scene id → { xSpan, ySpan, zSpan } — measured from the proxy points
const sceneExtents   = new Map();
// scene id → [x, y, z], the MEAN of every proxy point in the scene's own
// local space (i.e. an offset from worldPositions, not a world position by
// itself). Used wherever "how far away is this scene" needs to mean the
// actual mass of splats, not just the map slot it was placed at — see
// sceneWorldCenter() below. A bounding-box centre ((min+max)/2) would get
// dragged badly off by a handful of stray far-flung gaussians; the mean is
// only pulled proportionally to how many points are actually out there.
const sceneCentroids = new Map();

// ── Ground-plane placement, shared with the main-page map ─────────────────
// A memory's spot lives on the server as (x_pct, y_pct) in a unit square — the
// SAME numbers the main-page map edits. The viewer maps that square onto the
// horizontal GROUND plane (Y = 0, the floor you look across), so the map and
// the 3D world are always the same layout. GROUND_SPAN is the world size of
// that square; it must be shared with the main page (see index.html map).
const GROUND_SPAN = 640;   // world units across the full unit-square ground map
const GROUND_Y    = 0;     // everything sits on the floor plane

function pctToWorld(p) {
  return [(p.x_pct - 0.5) * GROUND_SPAN, GROUND_Y, (p.y_pct - 0.5) * GROUND_SPAN];
}

async function fetchServerPositions() {
  try {
    const r = await fetch("/api/scene-positions", { cache: "no-store" });
    if (r.ok) {
      const data = await r.json();
      Object.entries(data).forEach(([id, pos]) => serverPositions.set(id, pos));
    }
  } catch {}
}

// Lattice step for the fallback layout: the largest scene footprint + a gap.
function worldCell() {
  let maxExt = 0;
  for (const e of sceneExtents.values()) {
    maxExt = Math.max(maxExt, e.xSpan || 0, e.zSpan || 0);
  }
  if (!isFinite(maxExt) || maxExt <= 0) maxExt = WORLD_SPACING;
  return maxExt + SCENE_GAP;
}

// Fallback only, for a scene the server has no position for yet: claim the
// innermost free cell on the GROUND plane (2D shells around the origin), so it
// still doesn't overlap anything already placed.
function placeSceneNoOverlap(scene) {
  if (worldPositions.has(scene.id)) return worldPositions.get(scene.id);
  const cell = worldCell();
  const occupied = [...worldPositions.values()];
  const isFree = p => occupied.every(o =>
    Math.hypot(p[0] - o[0], p[2] - o[2]) >= cell * 0.999);

  for (let r = 0; r < 64; r++) {
    for (let ix = -r; ix <= r; ix++)
      for (let iz = -r; iz <= r; iz++) {
        if (Math.max(Math.abs(ix), Math.abs(iz)) !== r) continue;
        const p = [ix * cell, GROUND_Y, iz * cell];
        if (isFree(p)) { worldPositions.set(scene.id, p); return p; }
      }
  }
  const p = [(occupied.length + 1) * cell, GROUND_Y, 0];
  worldPositions.set(scene.id, p);
  return p;
}

// Resolve one scene's world spot: server position if it has one, else a
// non-overlapping ground cell.
function resolveWorldPosition(scene) {
  const sp = serverPositions.get(scene.id);
  if (sp) {
    const p = pctToWorld(sp);
    worldPositions.set(scene.id, p);
    return p;
  }
  return placeSceneNoOverlap(scene);
}

// Lay out a whole set from the authoritative server positions (server-placed
// first, so any fallbacks fill the gaps between them without overlap).
function relayoutWorldEven(orderedScenes) {
  for (const s of orderedScenes) worldPositions.delete(s.id);
  const placed = orderedScenes.filter(s => serverPositions.has(s.id));
  const rest   = orderedScenes.filter(s => !serverPositions.has(s.id));
  for (const s of placed) resolveWorldPosition(s);
  for (const s of rest)   placeSceneNoOverlap(s);
}

// A scene arriving later (poll) — use its server spot, or drop it into the
// first free cell without disturbing anything already placed.
function assignWorldPosition(scene) {
  return resolveWorldPosition(scene);
}

// Live-move a memory to a new world position: slide its proxy and audio right
// away, and if its full splat is loaded, drop it so it reloads at the new spot
// when next approached, same as everywhere else a splat gets positioned.
function moveSceneTo(id, p) {
  worldPositions.set(id, p);
  const proxy = pointCloudProxies.get(id);
  if (proxy) placeSceneObject(proxy, id, p);
  const src = _activeSources.get(id);
  if (src) src.worldPos = [p[0], p[1], p[2]];
  // If its full splat is loaded, drop it — updateFullResLOD reloads it at the
  // new spot next tick if it's still among the closest.
  if (loadedSplatIds.includes(id)) {
    enqueueSplatOp(`move-reload ${id}`, () => splatUnload(id));
  }
  refreshBoundaryBoxes();
}

// Poll the server for map edits and apply any moved memory in real time — this
// is what makes dragging a memory on the main-page map slide it in the viewer.
async function syncServerPositions() {
  if (!worldMode) return;
  try {
    const r = await fetch("/api/scene-positions", { cache: "no-store" });
    if (!r.ok) return;
    const data = await r.json();
    for (const [id, sp] of Object.entries(data)) {
      const prev = serverPositions.get(id);
      if (prev && prev.x_pct === sp.x_pct && prev.y_pct === sp.y_pct) continue;
      serverPositions.set(id, sp);
      if (worldPositions.has(id)) moveSceneTo(id, pctToWorld(sp));
    }
  } catch {}
}

// Runs fn(item) over items with at most `limit` in flight at once — plain
// Promise.all over dozens of scenes means dozens of concurrent full PLY
// downloads (each ~66MB just to build a proxy), which is its own way to
// choke a weak connection/laptop even though nothing renders expensively.
async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

// ── Distant-view point-cloud proxies ─────────────────────────────────────
// Cheap stand-ins for every scene that isn't currently being visited: just
// the splat centers + colours as plain THREE.Points (a few thousand points,
// heavily decimated), rendered straight into the viewer's own THREE scene —
// entirely outside the gaussian-splat pipeline, so no covariance textures,
// no per-splat sorting, no LOD tree. That pipeline is what a laptop actually
// chokes on with 10 full splats loaded at once (~23M gaussians); a scene's
// full quality only ever loads for whichever ONE scene the visitor is
// currently near, everything else stays a lightweight point cloud.
const POINT_CLOUD_MAX_POINTS = 12_000;
// Base world-space point size at "1 unit from camera" — see uPixelScale below,
// which turns this into an actual on-screen pixel size every frame.
const POINT_SIZE = 5;
// Distance (world units) from the camera within which a point is fully
// visible/at max size, and beyond which it fades all the way to invisible.
// Tunable — these are a first pass; adjust to taste once you can see it live.
const POINT_FADE_NEAR = 1;
const POINT_FADE_FAR  = 140;
// On-screen size clamp in pixels: MAX makes close-up points read as "quite
// large" instead of a fine mist; MIN keeps far-but-still-visible points from
// shimmering at sub-pixel size. Keep MAX modest — each point is ONE flat,
// unblended splat colour, so blowing it up too far turns a soft-looking
// cloud into a mosaic of harsh, oversized, single-colour tiles.
const POINT_MAX_PIXEL_SIZE = 80;
const POINT_MIN_PIXEL_SIZE = 0;
// Proxies are now tiny pre-decimated blobs (~180 KB) served by
// /api/scene-proxy, not the full ~64 MB PLY — so we can fetch many at once.
const PROXY_BUILD_CONCURRENCY = 8;
const pointCloudProxies = new Map(); // scene id → THREE.Points

// ── Point-cloud shader ───────────────────────────────────────────────────
// A custom shader instead of THREE.PointsMaterial for two reasons at once:
//
//   1. DISTANCE-BASED SIZE/FADE. Close points should read as "quite large",
//      far ones should fade out entirely — plain sizeAttenuation only
//      shrinks them, it doesn't clamp a max size or cut off a min.
//
//   2. CORRECT OCCLUSION AGAINST SPLATS. Spark's splats explicitly depth-test
//      against opaque scene geometry (see SparkRenderer's depthTest option),
//      so as long as these points stay OPAQUE (draw fully or not at all —
//      never alpha-blended) they occlude/get occluded by splats correctly.
//      The fade above is done with a dithered (screen-door) discard instead
//      of real alpha blending for exactly this reason: real transparency
//      would move points into THREE's transparent render queue, sorted by
//      object rather than per-pixel against the splats — which is what was
//      letting a proxy that should be hidden behind a splat show through.
const POINT_VERTEX_SHADER = /* glsl */ `
  attribute vec3 color;
  varying vec3 vColor;
  varying float vVisible;

  uniform float uPixelScale;   // world units -> pixels at 1 unit of distance
  uniform float uMaxPixelSize;
  uniform float uMinPixelSize;
  uniform float uFadeNear;
  uniform float uFadeFar;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float dist = -mvPosition.z;

    vVisible = 1.0 - smoothstep(uFadeNear, uFadeFar, dist);
    gl_PointSize = clamp(uPixelScale / max(dist, 0.001), uMinPixelSize, uMaxPixelSize);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const POINT_FRAGMENT_SHADER = /* glsl */ `
  varying vec3 vColor;
  varying float vVisible;

  float ditherThreshold(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
  }

  // Custom ShaderMaterial gets none of THREE's automatic linear->sRGB output
  // conversion that built-in materials apply — vColor arrives here already
  // decoded to linear (see srgbToLinear() on the JS side), so this is the
  // matching re-encode for display. Without it, colour comes out too dark in
  // the midtones; getting the decode/encode direction backwards instead
  // (encoding twice) is what produces oversaturated, blown-out colour.
  vec3 linearToSRGB(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
  }

  void main() {
    // Soft round sprite instead of a hard-edged square — feathered edge
    // reads as a natural blob and blends into neighbouring points instead
    // of tiling into an obvious flat-coloured mosaic.
    float d = length(gl_PointCoord - 0.5);
    float shape = 1.0 - smoothstep(0.3, 0.5, d);
    if (shape <= 0.0 || vVisible * shape < ditherThreshold(gl_FragCoord.xy)) discard;
    gl_FragColor = vec4(linearToSRGB(vColor), 1.0);
  }
`;

// Shared by every point cloud's material, so updating one entry (on resize)
// updates the on-screen size for every proxy at once.
const pointUniforms = {
  uPixelScale:   { value: 1 },
  uMaxPixelSize: { value: POINT_MAX_PIXEL_SIZE },
  uMinPixelSize: { value: POINT_MIN_PIXEL_SIZE },
  uFadeNear:     { value: POINT_FADE_NEAR },
  uFadeFar:      { value: POINT_FADE_FAR },
};

// uPixelScale depends on the camera's vertical FOV and the canvas height —
// the same relationship THREE's own sizeAttenuation uses internally. Call
// once the camera exists and again on every resize.
function updatePointPixelScale() {
  const cam = viewer?.camera;
  if (!cam) return;
  const fovRad = cam.fov * (Math.PI / 180);
  pointUniforms.uPixelScale.value = POINT_SIZE * (window.innerHeight / (2 * Math.tan(fovRad / 2)));
}

// Fetches the tiny pre-decimated proxy blob (see apps/api/proxy.py) and
// unpacks it into plain Float32 position/colour arrays. This is ~180 KB over
// the wire instead of the full ~64 MB PLY — the single change that takes the
// distant-view layer from ~45 s to a second or two for the whole archive.
async function fetchProxyPoints(sceneId) {
  const res = await fetch(`/api/scene-proxy/${encodeURIComponent(sceneId)}`);
  if (!res.ok) throw new Error(`proxy HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer()).buffer;
  const dv  = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "MPX1") throw new Error("bad proxy magic");
  const count = dv.getUint32(4, true);
  const positions = new Float32Array(buf, 8, count * 3);        // 8 is 4-byte aligned
  const colU8     = new Uint8Array(buf, 8 + count * 12, count * 3);
  const colors    = new Float32Array(count * 3);
  // These bytes are ordinary sRGB-encoded colour (same convention as any
  // photo pixel), but our point-cloud shader is a custom ShaderMaterial —
  // unlike built-in materials, THREE doesn't auto-convert vertex colours for
  // those, and it expects them in linear space. Decode sRGB->linear here so
  // the shader's own linear->sRGB encode (see POINT_FRAGMENT_SHADER) round-
  // trips back to the original colour instead of gamma-compounding it into
  // an oversaturated mess.
  for (let i = 0; i < colors.length; i++) colors[i] = srgbToLinear(colU8[i] / 255);
  return { positions, colors, count };
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

// Measure a scene's on-plane footprint from the proxy points (5th–95th
// percentile, ignoring stray outliers) so the world can pack scenes edge-to-
// edge and the overview camera can frame them tightly.
function recordExtentFromPoints(id, data) {
  if (sceneExtents.has(id)) return;
  const n = data.count;
  const xs = new Float32Array(n), ys = new Float32Array(n), zs = new Float32Array(n);
  let sumX = 0, sumY = 0, sumZ = 0;
  for (let i = 0; i < n; i++) {
    const x = data.positions[i * 3], y = data.positions[i * 3 + 1], z = data.positions[i * 3 + 2];
    xs[i] = x; ys[i] = y; zs[i] = z;
    sumX += x; sumY += y; sumZ += z;
  }
  sceneCentroids.set(id, [sumX / n, sumY / n, sumZ / n]);

  xs.sort(); ys.sort(); zs.sort();
  const lo = Math.floor(n * 0.05), hi = Math.max(lo, Math.ceil(n * 0.95) - 1);
  sceneExtents.set(id, {
    xSpan: Math.max(4, xs[hi] - xs[lo]),
    ySpan: Math.max(4, ys[hi] - ys[lo]),
    zSpan: Math.max(4, zs[hi] - zs[lo]),
  });
}

// World-space centre of mass for a scene: its map placement (worldPositions)
// plus the local centroid offset measured above — this is "where the splats
// actually are", as opposed to worldPositions alone, which is just the map
// slot the memory was assigned and stays fixed regardless of how its own
// content happens to be distributed around that slot's local origin. Falls
// back to the placement position alone if the centroid isn't known yet
// (proxy still loading).
function sceneWorldCenter(id) {
  const pos = worldPositions.get(id);
  if (!pos) return null;
  const c = sceneCentroids.get(id);
  if (!c) return pos;
  return [pos[0] + c[0], pos[1] + c[1], pos[2] + c[2]];
}

// ── Normalising memory size ───────────────────────────────────────────────
// Different captures end up wildly different physical sizes (room size,
// capture distance, etc.), which reads as inconsistent when walking between
// memories that are supposed to feel like one collection. Scenes noticeably
// bigger than their peers get scaled down toward the group's typical size;
// nothing is ever scaled up (a tiny memory blown up just looks blurrier, not
// "normal"), and nothing shrinks below its natural size either.
const sceneScales = new Map(); // scene id → uniform scale factor, 1 = untouched
let medianSceneSize = null;    // frozen per world-build; see establishSceneScales()
const SCENE_SCALE_TOLERANCE = 1.3; // up to 30% larger than the median is left alone

function sceneFootprintSize(id) {
  const ext = sceneExtents.get(id);
  return ext ? Math.max(ext.xSpan, ext.zSpan) : null;
}

function computeSceneScale(id) {
  const size = sceneFootprintSize(id);
  if (!size || !medianSceneSize) return 1;
  return size > medianSceneSize * SCENE_SCALE_TOLERANCE ? medianSceneSize / size : 1;
}

// Call once extents are known for a full batch of scenes (a world build) to
// (re)establish what "typical size" means for this set, then scale each one
// against it. Incremental arrivals (poll()) just call computeSceneScale()
// directly instead, measuring against this frozen baseline so already-placed
// memories don't visibly resize out from under someone exploring the world.
function establishSceneScales(sceneIds) {
  const sizes = sceneIds.map(sceneFootprintSize).filter(Boolean).sort((a, b) => a - b);
  if (sizes.length === 0) return;
  medianSceneSize = sizes[Math.floor(sizes.length / 2)];
  for (const id of sceneIds) sceneScales.set(id, computeSceneScale(id));
}

// Positions AND scales a point-cloud proxy or full splat mesh for a scene.
// worldPos is just the map slot the memory was assigned — scaling around
// that raw point would drag the scene's actual content off-slot, so this
// nudges position by the centroid offset (scaled the same amount) to keep
// the scene's own centre of mass anchored at worldPos + centroid regardless
// of how much (if any) it got shrunk.
function placeSceneObject(obj, id, worldPos) {
  const scale = sceneScales.get(id) ?? 1;
  const c = sceneCentroids.get(id) || [0, 0, 0];
  obj.scale.setScalar(scale);
  obj.position.set(
    worldPos[0] + c[0] * (1 - scale),
    worldPos[1] + c[1] * (1 - scale),
    worldPos[2] + c[2] * (1 - scale),
  );
}

async function addPointCloudProxy(scene, worldPos) {
  if (pointCloudProxies.has(scene.id) || !viewer) return;
  const t0 = performance.now();
  const data = await fetchProxyPoints(scene.id);
  if (!data || data.count < 1 || !viewer) return;
  recordExtentFromPoints(scene.id, data);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(data.colors, 3));
  // See the "Point-cloud shader" block above: opaque + dithered fade instead
  // of THREE.PointsMaterial, so size/visibility scale with distance from the
  // camera (large close up, invisible far away) without breaking occlusion
  // against the gaussian splats.
  const material = new THREE.ShaderMaterial({
    uniforms: pointUniforms,
    vertexShader: POINT_VERTEX_SHADER,
    fragmentShader: POINT_FRAGMENT_SHADER,
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });
  const points = new THREE.Points(geometry, material);
  // No scale established yet for a scene this new (see establishSceneScales)
  // — placeSceneObject degrades to scale=1, corrected once the batch is done.
  placeSceneObject(points, scene.id, worldPos);
  viewer.threeScene.add(points);
  pointCloudProxies.set(scene.id, points);
  console.log(`[viewer] proxy ready for ${scene.id}: ${data.count} pts in ${(performance.now() - t0).toFixed(0)}ms`);
}

function hidePointCloudProxy(sceneId) {
  const p = pointCloudProxies.get(sceneId);
  if (p) p.visible = false;
}

function showPointCloudProxy(sceneId) {
  const p = pointCloudProxies.get(sceneId);
  if (p) p.visible = true;
}

function clearPointCloudProxies() {
  for (const p of pointCloudProxies.values()) {
    viewer?.threeScene?.remove(p);
    p.geometry.dispose();
    p.material.dispose();
  }
  pointCloudProxies.clear();
}

// ── Debug: memory boundary boxes ─────────────────────────────────────────
// Green wireframe box per scene, sized from the same sceneExtents used to
// lay the world out and frame the overview camera — this is the actual
// bounding volume the app already computes, just made visible on demand.
let showBoundaryBoxes = false;
const boundaryBoxHelpers = new Map(); // scene id → THREE.Box3Helper

function refreshBoundaryBoxes() {
  if (!showBoundaryBoxes || !viewer) return;
  const liveIds = new Set();
  for (const s of worldScenes) {
    // Centred on sceneWorldCenter (not the raw placement slot) and scaled by
    // the same factor placeSceneObject() renders at, so this box matches the
    // scene's actual on-screen footprint — including any size normalisation.
    const center = sceneWorldCenter(s.id);
    if (!center) continue;
    liveIds.add(s.id);
    const ext = sceneExtents.get(s.id);
    const scale = sceneScales.get(s.id) ?? 1;
    const half = ext
      ? [ext.xSpan / 2 * scale, ext.ySpan / 2 * scale, ext.zSpan / 2 * scale]
      : [WORLD_SPACING / 2, WORLD_SPACING / 2, WORLD_SPACING / 2];
    const box = new THREE.Box3(
      new THREE.Vector3(center[0] - half[0], center[1] - half[1], center[2] - half[2]),
      new THREE.Vector3(center[0] + half[0], center[1] + half[1], center[2] + half[2]),
    );
    const existing = boundaryBoxHelpers.get(s.id);
    if (existing) {
      existing.box.copy(box);
    } else {
      const helper = new THREE.Box3Helper(box, new THREE.Color(0x00ff00));
      boundaryBoxHelpers.set(s.id, helper);
      viewer.threeScene.add(helper);
    }
  }
  // Drop helpers for scenes that dropped out of the current world set.
  for (const [id, helper] of boundaryBoxHelpers) {
    if (liveIds.has(id)) continue;
    viewer.threeScene.remove(helper);
    helper.geometry.dispose();
    helper.material.dispose();
    boundaryBoxHelpers.delete(id);
  }
}

function clearBoundaryBoxes() {
  for (const helper of boundaryBoxHelpers.values()) {
    viewer?.threeScene?.remove(helper);
    helper.geometry.dispose();
    helper.material.dispose();
  }
  boundaryBoxHelpers.clear();
}

function setBoundaryBoxesVisible(enabled) {
  showBoundaryBoxes = enabled;
  if (enabled) refreshBoundaryBoxes();
  else clearBoundaryBoxes();
}

// ── Full-resolution splats: load the closest N ───────────────────────────
// One rule: the MAX_LOADED_SPLATS scenes closest to the camera are shown as
// full gaussian splats; every other scene stays a lightweight point-cloud
// proxy. As you move, "closest" is recomputed each tick and the loaded set
// follows. No enter/exit distance, no direction, no preload/LRU — just
// "closest is loaded". When the closest scene changes, the new one loads
// before the old one is dropped, so the swap never leaves a blank frame.
//
// FULLRES_KEEP is the fraction of gaussians the server keeps in the on-demand
// full-res splat (see /api/scene-splat). 1.0 = original ~1.17M-gaussian scene;
// lower = lighter + proportionally faster to load, at some loss of density.
const FULLRES_KEEP        = 1;
const MAX_LOADED_SPLATS   = 2;

function splatUrl(scene) {
  // Decimated (lighter) full-res splat when the backend offers it; the raw
  // full-quality PLY otherwise.
  return scene.splat_url ? `${scene.splat_url}?keep=${FULLRES_KEEP}` : scene.ply_url;
}

// Fetches a scene's PLY bytes and builds a Spark SplatMesh at the given
// position. Fetched manually (rather than passing the URL straight to
// SplatMesh) because these URLs have no file extension for Spark's own
// format-sniffing to key off (they're routed through /api/scene-splat).
async function loadSplatMesh(url, position) {
  const res = await withLoadRetry(async () => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status} loading ${url}`);
    return r;
  });
  const fileBytes = await res.arrayBuffer();
  const mesh = new SplatMesh({ fileBytes, fileType: SplatFileType.PLY });
  mesh.position.set(position[0], position[1], position[2]);
  viewer.threeScene.add(mesh);
  await mesh.initialized;
  return mesh;
}

// ── Serialised splat load/unload ─────────────────────────────────────────
// Every add/remove goes through one queue, one at a time, so the loaded-set
// bookkeeping never gets mutated by two overlapping ops.
let loadedSplatIds = [];          // scene ids currently loaded
let splatMeshes    = new Map();   // scene id -> SplatMesh currently in the scene
let desiredIds     = [];          // scene ids that SHOULD be loaded (closest N)
let splatBusy      = false;       // an add/remove is in flight

function enqueueSplatOp(label, fn) {
  splatBusy = true;
  Promise.resolve()
    .then(fn)
    .catch(err => console.error(`[viewer] splat op '${label}' failed:`, err))
    .finally(() => { splatBusy = false; });
}

async function splatLoad(sceneId) {
  if (!viewer || loadedSplatIds.includes(sceneId)) return;
  const scene = worldScenes.find(s => s.id === sceneId) || scenes.find(s => s.id === sceneId);
  if (!scene) return;
  const t0 = performance.now();
  const pos = worldPositions.get(sceneId) || [0, 0, 0];
  const mesh = await loadSplatMesh(splatUrl(scene), pos);
  // Full-res quality gets the same size normalisation as its proxy, so the
  // swap between them isn't also a jarring size change.
  placeSceneObject(mesh, sceneId, pos);
  splatMeshes.set(sceneId, mesh);
  loadedSplatIds.push(sceneId);
  hidePointCloudProxy(sceneId);
  console.log(`[viewer] splat loaded ${sceneId} in ${(performance.now() - t0).toFixed(0)}ms (${loadedSplatIds.length} loaded)`);
}

async function splatUnload(sceneId) {
  const mesh = splatMeshes.get(sceneId);
  if (!mesh || !viewer) return;
  viewer.threeScene.remove(mesh);
  mesh.dispose();
  splatMeshes.delete(sceneId);
  const idx = loadedSplatIds.indexOf(sceneId);
  if (idx !== -1) loadedSplatIds.splice(idx, 1);
  showPointCloudProxy(sceneId);
  console.log(`[viewer] splat unloaded ${sceneId} (${loadedSplatIds.length} loaded)`);
}

function resetSplatState() {
  loadedSplatIds = [];
  splatMeshes = new Map();
  desiredIds = [];
  splatBusy = false;
}

// Debug render-mode toggle: "point cloud" forces every scene down to its
// proxy and blocks reconcileSplats() from loading anything (see the guard
// there) — "splat" (the normal default) resumes the usual closest-N loading.
let forcePointCloudOnly = false;

function setPointCloudOnly(enabled) {
  forcePointCloudOnly = enabled;
  if (!enabled) return; // normal loading just picks back up next frame
  splatBusy = true;
  (async () => {
    // Not routed through enqueueSplatOp — this needs to drain everything in
    // one go, not interleave with the reconciler (already blocked above).
    for (const id of [...loadedSplatIds]) {
      await splatUnload(id); // re-shows each scene's proxy as it unloads
    }
    desiredIds = [];
    splatBusy = false;
  })();
}

// Nudge the loaded set one step toward desiredIds. Does ONE op per call;
// updateFullResLOD keeps calling until settled. Loads a missing desired scene
// before dropping any stale one, so a swap never blanks the view.
function reconcileSplats() {
  if (!viewer || splatBusy || forcePointCloudOnly) return;

  // 1. Load the closest desired scene that isn't loaded yet. This runs BEFORE
  //    eviction so the incoming splat is up before the outgoing one leaves —
  //    a swap never blanks the view.
  const toLoad = desiredIds.find(id => !loadedSplatIds.includes(id));
  if (toLoad) { enqueueSplatOp(`load ${toLoad}`, () => splatLoad(toLoad)); return; }

  // 2. Everything desired is loaded — now drop ANY loaded scene that's no
  //    longer desired (not just when over the cap), so a fly across the world
  //    can't leave a pile of stale splats loaded. Converges to exactly the
  //    desired set once the camera settles.
  const victim = loadedSplatIds.find(id => !desiredIds.includes(id));
  if (victim) { enqueueSplatOp(`evict ${victim}`, () => splatUnload(victim)); return; }
}

// Called every few frames from updateLOD() with the camera's world position.
// Picks the closest MAX_LOADED_SPLATS scenes as the desired set, then lets
// reconcileSplats() load/unload toward it.
function updateFullResLOD(cx, cy, cz) {
  if (!worldMode || worldScenes.length === 0) return;

  // Rank by RAW distance to each scene's rendered centre of mass — the exact
  // point placeSceneObject() puts its proxy/splat at, so "closest" means
  // closest to what you actually see. No radius/edge fudge: subtracting a
  // footprint radius and clamping at 0 made every scene you're roughly inside
  // tie at distance 0, and with one splat slot the tie went to list order —
  // so a scene you're standing in could never win the slot from an earlier-
  // listed neighbour no matter how close you got.
  const ranked = [];
  for (const s of worldScenes) {
    const center = sceneWorldCenter(s.id);
    if (!center) continue;
    ranked.push({ id: s.id, d: Math.hypot(cx - center[0], cy - center[1], cz - center[2]) });
  }
  ranked.sort((a, b) => a.d - b.d);
  desiredIds = ranked.slice(0, MAX_LOADED_SPLATS).map(r => r.id);

  reconcileSplats();
}

// Move the camera to the canonical front view of a scene in Memory Verse.
// Audio for ALL scenes runs simultaneously — distance controls volume via
// updateSpatialAudio() each frame, so no audio changes happen here.
function focusWorldScene(sceneId, worldPos) {
  flyToScene(worldPos, true);
  worldFocusedId = sceneId;
  // Splat loading is purely distance-driven (see updateFullResLOD) — this
  // click/auto-advance/mobile-pick just flies the camera over; the splat
  // loads on its own once the camera is the closest thing to that scene.
  // Update auto-advance index so it continues from the current scene
  const idx = worldScenes.findIndex(s => s.id === sceneId);
  if (idx !== -1) _worldAutoIndex = idx;
  startWorldDwell(); // reset 60s countdown
  // Show story overlay for the focused memory (disappears after reading time)
  const scene = worldScenes.find(s => s.id === sceneId) || scenes.find(s => s.id === sceneId);
  if (storyOverlayTimer) { clearTimeout(storyOverlayTimer); storyOverlayTimer = null; }
  storyOverlayEl.classList.add("hidden");
  if (scene?.story) {
    overlayStoryTextEl.textContent = "";
    storyOverlayEl.classList.remove("hidden");
    const c = typewrite(overlayStoryTextEl, scene.story, 32);
    cancelTypewriters.push(c);
    const visibleMs = scene.story.length * 32 + 8000;
    storyOverlayTimer = setTimeout(() => storyOverlayEl.classList.add("hidden"), visibleMs);
  }
}

let chain = Promise.resolve();
const runExclusive = fn => { chain = chain.then(fn, fn); return chain; };

// ── Color extraction ──────────────────────────────────────────────────────

function hslToRgb(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
  };
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

async function extractSceneColor(imageUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    // No crossOrigin attribute — this image is same-origin (served by our
    // own backend), and setting crossOrigin="anonymous" anyway forces the
    // browser into CORS mode, which can silently fail to tag the response
    // as CORS-cleared depending on header quirks and taint the canvas. For
    // a same-origin image, leaving this off is both correct and simpler.
    img.onload = () => {
      try {
        const SIZE = 80;
        const canvas = document.createElement("canvas");
        canvas.width = SIZE; canvas.height = SIZE;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, SIZE, SIZE);
        const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

        let bestS = 0, bestH = 180;

        for (let i = 0; i < data.length; i += 4) {
          const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
          const max = Math.max(r, g, b), min = Math.min(r, g, b);
          const l = (max + min) / 2;
          // Skip very dark or blown-out pixels
          if (l < 0.12 || l > 0.92) continue;
          const d = max - min;
          const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
          if (s > bestS) {
            bestS = s;
            let h;
            if (max === r)      h = ((g - b) / d) % 6;
            else if (max === g) h = (b - r) / d + 2;
            else                h = (r - g) / d + 4;
            bestH = (h * 60 + 360) % 360;
          }
        }

        if (bestS < 0.14) {
          // B&W or very desaturated — bright cyan is most visible on dark bg
          resolve([100, 200, 255]);
          return;
        }

        // Lock saturation high, set L to 0.70 for max screen visibility
        resolve(hslToRgb(bestH / 360, Math.min(bestS * 1.3, 1), 0.70));
      } catch (err) {
        console.error("Colour extraction failed:", err);
        resolve([100, 200, 255]);
      }
    };
    img.onerror = () => resolve([100, 200, 255]);
    img.src = imageUrl;
  });
}

function applyHudColor([r, g, b]) {
  const val = `${r}, ${g}, ${b}`;
  hudEl.style.setProperty("--hc", val);
  storyOverlayEl.style.setProperty("--hc", val);
  // Set on root too so elements outside the HUD (e.g. the nav hint and the
  // custom cursor, which is styled with rgb(var(--hc))) inherit it
  document.documentElement.style.setProperty("--hc", val);
  // Tab icon matches the same colour as the cursor/HUD — same source, so
  // they're never out of sync with each other.
  setFaviconColor(r, g, b);
}

function setFaviconColor(r, g, b) {
  const iconLink = document.querySelector('link[rel="icon"]');
  if (!iconLink) return;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">`
    + `<circle cx="16" cy="16" r="15" fill="rgb(${r},${g},${b})"/>`
    + `<circle cx="16" cy="16" r="5" fill="#06080c"/></svg>`;
  iconLink.href = `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

// ── Typewriter ────────────────────────────────────────────────────────────

// Returns a cancel function so we can abort mid-type on transition
function typewrite(el, text, charDelayMs = 52) {
  el.textContent = "";
  // Add blinking cursor span
  const cursor = document.createElement("span");
  cursor.className = "hud-cursor";
  cursor.textContent = "█";
  el.appendChild(cursor);

  let i = 0;
  let cancelled = false;
  let timer = null;

  const step = () => {
    if (cancelled) return;
    if (i < text.length) {
      // Insert char before cursor
      cursor.insertAdjacentText("beforebegin", text[i++]);
      timer = setTimeout(step, charDelayMs);
    } else {
      // Done typing — blink cursor for 1.2s then remove it
      timer = setTimeout(() => {
        if (!cancelled) cursor.remove();
      }, 1200);
    }
  };
  timer = setTimeout(step, charDelayMs);

  return () => { cancelled = true; clearTimeout(timer); };
}

let cancelTypewriters = [];

function clearTypewriters() {
  cancelTypewriters.forEach(fn => fn());
  cancelTypewriters = [];
}

// ── Overlay ───────────────────────────────────────────────────────────────

function overlayTo(opacity) {
  return new Promise(resolve => {
    overlayEl.style.transition = `opacity ${OVERLAY_FADE_MS}ms ease`;
    overlayEl.style.opacity    = String(opacity);
    setTimeout(resolve, OVERLAY_FADE_MS);
  });
}

// ── Scene HUD ─────────────────────────────────────────────────────────────

async function showHud(scene, onDwellEnd) {
  clearTypewriters();

  // Extract color from source image then apply
  if (scene.image_url) {
    const color = await extractSceneColor(scene.image_url);
    applyHudColor(color);
  }

  captionNameEl.textContent  = "";
  captionAuthEl.textContent  = "";
  captionYearEl.textContent  = "";
  captionStoryEl.textContent = "";
  hudEl.classList.remove("hidden");

  // Per-scene audio — played spatially: single scenes are always at world
  // origin [0,0,0], so the sound comes from directly ahead of the camera.
  stopAllAudio();
  playSceneAudio(scene, [0, 0, 0]);

  const c1 = typewrite(captionNameEl, (scene.name || "Untitled").toUpperCase(), 55);
  cancelTypewriters.push(c1);

  if (scene.author) {
    authorRowEl.style.display = "";
    const c2 = typewrite(captionAuthEl, (scene.author).toUpperCase(), 48);
    cancelTypewriters.push(c2);
  } else {
    authorRowEl.style.display = "none";
  }

  if (scene.year) {
    yearRowEl.style.display = "";
    const c3 = typewrite(captionYearEl, scene.year, 60);
    cancelTypewriters.push(c3);
  } else {
    yearRowEl.style.display = "none";
  }

  // cluster_id is "<location-slug>__<decade>", assigned by the backend's
  // memory brain — show the era half as a quiet hint that this scene is
  // grouped with other memories of the same place and decade.
  const era = scene.cluster_id ? scene.cluster_id.split("__")[1] : "";
  if (era && era !== "undated") {
    clusterRowEl.style.display = "";
    const c6 = typewrite(captionClusterEl, era.toUpperCase(), 60);
    cancelTypewriters.push(c6);
  } else {
    clusterRowEl.style.display = "none";
  }

  if (scene.story) {
    storyRowEl.style.display = "";
    const c4 = typewrite(captionStoryEl, scene.story, 26);
    cancelTypewriters.push(c4);
  } else {
    storyRowEl.style.display = "none";
  }
  // Story overlay only appears when a memory is explicitly selected in explore mode

  // Start dwell ring once text starts appearing
  startDwell(onDwellEnd);
}

function hideHud() {
  clearTypewriters();
  hudEl.classList.add("hidden");
  storyOverlayEl.classList.add("hidden");
  if (storyOverlayTimer) { clearTimeout(storyOverlayTimer); storyOverlayTimer = null; }
  stopAllAudio();
}

// ── Dwell timer ───────────────────────────────────────────────────────────

let timerRaf      = null;
let timerStart    = null;
let timerCallback = null;

function startDwell(onComplete) {
  if (timerRaf) cancelAnimationFrame(timerRaf);
  timerRaf = null;
  timerCallback = onComplete;
  timerStart    = performance.now();
  timerArcEl.style.strokeDashoffset = TIMER_C;

  const tick = (now) => {
    const p = Math.min((now - timerStart) / DWELL_MS, 1);
    timerArcEl.style.strokeDashoffset = TIMER_C * (1 - p);
    if (p < 1) {
      timerRaf = requestAnimationFrame(tick);
    } else {
      timerRaf = null;
      if (timerCallback) timerCallback();
    }
  };
  timerRaf = requestAnimationFrame(tick);
}

function stopDwell() {
  if (timerRaf) cancelAnimationFrame(timerRaf);
  timerRaf = null; timerCallback = null;
  timerArcEl.style.strokeDashoffset = TIMER_C;
}

// ── World auto-advance ────────────────────────────────────────────────────
// After WORLD_DWELL_MS of no user selection, the camera flies to the next
// memory in the world and the cycle continues indefinitely.

let _worldDwellTimer = null;
let _worldAutoIndex  = 0;

function startWorldDwell() {
  clearTimeout(_worldDwellTimer);
  _worldDwellTimer = setTimeout(_worldAdvance, WORLD_DWELL_MS);
}

function stopWorldDwell() {
  clearTimeout(_worldDwellTimer);
  _worldDwellTimer = null;
}

function _worldAdvance() {
  if (!worldMode || worldScenes.length === 0) return;
  _worldAutoIndex = (_worldAutoIndex + 1) % worldScenes.length;
  const s = worldScenes[_worldAutoIndex];
  const pos = worldPositions.get(s.id);
  if (pos) {
    focusWorldScene(s.id, pos);
  } else {
    // Position not ready yet — skip to next after a short gap
    _worldDwellTimer = setTimeout(_worldAdvance, 2000);
  }
}

// ── Gaussian viewer (Spark) ──────────────────────────────────────────────
// There's no all-in-one "Viewer" class here like the old library had — Spark
// is just a THREE.Object3D (SparkRenderer) you add to a normal THREE scene,
// so we own the scene/camera/renderer/render-loop directly. `viewer` keeps
// the same shape (.camera / .threeScene) the rest of this file already
// expects, to keep this swap as close to a drop-in as possible.
//
// No built-in controls to disable here (unlike the old library, which had
// its own OrbitControls + a window-level keydown handler that had to be
// turned off) — camera control has always been our own applyFPSCamera()/
// flyLoop() below, driving a plain THREE.PerspectiveCamera.

function ensureViewer() {
  if (viewer) return;
  console.log("[viewer] creating Spark renderer");

  // Matches the ~55°/tan≈0.52 FOV assumed by worldOverviewPos()'s framing math.
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.05, 4000);
  camera.position.set(...INIT_POS);

  const renderer = new THREE.WebGLRenderer({ antialias: false }); // AA doesn't help splats, only costs perf (Spark docs)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewerEl.appendChild(renderer.domElement);

  const threeScene = new THREE.Scene();
  const spark = new SparkRenderer({ renderer });
  threeScene.add(spark);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    updatePointPixelScale();
  });

  viewer = { camera, threeScene, renderer, spark };
  applyFPSCamera();
  updatePointPixelScale();
  console.log("[viewer] Spark renderer ready");
}

// ── FPS camera state ──────────────────────────────────────────────────────

let _yaw   = 0;   // horizontal look angle (radians)
let _pitch = 0;   // vertical look angle   (radians)

// Applies yaw/pitch to the camera by building an explicit orthonormal basis
// (right / up / forward) from a fixed world-up reference. This avoids the
// roll/skew artifacts that cam.lookAt() can introduce near steep pitch
// angles (its internal up-vector cross-product becomes unstable there).
// cameraUp is (0,-1,0) in this scene, so world -Y is visually "up".
function applyFPSCamera() {
  const cam = viewer?.camera;
  if (!cam) return;

  const cosP = Math.cos(_pitch);
  const forward = {
    x: Math.sin(_yaw) * cosP,
    y: -Math.sin(_pitch),
    z: Math.cos(_yaw) * cosP,
  };
  const worldUp = { x: 0, y: -1, z: 0 };

  // right = forward × worldUp, normalized
  let right = {
    x: forward.y * worldUp.z - forward.z * worldUp.y,
    y: forward.z * worldUp.x - forward.x * worldUp.z,
    z: forward.x * worldUp.y - forward.y * worldUp.x,
  };
  const rLen = Math.hypot(right.x, right.y, right.z) || 1;
  right = { x: right.x / rLen, y: right.y / rLen, z: right.z / rLen };

  // camUp = right × forward — always orthogonal, never twists/rolls
  const camUp = {
    x: right.y * forward.z - right.z * forward.y,
    y: right.z * forward.x - right.x * forward.z,
    z: right.x * forward.y - right.y * forward.x,
  };

  // Build vectors via the camera's own Vector3 constructor (no THREE import needed)
  const Vec3   = cam.position.constructor;
  const xAxis  = new Vec3(right.x, right.y, right.z);
  const yAxis  = new Vec3(camUp.x, camUp.y, camUp.z);
  const zAxis  = new Vec3(-forward.x, -forward.y, -forward.z); // camera looks down local -Z

  cam.matrix.makeBasis(xAxis, yAxis, zAxis);
  cam.quaternion.setFromRotationMatrix(cam.matrix);
  cam.updateMatrixWorld(true);
}

// ── Mouse FPS-look + click-to-reset ──────────────────────────────────────

let _dragging  = false;
let _prevX     = 0, _prevY = 0;
let _dragDist  = 0;

viewerEl.addEventListener("pointerdown", e => {
  _dragging = true;
  _prevX    = e.clientX;
  _prevY    = e.clientY;
  _dragDist = 0;
  viewerEl.setPointerCapture(e.pointerId);
});

viewerEl.addEventListener("pointermove", e => {
  if (!_dragging) return;
  const dx = e.clientX - _prevX;
  const dy = e.clientY - _prevY;
  _dragDist += Math.abs(dx) + Math.abs(dy);

  const SENS = 0.0025;
  _yaw   -= dx * SENS;
  _pitch -= dy * SENS;
  _pitch  = Math.max(-1.30, Math.min(1.30, _pitch)); // clamp ~74°, stays well clear of gimbal zone

  _prevX = e.clientX;
  _prevY = e.clientY;

  applyFPSCamera();
});

viewerEl.addEventListener("pointerup", e => {
  if (_dragDist < 6) {
    if (worldMode) {
      // Always find the nearest visible scene — no radius cutoff.
      // If all scene centers are behind the camera (deeply inside one),
      // fall back to the overview so the user can re-orient.
      const target = findNearestWorldSceneAndId(e.clientX, e.clientY);
      if (target) focusWorldScene(target.id, target.pos);
      else flyToWorldOverview();
    }
    // Single tap no longer resets camera — use double-click to reset
  }
  _dragging = false;
});

viewerEl.addEventListener("dblclick", () => {
  if (!worldMode) resetCamera();
});

// Scroll = move forward / backward along look direction
viewerEl.addEventListener("wheel", e => {
  e.preventDefault();
  if (!viewer?.camera) return;
  const cam = viewer.camera;
  const m   = cam.matrixWorld.elements;
  const fwd = { x: -m[8], y: -m[9], z: -m[10] };
  const d   = e.deltaY > 0 ? 0.18 : -0.18;
  cam.position.x += fwd.x * d;
  cam.position.y += fwd.y * d;
  cam.position.z += fwd.z * d;
  applyFPSCamera();
}, { passive: false });

// ── Click-to-reset (fly back to origin) ──────────────────────────────────

function resetCamera() {
  if (!viewer?.camera) return;
  if (resetRaf) cancelAnimationFrame(resetRaf);

  const cam = viewer.camera;
  const sp  = { x: cam.position.x, y: cam.position.y, z: cam.position.z };
  const ep  = { x: INIT_POS[0],    y: INIT_POS[1],    z: INIT_POS[2]    };
  const sy  = _yaw, sp2 = _pitch;

  const t0  = performance.now();
  const DUR = 900;

  const tick = now => {
    const raw  = Math.min((now - t0) / DUR, 1);
    const ease = 1 - Math.pow(1 - raw, 3);  // cubic ease-out

    cam.position.x = sp.x + (ep.x - sp.x) * ease;
    cam.position.y = sp.y + (ep.y - sp.y) * ease;
    cam.position.z = sp.z + (ep.z - sp.z) * ease;

    // Simultaneously sweep yaw/pitch back to zero
    _yaw   = sy  * (1 - ease);
    _pitch = sp2 * (1 - ease);
    applyFPSCamera();

    if (raw < 1) {
      resetRaf = requestAnimationFrame(tick);
    } else {
      _yaw = 0; _pitch = 0;
      applyFPSCamera();
      resetRaf = null;
    }
  };
  resetRaf = requestAnimationFrame(tick);
}

// Generalised version of the same tween, used by World of Memories to fly
// the camera to a specific scene instead of always returning to the origin.
// Approaches from whichever side the camera already happens to be on, so the
// camera glides over rather than teleporting around the target.
// fromFront=true: always approach from the -Z side (canonical Memory Verse
// front view), regardless of where the camera currently is. Prevents the
// "black side" problem when the user navigated behind a scene.
function flyToScene(targetPos, fromFront = false) {
  if (!viewer?.camera) return;
  if (resetRaf) cancelAnimationFrame(resetRaf);

  const cam = viewer.camera;
  const sp  = { x: cam.position.x, y: cam.position.y, z: cam.position.z };

  let appX, appZ;
  if (fromFront) {
    appX = 0; appZ = -1; // canonical -Z approach (matching overview direction)
  } else {
    appX = sp.x - targetPos[0];
    appZ = sp.z - targetPos[2];
    const appLen = Math.hypot(appX, appZ) || 1;
    appX /= appLen; appZ /= appLen;
  }

  const STANDOFF = 3.2;
  const ep = {
    x: targetPos[0] + appX * STANDOFF,
    y: targetPos[1],
    z: targetPos[2] + appZ * STANDOFF,
  };

  const dx = targetPos[0] - ep.x;
  const dy = targetPos[1] - ep.y;
  const dz = targetPos[2] - ep.z;
  const horizDist = Math.hypot(dx, dz) || 1e-6;
  const targetYaw   = Math.atan2(dx, dz);
  const targetPitch = Math.max(-1.30, Math.min(1.30, -Math.atan2(dy, horizDist)));

  const sy = _yaw, sp2 = _pitch;
  // Shortest-path yaw delta, so it never spins the long way round
  let yawDelta = targetYaw - sy;
  yawDelta = ((yawDelta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

  const t0  = performance.now();
  const DUR = 1100;

  const tick = now => {
    const raw  = Math.min((now - t0) / DUR, 1);
    const ease = 1 - Math.pow(1 - raw, 3);

    cam.position.x = sp.x + (ep.x - sp.x) * ease;
    cam.position.y = sp.y + (ep.y - sp.y) * ease;
    cam.position.z = sp.z + (ep.z - sp.z) * ease;

    _yaw   = sy  + yawDelta * ease;
    _pitch = sp2 + (targetPitch - sp2) * ease;
    applyFPSCamera();

    if (raw < 1) {
      resetRaf = requestAnimationFrame(tick);
    } else {
      resetRaf = null;
    }
  };
  resetRaf = requestAnimationFrame(tick);
}

// Computes a camera position that frames the whole 3D cloud of memories: the
// axis-aligned bounding box over every scene (centre ± half-extent), then a
// pull-back along -Z far enough to fit its width/height, plus a margin for its
// depth. Looks straight down +Z at the cloud's centre.
function worldOverviewPos() {
  const activeScenes = worldScenes.length ? worldScenes : [];
  if (activeScenes.length === 0) return { x: 0, y: 0, z: -30, yaw: 0, pitch: 0 };

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const s of activeScenes) {
    const pos = worldPositions.get(s.id);
    if (!pos) continue;
    const ext = sceneExtents.get(s.id);
    const half = [
      ext ? ext.xSpan / 2 : WORLD_SPACING / 2,
      ext ? ext.ySpan / 2 : WORLD_SPACING / 2,
      ext ? ext.zSpan / 2 : WORLD_SPACING / 2,
    ];
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], pos[k] - half[k]);
      max[k] = Math.max(max[k], pos[k] + half[k]);
    }
  }
  if (min[0] === Infinity) return { x: 0, y: 0, z: -30, yaw: 0, pitch: 0 };

  const cx = (min[0] + max[0]) / 2;
  const cy = (min[1] + max[1]) / 2;
  const cz = (min[2] + max[2]) / 2;
  const width  = max[0] - min[0];
  const height = max[1] - min[1];
  const depth  = max[2] - min[2];
  // Pull back to fit the larger of width/height in the FOV (~55°, tan≈0.52),
  // then add half the depth plus a gap so nothing near the front clips.
  const pullback = Math.max((Math.max(width, height) / 2) / 0.5 + depth / 2 + SCENE_GAP, 30);
  return { x: cx, y: cy, z: cz - pullback, yaw: 0, pitch: 0 };
}

// Smoothly fly to the world overview — used as the Memory Verse "reset"
// when a click lands on empty space (mirrors resetCamera for single scenes).
function flyToWorldOverview() {
  if (!viewer?.camera) return;
  if (resetRaf) cancelAnimationFrame(resetRaf);

  const cam = viewer.camera;
  const sp = { x: cam.position.x, y: cam.position.y, z: cam.position.z };
  const { x: ex, y: ey, z: ez, yaw: ty, pitch: tp } = worldOverviewPos();
  const ep = { x: ex, y: ey, z: ez };

  const sy = _yaw, sp2 = _pitch;
  let yawDelta = ty - sy;
  yawDelta = ((yawDelta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

  const t0 = performance.now();
  const DUR = 1100;

  const tick = now => {
    const raw  = Math.min((now - t0) / DUR, 1);
    const ease = 1 - Math.pow(1 - raw, 3);

    cam.position.x = sp.x + (ep.x - sp.x) * ease;
    cam.position.y = sp.y + (ep.y - sp.y) * ease;
    cam.position.z = sp.z + (ep.z - sp.z) * ease;

    _yaw   = sy  + yawDelta * ease;
    _pitch = sp2 + (tp - sp2) * ease;
    applyFPSCamera();

    if (raw < 1) {
      resetRaf = requestAnimationFrame(tick);
    } else {
      _yaw = ty; _pitch = tp;
      applyFPSCamera();
      resetRaf = null;
    }
  };
  resetRaf = requestAnimationFrame(tick);
}

// Projects a world position to on-screen pixel coordinates, using the
// viewer's own camera/canvas — used to find which scene a click landed near.
function projectToScreen(worldPos) {
  const cam = viewer?.camera;
  if (!cam) return null;
  const Vec3 = cam.position.constructor;
  const v = new Vec3(worldPos[0], worldPos[1], worldPos[2]);
  v.project(cam);
  if (v.z > 1) return null; // behind the camera
  const rect = viewerEl.getBoundingClientRect();
  return {
    x: (v.x * 0.5 + 0.5) * rect.width + rect.left,
    y: (1 - (v.y * 0.5 + 0.5)) * rect.height + rect.top,
  };
}

// Always returns the nearest scene in worldScenes — no radius cutoff.
// Scenes whose center is behind the camera are skipped (projectToScreen returns null),
// but any visible scene is always reachable with a single tap.
function findNearestWorldScene(clickX, clickY) {
  let best = null, bestDist = Infinity;
  for (const scene of worldScenes) {
    const pos = worldPositions.get(scene.id);
    if (!pos) continue;
    const screen = projectToScreen(pos);
    if (!screen) continue;
    const d = Math.hypot(screen.x - clickX, screen.y - clickY);
    if (d < bestDist) { bestDist = d; best = pos; }
  }
  return best;
}

function findNearestWorldSceneAndId(clickX, clickY) {
  let best = null, bestDist = Infinity;
  for (const scene of worldScenes) {
    const pos = worldPositions.get(scene.id);
    if (!pos) continue;
    const screen = projectToScreen(pos);
    if (!screen) continue;
    const d = Math.hypot(screen.x - clickX, screen.y - clickY);
    if (d < bestDist) { bestDist = d; best = { pos, id: scene.id }; }
  }
  return best;
}

// ── World of Memories ────────────────────────────────────────────────────────
// The viewer is always in world mode — all memories live in one shared 3D
// environment. Positions come from /api/scene-positions (set by the mobile
// word cloud). Every scene gets a cheap point-cloud proxy up front; only the
// scene the visitor is currently near loads at full gaussian-splat quality
// (see updateFullResLOD()) — this is what actually bounds GPU/CPU cost
// regardless of how many memories exist in total.

async function buildWorldMode(sceneIds) {
  console.log("[viewer] buildWorldMode start", { requested: sceneIds, totalKnownScenes: scenes.length });
  await fetchServerPositions();
  console.log("[viewer] server positions fetched:", serverPositions.size);
  let targetScenes = sceneIds
    ? scenes.filter(s => sceneIds.includes(s.id))
    : scenes;
  if (DEBUG_LIMIT != null) {
    targetScenes = targetScenes.slice(0, DEBUG_LIMIT);
    console.log(`[viewer] DEBUG_LIMIT active — loading only ${targetScenes.length} scene(s):`, targetScenes.map(s => s.id));
  }
  if (targetScenes.length === 0) {
    console.warn("[viewer] buildWorldMode: no target scenes, aborting");
    return;
  }

  worldScenes = targetScenes;
  worldFocusedId = null;
  // Rough even spread now (before extents are known) so the camera has
  // something to frame; re-spread with real footprints once proxies load.
  relayoutWorldEven(targetScenes);

  worldMode = true;
  worldLoadedOrder = []; // proxy-loaded scene ids, in no particular order
  stopDwell();
  stopWorldDwell();
  hideHud();

  return runExclusive(async () => {
    await overlayTo(1);

    try {
      await removeAllScenes();
    } catch (err) {
      console.error("Failed clearing world:", err);
    }
    clearPointCloudProxies();
    clearBoundaryBoxes();
    resetSplatState();
    ensureViewer();
    stopAllAudio();

    // Camera framing and auto-advance only depend on the positions already
    // assigned above — not on any splat actually being loaded — so set them
    // up right away instead of blocking behind the full load.
    if (viewer?.camera) {
      const { x, y, z, yaw, pitch } = worldOverviewPos();
      viewer.camera.position.set(x, y, z);
      _yaw = yaw; _pitch = pitch;
      applyFPSCamera();
      console.log("[viewer] camera placed at overview", { x, y, z, yaw, pitch });
    }

    const midIdx = Math.floor(targetScenes.length / 2);
    const midScene = targetScenes[midIdx];
    if (midScene) {
      worldFocusedId = midScene.id;
      _worldAutoIndex = worldScenes.findIndex(s => s.id === midScene.id);
      if (_worldAutoIndex === -1) _worldAutoIndex = midIdx;
    }
    startWorldDwell();

    // Build a lightweight point-cloud proxy for every scene. This is the
    // ONLY thing that loads for all N memories up front — each proxy is a
    // few thousand plain THREE.Points, nowhere near the cost of a full
    // gaussian splat (covariance/colour textures, octree, sort worker).
    // Full quality is loaded on demand per-scene by updateFullResLOD() as
    // the visitor actually approaches, and by focusWorldScene() when a
    // memory is explicitly selected.
    if (splashStatusEl) splashStatusEl.textContent = "entering the memory verse…";
    if (splashCountEl)  splashCountEl.textContent  = "";

    const total = targetScenes.length;
    console.log(`[viewer] building ${total} point-cloud proxy(ies), ${PROXY_BUILD_CONCURRENCY} at a time...`);
    const t0 = performance.now();
    let doneCount = 0;
    await mapWithConcurrency(targetScenes, PROXY_BUILD_CONCURRENCY, async s => {
      try {
        await addPointCloudProxy(s, worldPositions.get(s.id) || assignWorldPosition(s));
        worldLoadedOrder.push(s.id);
      } catch (err) {
        console.error(`[viewer] ✗ failed building proxy for ${s.id}:`, err);
      }
      doneCount++;
      if (splashCountEl) splashCountEl.textContent = `${doneCount} / ${total}`;
    });
    console.log(`[viewer] proxies ready: ${worldLoadedOrder.length}/${total} in ${(performance.now() - t0).toFixed(0)}ms`);

    // Now that every proxy is loaded we know each scene's real footprint, so
    // re-spread everything evenly with a lattice sized to the true largest
    // scene — this is the pass that actually guarantees no overlaps. Then
    // slide each proxy to its final home and re-frame the overview so the whole
    // 3D cloud fits the view.
    relayoutWorldEven(targetScenes);
    establishSceneScales(targetScenes.map(s => s.id));
    for (const s of targetScenes) {
      const p   = pointCloudProxies.get(s.id);
      const pos = worldPositions.get(s.id);
      if (p && pos) placeSceneObject(p, s.id, pos);
    }
    refreshBoundaryBoxes();
    if (viewer?.camera && DEBUG_LIMIT == null) {
      const { x, y, z, yaw, pitch } = worldOverviewPos();
      viewer.camera.position.set(x, y, z);
      _yaw = yaw; _pitch = pitch;
      applyFPSCamera();
      console.log("[viewer] camera re-framed to overview after extents known", { x, y, z });
    }

    // Audio doesn't depend on splat quality — start it for every scene now,
    // same distance-based spatial mix as before.
    for (const s of targetScenes) playSceneAudio(s, worldPositions.get(s.id) || [0, 0, 0]);

    if (splashStatusEl) splashStatusEl.textContent = "memory verse ready";
    placeholderEl.classList.add("hidden");
    overlayEl.style.transition = "opacity 1000ms ease-in";
    overlayEl.style.opacity = "0";

    // Debug mode: fly straight to the (only) scene and load it at full
    // quality so it's obvious the camera actually reached it.
    if (DEBUG_LIMIT != null && midScene) {
      const pos = worldPositions.get(midScene.id);
      console.log("[viewer] DEBUG_LIMIT: flying camera to scene", midScene.id, pos);
      if (pos) focusWorldScene(midScene.id, pos);
    }
  });
}

// ── Transition ────────────────────────────────────────────────────────────

// Network fetches can flake (especially the ~64MB full-res PLYs) — a couple
// of short retries absorbs that instead of the whole load failing outright.
async function withLoadRetry(fn, attempts = 3, delayMs = 150) {
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === attempts - 1) throw err;
      await new Promise(r => setTimeout(r, delayMs));
    }
  }
}

async function removeAllScenes() {
  if (!viewer) return;
  for (const mesh of splatMeshes.values()) {
    viewer.threeScene.remove(mesh);
    mesh.dispose();
  }
  splatMeshes.clear();
}

async function transitionTo(scene) {
  hideHud();
  await overlayTo(1);

  try {
    await removeAllScenes();
  } catch (err) {
    console.error("Failed clearing the previous scene:", err);
  }
  placeholderEl.classList.add("hidden");
  ensureViewer();

  let loadPromise;
  try {
    loadPromise = loadSplatMesh(scene.ply_url, [0, 0, 0]).then(mesh => {
      splatMeshes.set(scene.id, mesh);
    });
  } catch (err) {
    console.error("loadSplatMesh threw:", err);
    await overlayTo(0);
    return null;
  }

  // Slow ease-in reveal — splats materialise as overlay lifts
  overlayEl.style.transition = "opacity 2200ms ease-in";
  overlayEl.style.opacity    = "0";
  await new Promise(r => setTimeout(r, 2200));

  return loadPromise;
}

// ── Scene management ──────────────────────────────────────────────────────

function goToIndex(index) {
  return runExclusive(async () => {
    if (index < 0 || index >= scenes.length) return;
    const scene = scenes[index];
    currentIndex = index;
    if (scene.id === activeSceneId) return;
    stopDwell();
    try {
      const loadPromise = await transitionTo(scene);
      // Must actually wait for the splat to finish loading here, inside the
      // exclusive lock — releasing it before this settles let the next
      // goToIndex call collide mid-load, which is exactly what was producing
      // "wrong scene, audio doesn't match, jumps around fast".
      if (loadPromise) {
        try {
          await loadPromise;
        } catch (err) {
          console.error("Splat load failed for", scene.id, err);
        }
      }
      activeSceneId = scene.id;
      const advance = () => {
        if (scenes.length < 2) {
          // Only one scene — restart the dwell
          showHud(scene, advance);
        } else {
          goToIndex((currentIndex + 1) % scenes.length);
        }
      };
      showHud(scene, advance);
    } catch (err) {
      console.error("Transition failed:", err);
    }
  });
}

// ── Polling ───────────────────────────────────────────────────────────────

async function poll() {
  try {
    const scenesRes = await fetch("/api/scenes", { cache: "no-store" });
    if (!scenesRes.ok) throw new Error(`HTTP ${scenesRes.status}`);
    clearTimeout(errorTimer);
    connectionEl.dataset.state = "";

    const all = await scenesRes.json();
    console.log(`[viewer] poll: /api/scenes returned ${all.length} total scene(s)`);
    if (!all.length) return;
    const known = new Set(scenes.map(s => s.id));
    const fresh = all.filter(s => !known.has(s.id));
    if (!fresh.length) return;
    console.log(`[viewer] poll: ${fresh.length} new scene(s):`, fresh.map(s => s.id));
    scenes.push(...fresh);
    // Pull the server's map positions first so a brand-new memory lands on its
    // authoritative (non-overlapping) spot instead of a fallback cell.
    await fetchServerPositions();
    fresh.forEach(assignWorldPosition);

    if (worldMode) {
      if (DEBUG_LIMIT != null) {
        console.log("[viewer] poll: DEBUG_LIMIT active — skipping in-place add of new memories");
      } else {
        // Add newly-arrived memories straight into the world, in place,
        // without disturbing whatever's already there or anyone exploring
        // it. Just a point-cloud proxy each, same as the initial build —
        // full quality only loads later, automatically, if the visitor
        // actually walks near one (updateFullResLOD).
        if (viewer) {
          console.log(`[viewer] poll: already in world mode — adding ${fresh.length} scene(s) as proxies`);
          const t0 = performance.now();
          await mapWithConcurrency(fresh, PROXY_BUILD_CONCURRENCY, async s => {
            try {
              await addPointCloudProxy(s, worldPositions.get(s.id) || [0, 0, 0]);
              worldLoadedOrder.push(s.id);
              worldScenes.push(s);
              playSceneAudio(s, worldPositions.get(s.id) || [0, 0, 0]);
            } catch (err) {
              console.error("[viewer] poll: failed adding proxy for new memory:", s.id, err);
            }
          });
          // Measure against the existing (frozen) median rather than
          // re-establishing it, so memories already on display don't
          // visibly resize just because a new arrival changed the average.
          for (const s of fresh) {
            const pos = worldPositions.get(s.id);
            const p   = pointCloudProxies.get(s.id);
            sceneScales.set(s.id, computeSceneScale(s.id));
            if (p && pos) placeSceneObject(p, s.id, pos);
          }
          refreshBoundaryBoxes();
          console.log(`[viewer] poll: added ${fresh.length} scene(s) to world in ${(performance.now() - t0).toFixed(0)}ms`);
        }
      }
    } else if (!started) {
      // Auto-enter world mode with all memories on first load
      console.log("[viewer] poll: first load — entering world mode");
      started = true;
      await fetchServerPositions();
      buildWorldMode(null);
    } else {
      // New memory arrived while in non-world mode — rebuild world to include it
      console.log("[viewer] poll: rebuilding world mode to include new memory");
      await fetchServerPositions();
      buildWorldMode(null);
    }
  } catch (err) {
    console.error("[viewer] poll failed:", err);
    clearTimeout(errorTimer);
    connectionEl.dataset.state = "error";
    errorTimer = setTimeout(() => { connectionEl.dataset.state = ""; }, ERROR_SHOW_MS);
  }
}

async function pollStatus() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (!res.ok) return;
    const { processing } = await res.json();
    loaderEl.classList.toggle("hidden", !processing);
  } catch { /* ignore */ }
}

// ── Mobile-app scene selection ──────────────────────────────────────────
// Purely additive: the mobile app's "browse memories" page can ask the
// viewer to jump to a specific memory. This only decides *which* index to
// jump to — it calls the same goToIndex() the normal rotation already uses,
// so dwell time, transitions, and auto-advance afterward are all unchanged.

let lastSelectionAt = 0;

async function pollSelection() {
  try {
    const res = await fetch("/api/select-scene", { cache: "no-store" });
    if (!res.ok) return;
    const { scene_id, selected_at } = await res.json();
    if (!scene_id || !selected_at || selected_at <= lastSelectionAt) return;
    lastSelectionAt = selected_at;

    let index = scenes.findIndex(s => s.id === scene_id);
    if (index === -1) {
      // Requested scene isn't in our local list yet — refresh and retry once.
      await poll();
      index = scenes.findIndex(s => s.id === scene_id);
    }
    if (index === -1) return;

    // Always fly in world mode — single-scene goToIndex is never used
    const s = scenes[index];
    const pos = worldPositions.get(s.id);
    if (pos && worldMode) {
      focusWorldScene(s.id, pos);
    } else if (!worldMode && scenes.length > 0) {
      // World not ready yet — queue the focus for when it finishes loading
      const waitAndFocus = setInterval(() => {
        if (!worldMode) return;
        clearInterval(waitAndFocus);
        const p = worldPositions.get(s.id);
        if (p) focusWorldScene(s.id, p);
      }, 500);
    }
  } catch { /* ignore */ }
}

let lastWorldSelectionAt = 0;
let worldSelectionBaselined = false;

// Mobile app multi-selects which memories to place in Memory Verse, then
// calls /api/world-selection — this is what actually enters/rebuilds the
// world with exactly that set, on whichever screen the viewer is open.
async function pollWorldSelection() {
  try {
    const res = await fetch("/api/world-selection", { cache: "no-store" });
    if (!res.ok) return;
    const { scene_ids, selected_at } = await res.json();

    // The very first check just records whatever's already sitting on the
    // server (e.g. left over from an earlier session) as the baseline,
    // instead of treating it as a brand-new request — without this, the
    // viewer would jump straight into Memory Verse on every page load
    // whenever someone had used the mobile selector previously. Every
    // viewer always starts on the normal single-scene rotation; only a
    // selection made *after* this point should ever open the world.
    if (!worldSelectionBaselined) {
      worldSelectionBaselined = true;
      lastWorldSelectionAt = selected_at || 0;
      return;
    }

    if (!scene_ids?.length || !selected_at || selected_at <= lastWorldSelectionAt) return;
    lastWorldSelectionAt = selected_at;
    buildWorldMode(scene_ids);
  } catch { /* ignore */ }
}

// ── Fullscreen (kiosk) mode ───────────────────────────────────────────────

function isFullscreen() {
  return !!document.fullscreenElement;
}

let _fullscreenRequested = false;

function enterFullscreen() {
  // Guards against a double-click firing this twice before the first
  // requestFullscreen() call resolves — that second call would otherwise log
  // a harmless but noisy "can only be initiated by a user gesture" warning.
  if (_fullscreenRequested || isFullscreen() || !document.documentElement.requestFullscreen) return;
  _fullscreenRequested = true;
  document.documentElement.requestFullscreen()
    .catch(() => {})
    .finally(() => { _fullscreenRequested = false; });
}

function toggleFullscreen() {
  if (isFullscreen()) {
    document.exitFullscreen?.();
  } else {
    enterFullscreen();
  }
}

// Fullscreen is opt-in only — the bottom-right button and the F key (below)
// are the only two things that ever call toggleFullscreen()/enterFullscreen().

// Placeholder stays visible until buildWorldMode finishes loading all scenes

// Manual toggle for testing/operator use
document.addEventListener("keydown", e => {
  if (e.code === "KeyF") toggleFullscreen();
});

// ── Debug / utility control buttons (bottom-right) ────────────────────────

boxesBtn?.addEventListener("click", () => {
  const next = !showBoundaryBoxes;
  setBoundaryBoxesVisible(next);
  boxesBtn.classList.toggle("active", next);
  boxesBtn.setAttribute("aria-pressed", String(next));
});

musicBtn?.addEventListener("click", () => {
  const enabling = audioMuted; // currently muted → this click turns it on
  setMusicEnabled(enabling);
  musicBtn.classList.toggle("active", enabling);
  musicBtn.setAttribute("aria-pressed", String(enabling));
});

fullscreenBtn?.addEventListener("click", () => toggleFullscreen());
document.addEventListener("fullscreenchange", () => {
  const fs = isFullscreen();
  fullscreenBtn?.classList.toggle("active", fs);
  fullscreenBtn?.setAttribute("aria-pressed", String(fs));
});

renderModeBtn?.addEventListener("click", () => {
  const next = !forcePointCloudOnly;
  setPointCloudOnly(next);
  renderModeBtn.textContent = next ? "POINTS" : "SPLAT";
  renderModeBtn.classList.toggle("active", next);
  renderModeBtn.setAttribute("aria-pressed", String(next));
});

// ── Keyboard fly navigation ───────────────────────────────────────────────

const _keys = new Set();
document.addEventListener("keydown", e => {
  _keys.add(e.code);
  // Stop browser from scrolling the page with arrow keys / space
  if (["Space","ArrowUp","ArrowDown","ArrowLeft","ArrowRight",
       "PageUp","PageDown"].includes(e.code)) e.preventDefault();
});
document.addEventListener("keyup", e => _keys.delete(e.code));

const BASE_SPEED = 0.05;   // units per frame (~3 units/sec at 60fps)
const TURN_SPEED = 0.032;  // radians per frame (~110°/sec at 60fps)

// ── Remote navigation from mobile ─────────────────────────────────────────
let _remoteNav = { move_x:0, move_z:0, move_y:0, turn_x:0, turn_y:0, gyro:false, gyro_yaw:null, gyro_pitch:null, ts:0 };

(async function _pollRemoteNav() {
  while (true) {
    try {
      const r = await fetch("/api/navigate", { cache: "no-store" });
      if (r.ok) _remoteNav = await r.json();
    } catch {}
    await new Promise(res => setTimeout(res, 50));
  }
})();

// Separate poll for reset-view signal from mobile
let _lastResetTs = 0;
(async function _pollResetView() {
  while (true) {
    try {
      const r = await fetch("/api/reset-view", { cache: "no-store" });
      if (r.ok) {
        const { ts } = await r.json();
        if (ts > _lastResetTs) {
          _lastResetTs = ts;
          if (!worldMode) resetCamera(); else flyToWorldOverview();
        }
      }
    } catch {}
    await new Promise(res => setTimeout(res, 100));
  }
})();

// ── Distance-based LOD ────────────────────────────────────────────────────
// Single-scene mode: fades splat scale down when the camera backs far away
// from the one loaded scene (navigating inside it always stays full quality).
//
// Memory Verse (world) mode: LOD is now a hard swap, not a fade — see
// updateFullResLOD() above. At most one scene is ever loaded at full
// gaussian-splat quality at a time (whichever the visitor is near); every
// other scene is just its lightweight point-cloud proxy. So there's nothing
// to scale/fade here — the loaded scene is always the near one, always at
// full quality, and swapping it out for a different one is a load/unload,
// not a gradual visual transition.

const LOD_LERP_IN   = 0.12; // fast restore when approaching
const LOD_LERP_OUT  = 0.025; // slow fade when retreating

let _lodCurrentScale = 1.0;
let _lodFrame        = 0;
let _lodPrevDist     = 0;

function lodScale(dist, near, far, minScale) {
  if (dist <= near) return 1.0;
  if (dist >= far)  return minScale;
  const t = (dist - near) / (far - near);
  return 1.0 - t * (1.0 - minScale);
}

function updateLOD() {
  const cam = viewer?.camera;
  if (!cam) return;
  if (++_lodFrame % 3 !== 0) return;

  const cx = cam.position.x, cy = cam.position.y, cz = cam.position.z;

  if (worldMode) {
    updateFullResLOD(cx, cy, cz);
    return;
  }

  // Single-scene mode always loads at most one mesh (see transitionTo()).
  const mesh = splatMeshes.values().next().value;
  if (!mesh) return; // nothing loaded yet

  const minDist = Math.hypot(cx, cy, cz);
  const targetScale = lodScale(minDist, 10, 32, 0.20);

  const approaching = minDist < _lodPrevDist;
  const lerp = approaching ? LOD_LERP_IN : LOD_LERP_OUT;
  _lodPrevDist = minDist;

  _lodCurrentScale += (targetScale - _lodCurrentScale) * lerp * 3;
  mesh.scale.setScalar(Math.max(0.10, _lodCurrentScale));
}

// Throttled POST of the camera pose (position on the ground plane + yaw) for
// the main-page map's "you are here" marker.
let _lastCamPost = 0;
function postCameraState() {
  const cam = viewer?.camera;
  if (!cam) return;
  const now = performance.now();
  if (now - _lastCamPost < 140) return;
  _lastCamPost = now;
  fetch("/api/camera-state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x: cam.position.x, z: cam.position.z, yaw: _yaw }),
  }).catch(() => {});
}

function flyLoop() {
  requestAnimationFrame(flyLoop);
  updateLOD();

  // Apply remote mobile navigation (joystick / gyro from mobile app)
  {
    const rn = _remoteNav;
    const isRecent = rn.ts > 0 && (Date.now() - rn.ts < 600);
    if (isRecent && viewer) {
      const cam = viewer.camera;
      if (cam) {
        const hasAct = rn.move_x || rn.move_z || rn.move_y || rn.turn_x || rn.turn_y || rn.gyro;
        if (hasAct) {
          const m = cam.matrixWorld.elements;
          const right   = { x: m[0],  y: m[1],  z: m[2]  };
          const camUp   = { x: m[4],  y: m[5],  z: m[6]  };
          const forward = { x: -m[8], y: -m[9], z: -m[10] };
          let dx = 0, dy = 0, dz = 0;
          const add = (v, s) => { dx += v.x*s; dy += v.y*s; dz += v.z*s; };
          if (rn.move_z) add(forward, rn.move_z * BASE_SPEED);
          if (rn.move_x) add(right,   rn.move_x * BASE_SPEED);
          if (rn.move_y) add(camUp,   rn.move_y * BASE_SPEED);
          if (rn.gyro && rn.gyro_yaw !== null) {
            _yaw   = rn.gyro_yaw;
            _pitch = Math.max(-1.30, Math.min(1.30, rn.gyro_pitch ?? 0));
          } else {
            if (rn.turn_x) { _yaw += rn.turn_x * TURN_SPEED; }
            if (rn.turn_y) { _pitch -= rn.turn_y * TURN_SPEED; _pitch = Math.max(-1.30, Math.min(1.30, _pitch)); }
          }
          cam.position.x += dx; cam.position.y += dy; cam.position.z += dz;
          applyFPSCamera();
        }
      }
    }
  }

  // Spatial audio runs every frame so distance-based gain updates
  // continuously — even when the camera is stationary or mid-flyTo.
  updateSpatialAudio();

  // Report camera pose to the server a few times a second so the main-page map
  // can show where the viewer is and which way it's looking.
  postCameraState();

  if (viewer && _keys.size > 0) {
    const cam = viewer.camera;
    if (cam) {
      // Camera axes from world matrix — always current after applyFPSCamera()
      const m = cam.matrixWorld.elements;
      const right   = { x: m[0],  y: m[1],  z: m[2]  };
      const camUp   = { x: m[4],  y: m[5],  z: m[6]  };
      const forward = { x: -m[8], y: -m[9], z: -m[10] };

      const sprint = _keys.has("ShiftLeft") || _keys.has("ShiftRight");
      const speed  = BASE_SPEED * (sprint ? 4 : 1);

      let dx = 0, dy = 0, dz = 0;
      const add = (v, s) => { dx += v.x * s; dy += v.y * s; dz += v.z * s; };

      // W/S + Up/Down: forward/back. A/D: strafe sideways. E/Q: up/down.
      if (_keys.has("KeyW") || _keys.has("ArrowUp"))    add(forward,  speed);
      if (_keys.has("KeyS") || _keys.has("ArrowDown"))  add(forward, -speed);
      if (_keys.has("KeyA"))                            add(right,   -speed);
      if (_keys.has("KeyD"))                            add(right,    speed);
      if (_keys.has("KeyE") || _keys.has("PageUp"))     add(camUp,    speed);
      if (_keys.has("KeyQ") || _keys.has("PageDown"))   add(camUp,   -speed);

      // Side arrow keys turn your head (yaw) instead of strafing
      let turned = false;
      if (_keys.has("ArrowLeft"))  { _yaw -= TURN_SPEED; turned = true; }
      if (_keys.has("ArrowRight")) { _yaw += TURN_SPEED; turned = true; }
      if (turned) {
        // Keep yaw bounded so it never loses precision in a long-running kiosk
        _yaw = ((_yaw + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      }

      if (dx !== 0 || dy !== 0 || dz !== 0 || turned) {
        cam.position.x += dx;
        cam.position.y += dy;
        cam.position.z += dz;
        applyFPSCamera();  // re-apply look direction after position/yaw changes
      }
    }
  }

  // Actually draw the frame — Spark hooks into this via SparkRenderer's
  // onBeforeRender, so a normal renderer.render() call is all it needs.
  if (viewer) viewer.renderer.render(viewer.threeScene, viewer.camera);
}

flyLoop();

// ── Keyboard nav hint: small tab, expands/closes on click ─────────────────



// ── Boot ──────────────────────────────────────────────────────────────────

poll();
const _pollId      = setInterval(poll, POLL_INTERVAL);
const _statusId    = setInterval(pollStatus, STATUS_INTERVAL);
const _selectionId = setInterval(pollSelection, 2_000);  // snappier than POLL_INTERVAL — this is a direct user action
const _worldSelectionId = setInterval(pollWorldSelection, 2_000);
const _posSyncId   = setInterval(syncServerPositions, 600); // live map-drag → viewer moves

// A dev-server hot-reload of this module without a full page reload would
// otherwise leave the old poll/select/status intervals running alongside
// the new ones — multiple overlapping goToIndex() calls racing each other
// is exactly what produces "wrong scene plays, audio doesn't match, jumps
// around fast". Clearing them on dispose guarantees only one set ever runs.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    clearInterval(_pollId);
    clearInterval(_statusId);
    clearInterval(_selectionId);
    clearInterval(_worldSelectionId);
    clearInterval(_posSyncId);
  });
}
