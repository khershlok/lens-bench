// Lens Bench: traces light through a thick lens using Snell's law at each surface.
// World units are millimetres, with y pointing up and the lens centred on the origin.

type Vec = { x: number; y: number };
type LensKind = "convex" | "concave";
type SourceMode = "parallel" | "point" | "single";

const vec = (x: number, y: number): Vec => ({ x, y });
const add = (a: Vec, b: Vec): Vec => vec(a.x + b.x, a.y + b.y);
const sub = (a: Vec, b: Vec): Vec => vec(a.x - b.x, a.y - b.y);
const scale = (a: Vec, s: number): Vec => vec(a.x * s, a.y * s);
const dot = (a: Vec, b: Vec): number => a.x * b.x + a.y * b.y;
const length = (a: Vec): number => Math.hypot(a.x, a.y);
const unit = (a: Vec): Vec => scale(a, 1 / length(a));
const deg = (rad: number): number => (rad * 180) / Math.PI;
const rad = (d: number): number => (d * Math.PI) / 180;

interface ArcSurface {
  kind: "arc";
  name: string;
  c: Vec;
  r: number;
  capSign: number; // which side of the circle's centre the lens face sits on
}

interface EdgeSurface {
  kind: "edge";
  y: number;
  x0: number;
  x1: number;
}

type Surface = ArcSurface | EdgeSurface;

interface Lens {
  kind: LensKind;
  n: number;
  R: number;
  h: number;
  t: number;
  surfaces: Surface[];
  outline: Vec[];
  focal: number;
}

interface HitEvent {
  p: Vec;
  normal: Vec; // points back toward the side the ray came from
  dirIn: Vec;
  dirOut: Vec;
  thetaI: number;
  thetaT: number;
  n1: number;
  n2: number;
  tir: boolean;
  surface: string;
}

interface Segment {
  a: Vec;
  b: Vec;
  n: number;
}

interface TraceResult {
  segments: Segment[];
  events: HitEvent[];
  absorbed: boolean;
  exitPoint: Vec | null;
  exitDir: Vec | null;
}

const APERTURE = 40;
const EDGE = 4;
const EPS = 1e-7;

function buildLens(kind: LensKind, R: number, n: number): Lens {
  const h = APERTURE;
  const sag = R - Math.sqrt(R * R - h * h);
  let t: number;
  let front: ArcSurface;
  let back: ArcSurface;

  if (kind === "convex") {
    t = EDGE + 2 * sag;
    front = { kind: "arc", name: "front", c: vec(-t / 2 + R, 0), r: R, capSign: -1 };
    back = { kind: "arc", name: "back", c: vec(t / 2 - R, 0), r: R, capSign: 1 };
  } else {
    t = EDGE;
    front = { kind: "arc", name: "front", c: vec(-t / 2 - R, 0), r: R, capSign: 1 };
    back = { kind: "arc", name: "back", c: vec(t / 2 + R, 0), r: R, capSign: -1 };
  }

  const faceX = (s: ArcSurface, y: number): number =>
    s.c.x + s.capSign * Math.sqrt(s.r * s.r - y * y);

  const outline: Vec[] = [];
  const steps = 48;
  for (let i = 0; i <= steps; i++) {
    const y = -h + (2 * h * i) / steps;
    outline.push(vec(faceX(front, y), y));
  }
  for (let i = 0; i <= steps; i++) {
    const y = h - (2 * h * i) / steps;
    outline.push(vec(faceX(back, y), y));
  }

  const edgeX0 = faceX(front, h);
  const edgeX1 = faceX(back, h);
  const surfaces: Surface[] = [
    front,
    back,
    { kind: "edge", y: h, x0: Math.min(edgeX0, edgeX1), x1: Math.max(edgeX0, edgeX1) },
    { kind: "edge", y: -h, x0: Math.min(edgeX0, edgeX1), x1: Math.max(edgeX0, edgeX1) },
  ];

  // Lensmaker's equation for a thick lens with R1 = ±R, R2 = ∓R.
  const R1 = kind === "convex" ? R : -R;
  const R2 = -R1;
  const power = (n - 1) * (1 / R1 - 1 / R2 + ((n - 1) * t) / (n * R1 * R2));
  const focal = power === 0 ? Infinity : 1 / power;

  return { kind, n, R, h, t, surfaces, outline, focal };
}

function intersect(o: Vec, d: Vec, s: Surface, h: number): number | null {
  if (s.kind === "edge") {
    if (Math.abs(d.y) < 1e-12) return null;
    const k = (s.y - o.y) / d.y;
    if (k <= EPS) return null;
    const x = o.x + d.x * k;
    return x >= s.x0 - 1e-9 && x <= s.x1 + 1e-9 ? k : null;
  }
  const f = sub(o, s.c);
  const b = dot(f, d);
  const disc = b * b - (dot(f, f) - s.r * s.r);
  if (disc < 0) return null;
  const root = Math.sqrt(disc);
  let best: number | null = null;
  for (const k of [-b - root, -b + root]) {
    if (k <= EPS) continue;
    const p = add(o, scale(d, k));
    if (Math.abs(p.y) > h + 1e-9) continue;
    if (Math.sign(p.x - s.c.x) !== s.capSign) continue;
    if (best === null || k < best) best = k;
  }
  return best;
}

function trace(origin: Vec, direction: Vec, lens: Lens, n: number, reach: number): TraceResult {
  const segments: Segment[] = [];
  const events: HitEvent[] = [];
  let o = origin;
  let d = unit(direction);
  let inside = false;

  for (let bounce = 0; bounce < 10; bounce++) {
    let nearest = Infinity;
    let hitSurface: Surface | null = null;
    for (const s of lens.surfaces) {
      const k = intersect(o, d, s, lens.h);
      if (k !== null && k < nearest) {
        nearest = k;
        hitSurface = s;
      }
    }

    const medium = inside ? n : 1;
    if (!hitSurface) {
      segments.push({ a: o, b: add(o, scale(d, reach)), n: medium });
      return { segments, events, absorbed: false, exitPoint: o, exitDir: d };
    }

    const p = add(o, scale(d, nearest));
    segments.push({ a: o, b: p, n: medium });

    if (hitSurface.kind === "edge") {
      return { segments, events, absorbed: true, exitPoint: null, exitDir: null };
    }

    let normal = unit(sub(p, hitSurface.c));
    if (dot(normal, d) > 0) normal = scale(normal, -1);
    const n1 = inside ? n : 1;
    const n2 = inside ? 1 : n;
    const eta = n1 / n2;
    const cosI = -dot(normal, d);
    const k = 1 - eta * eta * (1 - cosI * cosI);
    const thetaI = Math.acos(Math.min(1, Math.max(-1, cosI)));

    let out: Vec;
    let tir = false;
    let thetaT = NaN;
    if (k < 0) {
      tir = true;
      out = unit(sub(d, scale(normal, 2 * dot(d, normal))));
    } else {
      out = unit(add(scale(d, eta), scale(normal, eta * cosI - Math.sqrt(k))));
      thetaT = Math.asin(Math.min(1, eta * Math.sin(thetaI)));
      inside = !inside;
    }

    events.push({ p, normal, dirIn: d, dirOut: out, thetaI, thetaT, n1, n2, tir, surface: hitSurface.name });
    o = p;
    d = out;
  }
  return { segments, events, absorbed: true, exitPoint: null, exitDir: null };
}

// ---------- state ----------

interface State {
  kind: LensKind;
  mode: SourceMode;
  n: number;
  R: number;
  angle: number;
  count: number;
  speed: number;
  sweep: boolean;
  dispersion: boolean;
  normals: boolean;
  virtual: boolean;
  singleY: number;
  source: Vec;
}

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state: State = {
  kind: "convex",
  mode: "parallel",
  n: 1.52,
  R: 80,
  angle: 0,
  count: 11,
  speed: reduceMotion ? 0 : 1,
  sweep: !reduceMotion,
  dispersion: false,
  normals: true,
  virtual: false,
  singleY: 22,
  source: vec(-120, 12),
};

const hashTokens = location.hash.replace("#", "").split("-");
for (const token of hashTokens) {
  if (token === "convex" || token === "concave") state.kind = token;
  if (token === "parallel" || token === "point" || token === "single") state.mode = token;
}
state.virtual = state.kind === "concave";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const canvas = $<HTMLCanvasElement>("bench");
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
const readout = $<HTMLDivElement>("readout");
const hint = $<HTMLParagraphElement>("hint");

const COLORS = {
  grid: "rgba(150, 178, 214, 0.07)",
  gridMajor: "rgba(150, 178, 214, 0.14)",
  axis: "rgba(186, 206, 232, 0.35)",
  glassFill: "rgba(120, 214, 230, 0.13)",
  glassEdge: "rgba(160, 232, 244, 0.75)",
  ray: [255, 196, 92] as [number, number, number],
  normal: "rgba(214, 226, 244, 0.55)",
  angleIn: "#ffc45c",
  angleOut: "#7fe0f0",
  label: "rgba(222, 232, 246, 0.85)",
  focal: "#f28fb4",
  source: "#ffe2a6",
};

const SPECTRUM: { rgb: [number, number, number]; dn: number }[] = [
  { rgb: [255, 84, 84], dn: -0.014 },
  { rgb: [96, 255, 128], dn: 0 },
  { rgb: [96, 140, 255], dn: 0.022 },
];

let view = { w: 0, h: 0, ox: 0, oy: 0, s: 1, dpr: 1 };

function resize(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const w = rect.width;
  const h = rect.height;
  const s = Math.min(w / 330, h / 150);
  view = { w, h, ox: w * 0.47, oy: h / 2, s, dpr };
}

const toScreen = (p: Vec): Vec => vec(view.ox + p.x * view.s, view.oy - p.y * view.s);
const toWorld = (x: number, y: number): Vec => vec((x - view.ox) / view.s, (view.oy - y) / view.s);
const worldLeft = (): number => -view.ox / view.s;
const worldRight = (): number => (view.w - view.ox) / view.s;

// ---------- rays ----------

interface RaySpec {
  origin: Vec;
  dir: Vec;
}

function raySpecs(lens: Lens): RaySpec[] {
  const startX = worldLeft() - 5;
  const specs: RaySpec[] = [];
  const reachHeight = lens.h * 0.94;

  if (state.mode === "parallel") {
    const a = rad(state.angle);
    const dir = vec(Math.cos(a), Math.sin(a));
    for (let i = 0; i < state.count; i++) {
      const yAtLens = state.count === 1 ? 0 : -reachHeight + (2 * reachHeight * i) / (state.count - 1);
      specs.push({ origin: vec(startX, yAtLens - Math.tan(a) * (0 - startX)), dir });
    }
  } else if (state.mode === "point") {
    const frontX = -lens.t / 2;
    for (let i = 0; i < state.count; i++) {
      const yAtLens = state.count === 1 ? 0 : -reachHeight + (2 * reachHeight * i) / (state.count - 1);
      specs.push({ origin: state.source, dir: unit(sub(vec(frontX, yAtLens), state.source)) });
    }
  } else {
    const a = rad(state.angle);
    const dir = vec(Math.cos(a), Math.sin(a));
    specs.push({ origin: vec(startX, state.singleY - Math.tan(a) * (0 - startX)), dir });
  }
  return specs;
}

// ---------- drawing ----------

function drawGrid(): void {
  const left = Math.floor(worldLeft() / 10) * 10;
  const right = worldRight();
  const top = view.oy / view.s;
  ctx.lineWidth = 1;
  for (let x = left; x <= right; x += 10) {
    const sx = toScreen(vec(x, 0)).x;
    ctx.strokeStyle = x % 50 === 0 ? COLORS.gridMajor : COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, view.h);
    ctx.stroke();
  }
  for (let y = -Math.floor(top / 10) * 10; y <= top; y += 10) {
    const sy = toScreen(vec(0, y)).y;
    ctx.strokeStyle = y % 50 === 0 ? COLORS.gridMajor : COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(0, sy);
    ctx.lineTo(view.w, sy);
    ctx.stroke();
  }

  ctx.strokeStyle = COLORS.axis;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(0, view.oy);
  ctx.lineTo(view.w, view.oy);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(186, 206, 232, 0.5)";
  ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.fillText("optical axis", 8, view.oy - 6);
  const barStart = toScreen(vec(worldRight() - 62, -64));
  const barEnd = toScreen(vec(worldRight() - 12, -64));
  if (barStart.y < view.h - 6) {
    ctx.strokeStyle = "rgba(186, 206, 232, 0.6)";
    ctx.beginPath();
    ctx.moveTo(barStart.x, barStart.y);
    ctx.lineTo(barEnd.x, barEnd.y);
    ctx.moveTo(barStart.x, barStart.y - 4);
    ctx.lineTo(barStart.x, barStart.y + 4);
    ctx.moveTo(barEnd.x, barEnd.y - 4);
    ctx.lineTo(barEnd.x, barEnd.y + 4);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText("50 mm", (barStart.x + barEnd.x) / 2, barStart.y - 7);
  }
}

function drawLens(lens: Lens): void {
  ctx.beginPath();
  lens.outline.forEach((p, i) => {
    const s = toScreen(p);
    if (i === 0) ctx.moveTo(s.x, s.y);
    else ctx.lineTo(s.x, s.y);
  });
  ctx.closePath();
  const g = ctx.createLinearGradient(0, toScreen(vec(0, lens.h)).y, 0, toScreen(vec(0, -lens.h)).y);
  g.addColorStop(0, "rgba(150, 228, 240, 0.22)");
  g.addColorStop(0.5, COLORS.glassFill);
  g.addColorStop(1, "rgba(150, 228, 240, 0.2)");
  ctx.fillStyle = g;
  ctx.fill();
  ctx.strokeStyle = COLORS.glassEdge;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Ground (blackened) edges absorb stray rays, as on a real lens.
  ctx.strokeStyle = "rgba(10, 14, 22, 0.9)";
  ctx.lineWidth = 3;
  for (const s of lens.surfaces) {
    if (s.kind !== "edge") continue;
    const a = toScreen(vec(s.x0, s.y));
    const b = toScreen(vec(s.x1, s.y));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawFocal(lens: Lens): void {
  if (!isFinite(lens.focal) || Math.abs(lens.focal) > 400) return;
  ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
  ctx.textAlign = "center";
  for (const [x, label] of [
    [lens.focal, "F′"],
    [-lens.focal, "F"],
  ] as [number, string][]) {
    const p = toScreen(vec(x, 0));
    if (p.x < 0 || p.x > view.w) continue;
    ctx.fillStyle = COLORS.focal;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(label, p.x, p.y + 18);
  }
}

function strokePath(points: Vec[]): void {
  ctx.beginPath();
  points.forEach((p, i) => {
    const s = toScreen(p);
    if (i === 0) ctx.moveTo(s.x, s.y);
    else ctx.lineTo(s.x, s.y);
  });
  ctx.stroke();
}

const PULSE_PERIOD = 26; // mm of optical path between pulses

function drawRay(result: TraceResult, rgb: [number, number, number], time: number, faint: boolean): void {
  const [r, g, b] = rgb;
  const baseAlpha = faint ? 0.22 : 0.42;
  ctx.lineCap = "round";
  ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${baseAlpha})`;
  ctx.lineWidth = 1.4;
  for (const seg of result.segments) strokePath([seg.a, seg.b]);

  // Pulses travel at c/n, so they bunch together inside the glass.
  let opticalPath = 0;
  const travelled = time * 60;
  for (const seg of result.segments) {
    const segLen = length(sub(seg.b, seg.a));
    const period = (PULSE_PERIOD / seg.n) * view.s;
    const phase = (((travelled - opticalPath) / seg.n) * view.s) % period;
    const dash = Math.max(2, period * 0.18);
    ctx.setLineDash([dash, period - dash]);
    ctx.lineDashOffset = -phase;
    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.28)`;
    ctx.lineWidth = 7;
    strokePath([seg.a, seg.b]);
    ctx.strokeStyle = `rgba(${Math.min(255, r + 40)}, ${Math.min(255, g + 40)}, ${Math.min(255, b + 40)}, 0.95)`;
    ctx.lineWidth = 2.6;
    strokePath([seg.a, seg.b]);
    opticalPath += segLen * seg.n;
  }
  ctx.setLineDash([]);
  ctx.lineDashOffset = 0;
}

function drawVirtual(result: TraceResult, rgb: [number, number, number]): void {
  if (!result.exitPoint || !result.exitDir || result.events.length < 2) return;
  const back = add(result.exitPoint, scale(result.exitDir, -260));
  ctx.setLineDash([3, 5]);
  ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.32)`;
  ctx.lineWidth = 1;
  strokePath([result.exitPoint, back]);
  ctx.setLineDash([]);
}

const screenAngle = (d: Vec): number => Math.atan2(-d.y, d.x);

function drawAngle(p: Vec, u1: Vec, u2: Vec, radius: number, color: string, text: string): void {
  const c = toScreen(p);
  const a1 = screenAngle(u1);
  let diff = screenAngle(u2) - a1;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  if (Math.abs(diff) < 0.01) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(c.x, c.y, radius, a1, a1 + diff, diff < 0);
  ctx.stroke();
  const mid = a1 + diff / 2;
  ctx.fillStyle = color;
  ctx.font = "600 11px 'JetBrains Mono', ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, c.x + Math.cos(mid) * (radius + 16), c.y + Math.sin(mid) * (radius + 16));
  ctx.textBaseline = "alphabetic";
}

function drawNormals(result: TraceResult): void {
  const len = 16;
  for (const ev of result.events) {
    const a = add(ev.p, scale(ev.normal, len));
    const b = add(ev.p, scale(ev.normal, -len));
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = COLORS.normal;
    ctx.lineWidth = 1;
    strokePath([a, b]);
    ctx.setLineDash([]);
    const radius = Math.max(16, 9 * view.s);
    drawAngle(ev.p, ev.normal, scale(ev.dirIn, -1), radius, COLORS.angleIn, `${deg(ev.thetaI).toFixed(1)}°`);
    if (!ev.tir) {
      drawAngle(ev.p, scale(ev.normal, -1), ev.dirOut, radius, COLORS.angleOut, `${deg(ev.thetaT).toFixed(1)}°`);
    }
  }
}

function drawSource(): void {
  const p = toScreen(state.source);
  const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 22);
  glow.addColorStop(0, "rgba(255, 226, 166, 0.55)");
  glow.addColorStop(1, "rgba(255, 226, 166, 0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 22, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = COLORS.source;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = COLORS.label;
  ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText("drag me", p.x, p.y - 14);
}

// ---------- readout ----------

function fmt(n: number, digits = 1): string {
  return Number.isFinite(n) ? n.toFixed(digits) : "∞";
}

function renderReadout(lens: Lens, single: TraceResult | null): void {
  const kindText = lens.kind === "convex" ? "Converging (convex)" : "Diverging (concave)";
  const f = lens.focal;
  let html = `
    <div class="stat"><span>Lens</span><b>${kindText}</b></div>
    <div class="stat"><span>Focal length</span><b>${fmt(f)} mm</b></div>
    <div class="stat"><span>Centre thickness</span><b>${fmt(lens.t)} mm</b></div>`;

  if (single) {
    const evs = single.events;
    const rows = evs
      .map((ev, i) => {
        const where = ev.surface === "front" ? "Front face" : "Back face";
        const media = ev.n1 === 1 ? "air → glass" : "glass → air";
        const out = ev.tir ? `<em>total internal reflection</em>` : `${fmt(deg(ev.thetaT))}°`;
        return `<tr><td>${i + 1}. ${where}<small>${media}</small></td><td class="in">${fmt(deg(ev.thetaI))}°</td><td class="out">${out}</td></tr>`;
      })
      .join("");
    let note = "";
    if (single.absorbed) note = `<p class="note">This ray hits the lens's blackened edge and is absorbed.</p>`;
    else if (evs.length === 0) note = `<p class="note">This ray misses the lens. Drag it back toward the glass.</p>`;
    else if (single.exitDir) {
      const startAngle = state.angle;
      const endAngle = deg(Math.atan2(single.exitDir.y, single.exitDir.x));
      note = `<p class="note">Total bend: <b>${fmt(Math.abs(endAngle - startAngle))}°</b> ${
        endAngle - startAngle < 0 ? "downward" : "upward"
      }. Rays that enter farther from the centre meet the surface at a steeper angle and bend more.</p>`;
    }
    html += `
      <table class="snell">
        <thead><tr><th>Surface</th><th class="in">θ in</th><th class="out">θ out</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="3">No surfaces hit</td></tr>`}</tbody>
      </table>
      <p class="formula">n₁ sin θ₁ = n₂ sin θ₂ &nbsp;·&nbsp; n = ${state.n.toFixed(2)}</p>
      ${note}`;
  } else if (state.mode === "parallel") {
    html += `<p class="note">${
      lens.kind === "convex"
        ? "Parallel rays bend toward the axis and cross near the focal point F′."
        : "Parallel rays spread out. Trace them backward (dotted) and they appear to come from the virtual focus F′ on the left."
    }</p>`;
  } else {
    html += `<p class="note">${
      lens.kind === "convex"
        ? "Move the source past F to see the rays converge into a real image. Inside F, they still diverge."
        : "A concave lens always spreads light from a point source, so it can only form a virtual image."
    }</p>`;
  }
  readout.innerHTML = html;
}

// ---------- loop ----------

let time = 0;
let last = performance.now();
let lastReadoutKey = "";

function frame(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  time += dt * state.speed;
  if (state.mode === "single" && state.sweep && state.speed > 0) {
    state.singleY = Math.sin(time * 0.55) * APERTURE * 0.92;
    syncSingleSlider();
  }

  const lens = buildLens(state.kind, state.R, state.n);
  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  ctx.fillStyle = "#0b111c";
  ctx.fillRect(0, 0, view.w, view.h);
  drawGrid();
  drawLens(lens);
  drawFocal(lens);

  const reach = 600;
  const specs = raySpecs(lens);
  const bands = state.dispersion ? SPECTRUM : [{ rgb: COLORS.ray, dn: 0 }];
  let singleResult: TraceResult | null = null;

  ctx.globalCompositeOperation = "lighter";
  for (const band of bands) {
    const n = Math.max(1, state.n + band.dn);
    for (const spec of specs) {
      const result = trace(spec.origin, spec.dir, lens, n, reach);
      if (state.virtual) drawVirtual(result, band.rgb);
      drawRay(result, band.rgb, time, false);
      if (state.mode === "single" && band.dn === 0) singleResult = result;
    }
  }
  ctx.globalCompositeOperation = "source-over";

  if (state.mode === "single") {
    if (!singleResult) singleResult = trace(specs[0].origin, specs[0].dir, lens, state.n, reach);
    if (state.normals) drawNormals(singleResult);
  }
  if (state.mode === "point") drawSource();

  const key = JSON.stringify([
    state.kind, state.mode, state.n, state.R, state.angle,
    state.mode === "single" ? Math.round(state.singleY * 4) : 0,
  ]);
  if (key !== lastReadoutKey) {
    lastReadoutKey = key;
    renderReadout(lens, state.mode === "single" ? singleResult : null);
  }
  requestAnimationFrame(frame);
}

// ---------- controls ----------

function bindSegmented(id: string, onPick: (value: string) => void, current: string): void {
  const group = $<HTMLDivElement>(id);
  const buttons = Array.from(group.querySelectorAll<HTMLButtonElement>("button"));
  const sync = (value: string) =>
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === value)));
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      onPick(b.dataset.value as string);
      sync(b.dataset.value as string);
    }),
  );
  sync(current);
}

function bindRange(id: string, format: (v: number) => string, onInput: (v: number) => void, initial: number): HTMLInputElement {
  const input = $<HTMLInputElement>(id);
  const out = $<HTMLOutputElement>(`${id}-out`);
  input.value = String(initial);
  out.textContent = format(initial);
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    out.textContent = format(v);
    onInput(v);
  });
  return input;
}

function bindCheck(id: string, onChange: (v: boolean) => void, initial: boolean): HTMLInputElement {
  const input = $<HTMLInputElement>(id);
  input.checked = initial;
  input.addEventListener("change", () => onChange(input.checked));
  return input;
}

let singleSlider: HTMLInputElement;
const singleFormat = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(0)} mm`;
function syncSingleSlider(): void {
  singleSlider.value = String(state.singleY);
  $<HTMLOutputElement>("height-out").textContent = singleFormat(state.singleY);
}

function updateVisibility(): void {
  document.querySelectorAll<HTMLElement>("[data-modes]").forEach((el) => {
    el.hidden = !(el.dataset.modes as string).split(" ").includes(state.mode);
  });
  const hints: Record<SourceMode, string> = {
    parallel: "A beam of parallel rays, like sunlight. Tilt it with the beam angle slider.",
    point: "Light spreading from one point. Drag the glowing dot to move it.",
    single: "One ray with its surface normals and angles. Drag up or down on the bench to aim it.",
  };
  hint.textContent = hints[state.mode];
}

let virtualBox: HTMLInputElement;

bindSegmented("kind", (v) => {
  state.kind = v as LensKind;
  state.virtual = state.kind === "concave";
  virtualBox.checked = state.virtual;
}, state.kind);

bindSegmented("mode", (v) => {
  state.mode = v as SourceMode;
  updateVisibility();
}, state.mode);

bindRange("index", (v) => v.toFixed(2), (v) => (state.n = v), state.n);
bindRange("radius", (v) => `${v.toFixed(0)} mm`, (v) => (state.R = v), state.R);
bindRange("angle", (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}°`, (v) => (state.angle = v), state.angle);
bindRange("count", (v) => v.toFixed(0), (v) => (state.count = v), state.count);
bindRange("speed", (v) => (v === 0 ? "paused" : `${v.toFixed(1)}×`), (v) => (state.speed = v), state.speed);
singleSlider = bindRange("height", singleFormat, (v) => {
  state.singleY = v;
  state.sweep = false;
  sweepBox.checked = false;
}, state.singleY);

const sweepBox = bindCheck("sweep", (v) => (state.sweep = v), state.sweep);
bindCheck("dispersion", (v) => (state.dispersion = v), state.dispersion);
bindCheck("normals", (v) => (state.normals = v), state.normals);
virtualBox = bindCheck("virtual", (v) => (state.virtual = v), state.virtual);

document.querySelectorAll<HTMLButtonElement>("[data-n]").forEach((b) =>
  b.addEventListener("click", () => {
    const v = parseFloat(b.dataset.n as string);
    state.n = v;
    $<HTMLInputElement>("index").value = String(v);
    $<HTMLOutputElement>("index-out").textContent = v.toFixed(2);
  }),
);

// Dragging on the bench aims the single ray or moves the point source.
let dragging = false;
function handlePointer(e: PointerEvent): void {
  const rect = canvas.getBoundingClientRect();
  const w = toWorld(e.clientX - rect.left, e.clientY - rect.top);
  const lens = buildLens(state.kind, state.R, state.n);
  if (state.mode === "point") {
    state.source = vec(Math.min(w.x, -lens.t / 2 - 8 - (lens.kind === "concave" ? lens.R - Math.sqrt(lens.R ** 2 - lens.h ** 2) : 0)), Math.max(-70, Math.min(70, w.y)));
  } else if (state.mode === "single") {
    state.singleY = Math.max(-60, Math.min(60, w.y));
    state.sweep = false;
    sweepBox.checked = false;
    syncSingleSlider();
  }
}
canvas.addEventListener("pointerdown", (e) => {
  if (state.mode === "parallel") return;
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
  handlePointer(e);
});
canvas.addEventListener("pointermove", (e) => dragging && handlePointer(e));
canvas.addEventListener("pointerup", () => (dragging = false));
canvas.addEventListener("pointercancel", () => (dragging = false));

window.addEventListener("resize", resize);
resize();
updateVisibility();
requestAnimationFrame(frame);
