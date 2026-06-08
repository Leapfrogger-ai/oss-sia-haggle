const path = require("path");
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
const W = 13.33, H = 7.5;

const INK = "14142B", LIGHT = "F7F8FA", WHITE = "FFFFFF";
const GREEN = "0F9D58", BLUE = "4285F4", MUTE = "6B7280", ICE = "9FB3C8";
const HF = "Trebuchet MS", BF = "Calibri";
const img = (f) => path.resolve(__dirname, "..", f);

// ---------- Slide 1: title (dark) ----------
let s = p.addSlide();
s.background = { color: INK };
s.addText("HAGGLE", { x: 0.8, y: 1.9, w: 11, h: 1.1, fontFace: HF, fontSize: 60, bold: true, color: WHITE, charSpacing: 2 });
s.addText("A self-improving negotiation-outcome agent", { x: 0.85, y: 3.0, w: 11.5, h: 0.6, fontFace: HF, fontSize: 24, color: ICE });
s.addText(
  [{ text: "SIA improves an agent on ", options: { color: "D7DEE8" } },
   { text: "two axes", options: { color: GREEN, bold: true } },
   { text: " — its ", options: { color: "D7DEE8" } },
   { text: "code/prompt", options: { color: GREEN, bold: true } },
   { text: " AND its ", options: { color: "D7DEE8" } },
   { text: "model weights", options: { color: BLUE, bold: true } },
   { text: " — on real B2B negotiation data, entirely on Nebius.", options: { color: "D7DEE8" } }],
  { x: 0.85, y: 3.9, w: 11.2, h: 1.0, fontFace: BF, fontSize: 18, lineSpacingMultiple: 1.15 });
// two chips
s.addShape(p.ShapeType.roundRect, { x: 0.85, y: 5.0, w: 3.1, h: 0.55, fill: { color: "1E2447" }, line: { color: GREEN, width: 1 }, rectRadius: 0.08 });
s.addText("AXIS 1 · the harness", { x: 0.85, y: 5.0, w: 3.1, h: 0.55, align: "center", fontFace: BF, fontSize: 13, color: GREEN, bold: true });
s.addShape(p.ShapeType.roundRect, { x: 4.1, y: 5.0, w: 3.1, h: 0.55, fill: { color: "1E2447" }, line: { color: BLUE, width: 1 }, rectRadius: 0.08 });
s.addText("AXIS 2 · the weights", { x: 4.1, y: 5.0, w: 3.1, h: 0.55, align: "center", fontFace: BF, fontSize: 13, color: BLUE, bold: true });
s.addText("SIA × Nebius Token Factory  ·  one API key for inference + GPU fine-tuning", { x: 0.85, y: 6.7, w: 11.5, h: 0.4, fontFace: BF, fontSize: 12, color: MUTE });

// ---------- Slide 2: two axes (light) ----------
s = p.addSlide();
s.background = { color: LIGHT };
s.addText("Two axes of self-improvement — one task, one API key", { x: 0.6, y: 0.35, w: 12.1, h: 0.7, fontFace: HF, fontSize: 30, bold: true, color: INK });
// card helper
function card(x, tag, tagColor, big, sub) {
  s.addShape(p.ShapeType.roundRect, { x, y: 1.35, w: 5.95, h: 1.7, fill: { color: WHITE }, line: { color: "E3E7EC", width: 1 }, rectRadius: 0.06, shadow: { type: "outer", blur: 6, offset: 2, color: "D9DEE5", opacity: 0.5 } });
  s.addShape(p.ShapeType.rect, { x, y: 1.35, w: 0.12, h: 1.7, fill: { color: tagColor } });
  s.addText(tag, { x: x + 0.3, y: 1.5, w: 5.5, h: 0.35, fontFace: BF, fontSize: 13, bold: true, color: tagColor, margin: 0 });
  s.addText(big, { x: x + 0.28, y: 1.85, w: 5.6, h: 0.7, fontFace: HF, fontSize: 32, bold: true, color: INK, margin: 0 });
  s.addText(sub, { x: x + 0.3, y: 2.55, w: 5.5, h: 0.45, fontFace: BF, fontSize: 12.5, color: MUTE, margin: 0 });
}
card(0.6, "AXIS 1 · HARNESS  (code / prompt)", GREEN, "44%  →  65%", "SIA rewrote the agent each generation — no human in the loop.");
card(6.78, "AXIS 2 · WEIGHTS  (LoRA fine-tune)", BLUE, "loss 0.69 → 0.375", "Llama-3.3-70B tuned on Nebius GPUs — same API key.");
s.addImage({ path: img("3_two_axes_summary.png"), x: 1.66, y: 3.35, w: 10.0, h: 3.7 });

// ---------- Slide 3: what the loss means (light) ----------
s = p.addSlide();
s.background = { color: LIGHT };
s.addText("What the LoRA loss means — and the implication", { x: 0.6, y: 0.35, w: 12.1, h: 0.7, fontFace: HF, fontSize: 30, bold: true, color: INK });
s.addImage({ path: img("2_weights_losscurve.png"), x: 0.6, y: 1.45, w: 6.3, h: 3.94 });
const bullets = [
  { text: "Per-token cross-entropy on the predicted price — how “surprised” the model is by the correct settled price. Lower = more probability on the right answer.", options: { bullet: { code: "2022" }, color: "20242E" } },
  { text: "Held-out (validation) loss fell 0.69 → 0.375 (perplexity ~2.0 → 1.45) right alongside train loss, with a small final gap.", options: { bullet: { code: "2022" }, color: "20242E" } },
  { text: "→ it GENERALIZED, didn’t memorize: the LoRA taught Llama-70B the task’s price priors & output format in a way that transfers to unseen negotiations.", options: { bullet: { code: "2022" }, color: "20242E", bold: true } },
];
s.addText(bullets, { x: 7.15, y: 1.55, w: 5.55, h: 2.7, fontFace: BF, fontSize: 14.5, lineSpacingMultiple: 1.08, paraSpaceAfter: 10, valign: "top" });
// implication box
s.addShape(p.ShapeType.roundRect, { x: 7.15, y: 4.5, w: 5.55, h: 2.35, fill: { color: "ECF7F0" }, line: { color: GREEN, width: 1.25 }, rectRadius: 0.06 });
s.addText("IMPLICATION", { x: 7.4, y: 4.65, w: 5, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: GREEN, margin: 0 });
s.addText(
  [{ text: "The model itself got better at the task — a second, composable axis of self-improvement beyond prompt/code. ", options: {} },
   { text: "Lower held-out loss ⇒ higher expected served ±10% accuracy", options: { bold: true } },
   { text: " (live serving is one Nebius account toggle away).", options: {} }],
  { x: 7.4, y: 5.05, w: 5.05, h: 1.65, fontFace: BF, fontSize: 14, color: "1B3B2A", lineSpacingMultiple: 1.12, valign: "top", margin: 0 });

// ---------- Slide 4: why it matters (dark) ----------
s = p.addSlide();
s.background = { color: INK };
s.addText("Why it matters", { x: 0.8, y: 0.5, w: 11, h: 0.8, fontFace: HF, fontSize: 34, bold: true, color: WHITE });
const pts = [
  ["Real B2B pain", GREEN, "Deal-outcome prediction & forecasting — only 45% of sales leaders trust their forecasts. Negotiation is the core sales skill. Real data, un-gameable ±10% metric."],
  ["Two composable levers", BLUE, "SIA improves the scaffold (code/prompt); fine-tuning improves the weights. Orthogonal and stackable on the same task."],
  ["Open models, one Nebius key", GREEN, "Inference AND GPU fine-tuning through a single API key — no proprietary frontier model needed for the target. Reproducible and cheap."],
  ["Honest status", BLUE, "Harness curve + LoRA training curve are live; served base-vs-tuned accuracy is one support toggle (LoRA inference) away from the full 2×2."],
];
let y = 1.7;
for (const [h, c, body] of pts) {
  s.addShape(p.ShapeType.ellipse, { x: 0.85, y: y + 0.06, w: 0.28, h: 0.28, fill: { color: c } });
  s.addText(h, { x: 1.35, y: y, w: 4.4, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: WHITE, margin: 0 });
  s.addText(body, { x: 5.7, y: y - 0.05, w: 6.9, h: 1.1, fontFace: BF, fontSize: 14, color: "C7CEDA", lineSpacingMultiple: 1.05, valign: "top", margin: 0 });
  y += 1.28;
}

p.writeFile({ fileName: path.resolve(__dirname, "..", "Haggle_SIA_Nebius.pptx") }).then(f => console.log("wrote", f));
