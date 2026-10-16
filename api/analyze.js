// Vercel serverless function: POST /api/analyze
// Receives the multipart FormData that scan.html sends:
//   image      - the captured frame (JPEG or PNG)
//   um_per_px  - calibration, optional (µm per pixel)
//   width, height - the original frame size in pixels, used to turn the model's
//                   0-1 box coordinates into pixel boxes on that frame
// Returns: {"findings":[{"label","confidence","box":[x,y,w,h]}],"width_um","density_per_mm","note"}
//
// The API key comes only from process.env.ANTHROPIC_API_KEY (set in Vercel).
import Anthropic from "@anthropic-ai/sdk";

const MODEL = "claude-sonnet-5-5";
const MAX_IMAGE_BYTES = 4 * 1024 * 1024; // Vercel caps request bodies at 4.5 MB
const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

const SYSTEM = `You review nailfold capillaroscopy images for a student research prototype called Captrail.
Your output is shown to a clinician as an experimental screening aid. It is never a diagnosis.

Look for these capillary patterns and only report ones you can actually see:
- enlarged or giant capillary loops
- microhemorrhages
- avascular areas or capillary dropout (reduced density)
- abnormal or bushy (ramified) capillaries

For each finding, give a short label, a confidence from 0 to 1, and a bounding box in normalized coordinates
(x, y = top-left corner; w, h = size; all between 0 and 1 relative to the full image).

Measurements:
- width_um: typical capillary loop width in micrometers. Only give it when a calibration (µm per pixel) is provided
  and loops are clearly in focus; otherwise null.
- density_per_mm: capillaries per millimeter along the distal row. Same rule: only with calibration and a clear distal row; otherwise null.

note: one or two plain sentences on image quality and what a clinician should check. If the image is blurry,
not a nailfold, or too dark to judge, say so, return no findings, and leave the measurements null.
Be conservative: an empty findings list is a valid answer.`;

const nullableNumber = { anyOf: [{ type: "number" }, { type: "null" }] };
const OUTPUT_SCHEMA = {
  type: "object",
  properties: {
    findings: {
      type: "array",
      items: {
        type: "object",
        properties: {
          label: { type: "string" },
          confidence: { type: "number" },
          x: { type: "number" },
          y: { type: "number" },
          w: { type: "number" },
          h: { type: "number" },
        },
        required: ["label", "confidence", "x", "y", "w", "h"],
        additionalProperties: false,
      },
    },
    width_um: nullableNumber,
    density_per_mm: nullableNumber,
    note: { type: "string" },
  },
  required: ["findings", "width_um", "density_per_mm", "note"],
  additionalProperties: false,
};

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const clamp01 = (n) => Math.min(1, Math.max(0, Number(n) || 0));
const positive = (v) => {
  const n = parseFloat(v);
  return Number.isFinite(n) && n > 0 ? n : null;
};

export async function POST(request) {
  if (!process.env.ANTHROPIC_API_KEY) {
    return json({ error: "The analysis service isn't configured yet (missing API key)." }, 500);
  }

  let form;
  try {
    form = await request.formData();
  } catch {
    return json({ error: "Expected a multipart form upload with an image." }, 400);
  }

  const image = form.get("image");
  if (!image || typeof image === "string") return json({ error: "No image was attached." }, 400);
  if (!IMAGE_TYPES.has(image.type)) return json({ error: "Image must be JPEG, PNG or WebP." }, 415);
  if (image.size > MAX_IMAGE_BYTES) return json({ error: "Image is too large. Try a smaller resolution." }, 413);

  const umPerPx = positive(form.get("um_per_px"));
  const width = positive(form.get("width"));
  const height = positive(form.get("height"));
  const data = Buffer.from(await image.arrayBuffer()).toString("base64");

  const sizeLine = width && height ? `The original frame is ${Math.round(width)}×${Math.round(height)} pixels.` : "";
  const calibLine = umPerPx
    ? `Calibration for the original frame: ${umPerPx.toFixed(4)} µm per pixel.`
    : "No calibration was provided, so width_um and density_per_mm must be null.";

  const client = new Anthropic();
  let response;
  try {
    response = await client.beta.messages.create({
      model: MODEL,
      max_tokens: 16000,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: SYSTEM,
      output_config: { format: { type: "json_schema", schema: OUTPUT_SCHEMA } },
      messages: [
        {
          role: "user",
          content: [
            { type: "image", source: { type: "base64", media_type: image.type, data } },
            { type: "text", text: `Analyze this nailfold image. ${sizeLine} ${calibLine}`.trim() },
          ],
        },
      ],
    });
  } catch (error) {
    if (error instanceof Anthropic.RateLimitError) {
      return json({ error: "The analysis service is busy. Please try again in a minute." }, 429);
    }
    if (error instanceof Anthropic.AuthenticationError) {
      console.error("Anthropic auth failed: check ANTHROPIC_API_KEY");
      return json({ error: "The analysis service isn't configured correctly." }, 500);
    }
    if (error instanceof Anthropic.BadRequestError) {
      console.error("Anthropic rejected the request:", error.message);
      return json({ error: "The image couldn't be analyzed. Try another capture." }, 400);
    }
    if (error instanceof Anthropic.APIError) {
      console.error(`Anthropic API error ${error.status}:`, error.message);
      return json({ error: "The analysis service had a problem. Please try again." }, 502);
    }
    console.error("Analysis failed:", error);
    return json({ error: "Couldn't reach the analysis service. Please try again." }, 502);
  }

  if (response.stop_reason === "refusal") {
    return json({ error: "The model declined to analyze this image." }, 422);
  }
  if (response.stop_reason === "max_tokens") {
    return json({ error: "The analysis was cut off. Please try again." }, 502);
  }

  const text = response.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    console.error("Model returned non-JSON output:", text.slice(0, 500));
    return json({ error: "The analysis came back in an unexpected format. Please try again." }, 502);
  }

  // Convert normalized boxes to pixel boxes on the original frame, so scan.html can draw them directly.
  const W = width || 1;
  const H = height || 1;
  const findings = (parsed.findings || []).map((f) => {
    const x = clamp01(f.x), y = clamp01(f.y);
    const w = Math.min(clamp01(f.w), 1 - x), h = Math.min(clamp01(f.h), 1 - y);
    return {
      label: String(f.label || "finding"),
      confidence: clamp01(f.confidence),
      box: [Math.round(x * W), Math.round(y * H), Math.round(w * W), Math.round(h * H)],
    };
  });

  const round1 = (n) => (typeof n === "number" && Number.isFinite(n) ? Math.round(n * 10) / 10 : null);
  return json({
    findings,
    width_um: umPerPx ? round1(parsed.width_um) : null,
    density_per_mm: umPerPx ? round1(parsed.density_per_mm) : null,
    note: String(parsed.note || ""),
  });
}

export function GET() {
  return json({ error: "Send a POST with an image." }, 405);
}
