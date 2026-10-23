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

const SYSTEM = `Describe image visibility and acquisition quality for an experimental research notebook. This is external visual commentary, not the trained CapTrail detector. Do not infer disease, healthy status, confirmed vessels, lumen boundaries, physical widths, density, or referral decisions. Return findings as an empty array, width_um and density_per_mm as null. In note, briefly describe visible focus, illumination and uncertainty without medical conclusions.`;

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
  const origin=request.headers.get('origin');
  if((origin&&origin!==new URL(request.url).origin)||request.headers.get('x-captrail-request')!=='1')return json({error:'Same-origin request required.'},403);
  const declared=Number(request.headers.get('content-length')||0);
  if(declared>MAX_IMAGE_BYTES)return json({error:'Upload exceeds size budget.'},413);
  if (!process.env.ANTHROPIC_API_KEY) {
    return json({ error: "The analysis service isn't configured yet (missing API key)." }, 500);
  }

  let form;
  try {
    form = await request.formData();
  } catch {
    return json({ error: "Expected a multipart form upload with an image." }, 400);
  }

  if(form.get("external_commentary_consent")!=="true")return json({error:"Separate permission to send this image to Anthropic is required."},422);
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
      console.error("External commentary rejected request");
      // Billing problems come back as a 400; only the message tells them apart from a bad image.
      if (/credit balance/i.test(error.error?.error?.message ?? "")) {
        return json({ error: "The analysis service is out of API credits. Please tell the Captrail team." }, 503);
      }
      return json({ error: "The image couldn't be analyzed. Try another capture." }, 400);
    }
    if (error instanceof Anthropic.APIError) {
      console.error("External commentary API error", error.status);
      return json({ error: "The analysis service had a problem. Please try again." }, 502);
    }
    console.error("External commentary connection failed");
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
    console.error("External commentary returned invalid JSON");
    return json({ error: "The analysis came back in an unexpected format. Please try again." }, 502);
  }

  // Convert normalized boxes to pixel boxes on the original frame, so scan.html can draw them directly.
  const W = width || 1;
  const H = height || 1;
  const findings = [].map((f) => {
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
    width_um: null,
    density_per_mm: null,
    note: String(parsed.note || ""),
  });
}

export function GET() {
  return json({ error: "Send a POST with an image." }, 405);
}
