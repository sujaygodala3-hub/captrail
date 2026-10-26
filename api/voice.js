// Vercel serverless function: POST /api/voice
// The voice assistant on scan.html sends JSON with the *text* of a spoken question
// and a few facts about the page state. No images or audio are ever sent here.
//   { "question": "how do I calibrate?",
//     "state": { "connected": true, "image": true, "calibrated": true, "um_per_px": 6.094,
//                "zoomed": false, "enhance": "Off" } }
// Returns: { "reply": "One to three short sentences to speak aloud." }
//
// Same setup as api/analyze.js: the API key comes only from process.env.ANTHROPIC_API_KEY.
import Anthropic from "@anthropic-ai/sdk";

const MODEL = "claude-sonnet-5-5";
const MAX_QUESTION = 500;

const SYSTEM = `You are the Captrail setup helper, a voice assistant on the microscope page of Captrail, a student research prototype for nailfold capillaroscopy. People talk to you while they use a USB microscope on a 3D-printed base board and finger rest. Your reply is read aloud by the browser.

Answer in one to three short, plain spoken sentences. No lists, headings, markdown, emoji or URLs. If the page state matters to the answer (for example whether it is calibrated or an image is captured), use it.

Capture tips to use when they are relevant:
- Put one drop of mineral oil on the nailfold. Don't use plastic wrap.
- Turn the microscope's LED down low; bright light causes glare.
- Keep any glare on the nail, not on the skin.
- Focus with the microscope's focus wheel only, and don't touch it again after calibrating.
- Use Best of 8 for the sharpest image, and keep only sharp frames.
- To calibrate, image the calibration card's cross, click the outer edges of two long lines that are 2 millimeters apart, and enter 2000 micrometers. Recalibrate if zoom or focus changes.

Never diagnose, and never interpret capillaries, measurements or AI results as healthy or unhealthy. If someone asks what a result means for their health, say that Captrail is a screening research prototype, not a diagnostic tool, and that a doctor should review any concerns. If a question has nothing to do with Captrail or using the microscope, say briefly that you can only help with the Captrail setup.`;

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

// Keep only the state fields we expect, with the right types, so nothing else reaches the prompt.
function cleanState(raw) {
  const s = raw && typeof raw === "object" ? raw : {};
  const bool = (v) => v === true;
  const num = Number(s.um_per_px);
  return {
    microscope_connected: bool(s.connected),
    image_captured: bool(s.image),
    calibrated: bool(s.calibrated),
    um_per_px: Number.isFinite(num) && num > 0 ? Math.round(num * 1000) / 1000 : null,
    viewing_zoomed_image: bool(s.zoomed),
    enhance_mode: typeof s.enhance === "string" ? s.enhance.slice(0, 20) : "Off",
  };
}

// The reply is spoken, so strip any formatting the model adds anyway.
const speakable = (t) =>
  t.replace(/[*_#`>]+/g, "").replace(/^\s*[-•]\s+/gm, "").replace(/\s+/g, " ").trim();

export async function POST(request) {
  if (!process.env.ANTHROPIC_API_KEY) {
    return json({ error: "The voice helper isn't configured yet." }, 500);
  }
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Send the question as JSON." }, 400);
  }
  const question = typeof body?.question === "string" ? body.question.trim().slice(0, MAX_QUESTION) : "";
  if (!question) return json({ error: "No question was received." }, 400);
  const state = cleanState(body.state);

  const client = new Anthropic();
  let response;
  try {
    response = await client.beta.messages.create({
      model: MODEL,
      max_tokens: 2048,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      output_config: { effort: "low" },
      system: SYSTEM,
      messages: [
        {
          role: "user",
          content: `Page state: ${JSON.stringify(state)}\n\nQuestion: ${question}`,
        },
      ],
    });
  } catch (error) {
    if (error instanceof Anthropic.RateLimitError) {
      return json({ error: "The voice helper is busy. Try again in a moment." }, 429);
    }
    if (error instanceof Anthropic.AuthenticationError) {
      console.error("Voice helper request failed");
      return json({ error: "The voice helper isn't configured correctly." }, 500);
    }
    if (error instanceof Anthropic.BadRequestError) {
      console.error("Voice helper request failed");
      if (/credit balance/i.test(error.error?.error?.message ?? "")) {
        return json({ error: "The voice helper is out of API credits." }, 503);
      }
      return json({ error: "I couldn't answer that one. Try asking another way." }, 400);
    }
    if (error instanceof Anthropic.APIError) {
      console.error("Voice helper request failed");
      return json({ error: "The voice helper had a problem. Try again." }, 502);
    }
    console.error("Voice helper request failed");
    return json({ error: "Couldn't reach the voice helper. Try again." }, 502);
  }

  if (response.stop_reason === "refusal") {
    return json({ reply: "I can't help with that one. I can help you set up and use the Captrail microscope." });
  }
  const text = response.content.filter((b) => b.type === "text").map((b) => b.text).join(" ");
  const reply = speakable(text);
  if (!reply) return json({ error: "I didn't catch an answer. Try again." }, 502);
  return json({ reply });
}

export function GET() {
  return json({ error: "Send a POST with a question." }, 405);
}
