import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

try:
    import google.generativeai as genai
except Exception as exc:  # pragma: no cover - depends on environment
    genai = None
    import_error = exc
else:
    import_error = None


load_dotenv()

app = FastAPI(title="Universal Bridge")


def get_model_name() -> str:
    return "gemini-1.5-flash"


def normalize_text(value: str | None) -> str:
    return (value or "").strip()


def make_attachment(uploaded_file: UploadFile) -> dict[str, Any]:
    file_bytes = uploaded_file.file.read()
    file_name = uploaded_file.filename or "attachment"
    file_type = uploaded_file.content_type or ""

    if file_type.startswith("image/"):
        return {
            "kind": "image",
            "name": file_name,
            "bytes": file_bytes,
            "mime_type": file_type,
        }

    if file_type.startswith("audio/") or Path(file_name).suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg"}:
        return {
            "kind": "audio",
            "name": file_name,
            "bytes": file_bytes,
            "mime_type": file_type or "audio/mpeg",
        }

    text_value = file_bytes.decode("utf-8", errors="ignore")
    return {
        "kind": "text",
        "name": file_name,
        "text": text_value[:8000],
    }


def clean_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "scenario": "unstructured_result",
            "summary": cleaned,
            "urgency_level": "unknown",
            "actionable_steps": ["Review the raw model output manually."],
            "extracted_entities": {},
            "confidence": "low",
            "verification_notes": ["The model response was not valid JSON; the raw response is included in summary."],
            "raw_response": cleaned,
        }


def summarize_with_gemini(use_case: str, scenario_text: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    if genai is None:
        raise RuntimeError(f"google-generativeai import failed: {import_error}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to a .env file or environment variables.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(get_model_name())

    prompt = f"""
You are a universal bridge that converts messy, real-world inputs into structured and verified results that help society.

Task: Analyze the input and return valid JSON only for the scenario below.
Scenario: {use_case}

User context:
{normalize_text(scenario_text) or 'No free-text context was provided.'}

Return a JSON object with exactly these keys:
- scenario
- summary
- urgency_level
- actionable_steps
- extracted_entities
- confidence
- verification_notes

Rules:
1. Prioritize safety, emergency response, and public benefit.
2. Extract structured facts from any uploaded images, scans, text records, and audio notes.
3. If a fact is missing, say so explicitly in verification_notes.
4. Keep the output valid machine-readable JSON.
5. Do not include markdown fences.
"""

    inputs: list[Any] = [prompt]
    temp_files: list[str] = []

    try:
        for attachment in attachments:
            kind = attachment.get("kind")
            name = attachment.get("name", "attachment")

            if kind == "image":
                image = Image.open(io.BytesIO(attachment["bytes"]))
                inputs.append(image)
            elif kind == "audio":
                suffix = Path(name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(attachment["bytes"])
                    temp_path = tmp.name
                temp_files.append(temp_path)
                inputs.append(Path(temp_path))
            else:
                text_value = attachment.get("text", "")
                inputs.append(f"\n--- Attachment: {name} ---\n{text_value[:8000]}\n")

        response = model.generate_content(inputs)
        raw_text = response.text

    finally:
        for temp_path in temp_files:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return clean_json_response(raw_text)


UI_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Universal Bridge</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #dbe3ee;
      --primary: #3b82f6;
      --primary-dark: #1f5fcd;
      --success: #16a34a;
      --danger: #dc2626;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(135deg, #edf4ff 0%, #f8fafc 100%);
      color: var(--text);
    }

    .container {
      max-width: 1200px;
      margin: 32px auto;
      padding: 24px;
    }

    .header {
      margin-bottom: 20px;
    }

    .header h1 {
      margin: 0;
      font-size: clamp(2rem, 3vw, 3rem);
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .header p {
      color: var(--muted);
      margin-top: 10px;
      font-size: 1.05rem;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 24px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 24px;
    }

    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }

    select,
    textarea,
    input[type=\"file\"] {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 1rem;
      background: #fff;
      color: var(--text);
      margin-bottom: 18px;
    }

    textarea {
      min-height: 180px;
      resize: vertical;
    }

    button {
      border: none;
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      border-radius: 12px;
      padding: 13px 18px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease;
    }

    button:hover { transform: translateY(-1px); }

    .status {
      margin-top: 18px;
      min-height: 24px;
      font-size: 0.95rem;
      color: var(--muted);
    }

    .status.error { color: var(--danger); }
    .status.success { color: var(--success); }

    .result-box {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 300px;
      overflow: auto;
      font-size: 0.95rem;
      line-height: 1.5;
    }

    .example-list {
      margin-top: 18px;
      display: grid;
      gap: 10px;
    }

    .example-item {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      color: var(--muted);
    }

    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .container { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"header\">
      <h1>?? Universal Bridge</h1>
      <p>Turn messy human input into structured, verified, and actionable results using Gemini.</p>
    </div>

    <div class=\"grid\">
      <div class=\"panel\">
        <form id=\"bridgeForm\" enctype=\"multipart/form-data\">
          <label for=\"useCase\">Use case</label>
          <select id=\"useCase\" name=\"use_case\">
            <option>Emergency Medical Triage</option>
            <option>Disaster Relief &amp; Traffic Rerouting</option>
            <option>Custom Multi-Modal Input</option>
          </select>

          <label for=\"scenarioText\">Describe the situation</label>
          <textarea id=\"scenarioText\" name=\"scenario_text\" placeholder=\"Example: A man was found unconscious, has difficulty breathing, and is carrying a handwritten medication list plus a photo of a damaged ankle.\"></textarea>

          <label for=\"attachments\">Upload images, text records, or audio notes</label>
          <input id=\"attachments\" name=\"files\" type=\"file\" multiple accept=\"image/*,audio/*,.txt,.json,.md\" />

          <button type=\"submit\">Run Universal Bridge</button>
        </form>

        <div id=\"status\" class=\"status\"></div>
      </div>

      <div class=\"panel\">
        <h2 style=\"margin-top:0;\">Structured Output</h2>
        <div id=\"result\" class=\"result-box\">Waiting for input...</div>

        <div class=\"example-list\">
          <div class=\"example-item\"><strong>Emergency triage:</strong> handwritten medical records + voice note + symptoms</div>
          <div class=\"example-item\"><strong>Disaster relief:</strong> weather alerts + road photos + incident descriptions</div>
          <div class=\"example-item\"><strong>Dispatch payload:</strong> structured summary, urgency, and next steps</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const form = document.getElementById('bridgeForm');
    const status = document.getElementById('status');
    const result = document.getElementById('result');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      status.textContent = 'Processing...';
      status.className = 'status';
      result.textContent = 'Working on your input...';

      const formData = new FormData(form);

      try {
        const response = await fetch('/api/process', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Processing failed.');
        }

        result.textContent = JSON.stringify(data, null, 2);
        status.textContent = 'Processing completed successfully.';
        status.className = 'status success';
      } catch (error) {
        result.textContent = JSON.stringify({ error: error.message }, null, 2);
        status.textContent = error.message;
        status.className = 'status error';
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(content=UI_HTML)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process")
async def process_input(
    use_case: str = Form(...),
    scenario_text: str = Form(""),
    files: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    attachments = []

    for uploaded_file in files:
        if uploaded_file.filename:
            attachments.append(make_attachment(uploaded_file))

    try:
        result = summarize_with_gemini(use_case, scenario_text, attachments)
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
