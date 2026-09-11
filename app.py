import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

try:
    import google.generativeai as genai
except Exception as exc:  # pragma: no cover - depends on environment
    genai = None
    import_error = exc
else:
    import_error = None


load_dotenv()

st.set_page_config(page_title="Universal Bridge", page_icon="🌉", layout="wide")


def get_model_name() -> str:
    return "gemini-1.5-flash"


def normalize_text(value: str | None) -> str:
    return (value or "").strip()


def parse_uploaded_file(file_obj) -> dict[str, Any]:
    if file_obj is None:
        return {}

    file_bytes = file_obj.read()
    file_name = file_obj.name
    file_type = file_obj.type or ""

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
                inputs.append(
                    f"\n--- Attachment: {name} ---\n{text_value[:8000]}\n"
                )

        response = model.generate_content(inputs)
        raw_text = response.text

    finally:
        for temp_path in temp_files:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return clean_json_response(raw_text)


st.title("🌉 Universal Bridge")
st.caption("Turn messy real-world inputs into structured, verified, and actionable results using Gemini.")

st.markdown(
    """
    This demo illustrates a practical societal benefit: combining messy real-world inputs such as voice notes, scanned records, weather/news feeds, and images into a structured emergency or dispatch payload.
    """
)

with st.sidebar:
    st.header("Configuration")
    st.write("Set your GEMINI_API_KEY in a .env file or environment variables before running.")
    st.code("GEMINI_API_KEY=your_api_key_here")

    st.header("Scenario")
    use_case = st.selectbox(
        "Choose the use case",
        [
            "Emergency Medical Triage",
            "Disaster Relief & Traffic Rerouting",
            "Custom Multi-Modal Input",
        ],
    )

    st.header("Input mode")
    input_mode = st.radio("Preferred input type", ["Text + Files", "Voice note"])


if genai is None:
    st.error(
        "The google-generativeai package is not available. Install the dependencies first: `pip install -r requirements.txt`."
    )
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Inputs")
    scenario_text = st.text_area(
        "Describe the situation",
        height=180,
        placeholder="Example: A man was found unconscious, has difficulty breathing, and is carrying a handwritten medication list plus a photo of a damaged ankle.",
    )

    uploaded_files = st.file_uploader(
        "Upload images, scans, text records, or audio notes",
        accept_multiple_files=True,
    )

    if input_mode == "Voice note":
        voice_note = st.file_uploader("Upload a voice note", type=["wav", "mp3", "m4a", "ogg"])
    else:
        voice_note = None

with col2:
    st.subheader("Structured Output")
    output_placeholder = st.empty()


if st.button("Run Universal Bridge", type="primary"):
    attachments = []

    for uploaded_file in uploaded_files or []:
        attachments.append(parse_uploaded_file(uploaded_file))

    if voice_note is not None:
        attachments.append(parse_uploaded_file(voice_note))

    if not scenario_text.strip() and not attachments:
        st.warning("Please provide at least some text or an uploaded file to analyze.")
        st.stop()

    try:
        result = summarize_with_gemini(use_case, scenario_text, attachments)
        output_placeholder.json(result)
        st.success("Universal Bridge finished processing the input.")
    except Exception as exc:
        st.error(f"Processing failed: {exc}")


st.markdown("---")
st.subheader("Example use cases")
st.markdown(
    """
    1. Emergency medical triage: combine a voice note, scanned medication list, and symptoms.
    2. Disaster relief: merge weather alerts, news summaries, and photos of blocked roads.
    3. Real-time assistance: transform unstructured records into a verified dispatch payload.
    """
)
