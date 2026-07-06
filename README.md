# 🛡️ CallGuard — Conversation Threat Detection Prototype

A Python prototype that transcribes audio call recordings with speaker labels,
detects threatening/harmful content, and surfaces results in a Streamlit dashboard.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install system dependency (macOS)
brew install ffmpeg

# 3. Install Python packages
#    Install PyTorch first (CPU build — works everywhere):
pip install torch torchvision torchaudio
#    Then the rest:
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set HF_TOKEN if you want speaker diarization

# 5. Run the dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Step-by-Step Build Verification

### Step 1 — Plain Whisper transcription
```bash
# No diarization, no HF token needed
USE_DIARIZATION=false python pipeline/transcribe.py sample_audio/your_file.wav
```
**Expected output:** A printed segment list and a `*_transcript.json` file.

### Step 2 — whisperx + diarization
```bash
# Requires HF_TOKEN in .env
python pipeline/transcribe.py sample_audio/your_file.wav
```
**Expected output:** Same as Step 1 but each segment has `SPEAKER_00` / `SPEAKER_01` labels.

### Step 3+4 — Threat detector
```bash
# Uses built-in sample transcript (no audio file needed)
python pipeline/detector.py

# Or pass a real transcript JSON from Step 1/2:
python pipeline/detector.py your_transcript.json
```
**Expected output:** Segment list with `⚠ FLAGGED` markers and a summary block.

### Step 5 — Database
```bash
python db/database.py
```
**Expected output:** `Database ready: .../callguard.db` and a test call record printed.

### Step 6+7 — Full dashboard
```bash
streamlit run app.py
```
**Expected output:** Browser opens at http://localhost:8501 with two tabs.

---

## Project Structure

```
callguard/
├── app.py                  # Streamlit dashboard (Steps 6 & 7)
├── pipeline/
│   ├── transcribe.py       # whisperx wrapper (Steps 1 & 2)
│   ├── preprocess_audio.py # format conversion + resampling
│   ├── detector.py         # fuzzy match + context analysis (Steps 3 & 4)
│   └── keywords.json       # editable threat keyword list
├── db/
│   ├── models.py           # SQLAlchemy ORM schema (Step 5)
│   └── database.py         # connection + query helpers (Step 5)
├── sample_audio/           # put test recordings here
├── requirements.txt
├── .env.example
└── README.md
```

---

## Swapping the Classifier (Step 4 Extension)

The `analyze_segment()` function in `pipeline/detector.py` is the designed
swap point. Its contract is:

```python
def analyze_segment(segment: dict, context_window: list[dict]) -> dict:
    # Returns:
    # {
    #   "flag":            bool,
    #   "category":        str,
    #   "confidence":      float,   # 0.0 – 1.0
    #   "matched_keyword": str,
    # }
```

To plug in a different model or API, replace only the body of this function.
Nothing else in the codebase needs to change.

---

## Environment Variables

| Variable           | Default              | Description                              |
|--------------------|----------------------|------------------------------------------|
| `HF_TOKEN`         | *(required)*         | Hugging Face token for diarization       |
| `WHISPER_MODEL`    | `base`               | Whisper model size (tiny/base/small/...) |
| `COMPUTE_DEVICE`   | `cpu`                | `cpu` or `cuda`                          |
| `AUDIO_SAMPLE_RATE`| `16000`              | Resample target (Hz)                     |
| `DATABASE_PATH`    | `callguard.db`       | SQLite file path                         |
| `KEYWORDS_PATH`    | `pipeline/keywords.json` | Threat keyword file path            |
| `FUZZY_THRESHOLD`  | `80`                 | 0–100; lower = more permissive matching  |
| `CONTEXT_WINDOW_SIZE` | `4`              | Number of prior segments for context     |
| `USE_DIARIZATION`  | `true`               | Set `false` to skip speaker diarization  |

---

## Adding Test Audio

Put any `.wav` / `.mp3` file in `sample_audio/` and reference it in the
Streamlit uploader or CLI commands. Short clips (< 2 min) process fastest
during development.

---

## Diarization Setup

Speaker diarization requires accepting two model licenses on Hugging Face:

1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0

Then generate a token at https://huggingface.co/settings/tokens and add it
to `.env` as `HF_TOKEN`.

Without a token, set `USE_DIARIZATION=false` to use plain Whisper (no speaker labels).
