# 🛡️ CallGuard — Conversation Threat Detection System
### Alpha Version 2.0

> **An AI-powered audio analysis platform that listens to what humans miss.**
> CallGuard transcribes recorded conversations, identifies who is speaking, detects threatening or harmful language, and surfaces everything in a clean investigative dashboard — all running locally, with no API costs.

---

## What Problem Does This Solve?

In call centers, law enforcement, enterprise security, and personal safety contexts, thousands of hours of audio are recorded every day — and almost none of it gets reviewed. A human analyst can't realistically screen every call for threats, abuse, or coordinated harm planning.

CallGuard automates that first pass. It doesn't replace human judgment — it surfaces the moments that need it.

---

## What It Does

Upload a call recording and CallGuard will:

1. **Convert and clean the audio** — any format (mp3, m4a, wav, flac, ogg) gets normalized to 16kHz mono
2. **Transcribe with speaker separation** — Whisper handles the speech-to-text; pyannote.audio separates who said what
3. **Detect the language automatically** — supports 100 languages, with confidence scoring; falls back gracefully on short or noisy clips
4. **Match speakers to enrolled identities** — if you've enrolled someone's voice beforehand, their name replaces the generic "SPEAKER_00" label in the transcript
5. **Flag dangerous content** — two-pass detection catches threats, abuse, harm planning, and disaster-related language in English, Hindi (Devanagari + Roman), and Urdu (Nastaliq + Roman)
6. **Store everything** — every call, segment, flag, and voice match is persisted to SQLite so nothing is lost between sessions
7. **Show it all in a dashboard** — live transcript with highlighted flags, category breakdowns, timeline charts, and a full past-calls browser

---

## Alpha v2 — What Changed

Alpha v1 was the initial end-to-end build. Alpha v2 focused on making the core ML pipeline actually reliable:

**Bug fixes in v2:**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Hindi audio transcribed as garbled Latin text | Whisper auto-detect misidentified short Hindi clips as Norwegian (nn) at 56% confidence | Switched to explicit `detect_language()` with confidence scoring; added retry on full audio if confidence < 60%; force `task="transcribe"` to prevent silent translation to English |
| Voice matching always returning 0% similarity | Audio slices were being *rejected* when Whisper reported segment end times slightly beyond the actual audio length — causing empty slice lists and zero vectors | Changed rejection to clamping: `end = min(end_sample, len(wav_full))` |
| `SPEAKER_00` still showing after enrollment | `identify_speakers()` was skipping all segments labeled `"UNKNOWN"` (diarization-off mode) | Fixed grouping to include all speaker labels including UNKNOWN |
| Past Calls table crashing with Arrow type error | Mixed int/string in `Flags` column broke pyarrow serialization | Normalized all table columns to consistent string types |
| App slow on every analysis | Whisper and diarization models reloading from disk each time | Added `@st.cache_resource` caching — models load once per session |
| 3.85 GB alignment model downloading mid-analysis | whisperx alignment step was trying to download wav2vec2 for misdetected languages | Alignment step removed entirely — not needed for segment-level threat detection |

**New features in v2:**
- Voice enrollment and cross-call speaker recognition (resemblyzer 256-dim embeddings)
- `matched_name`, `match_confidence`, `match_status` columns on the Segment table
- Debug logging for every similarity score comparison
- `WHISPER_LANGUAGE` env override for forced language mode
- Manage Enrolled Voices tab with delete capability

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  ┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌─────────┐  │
│  │ Analyze Call │ │ Past Calls │ │  Enroll  │ │ Manage  │  │
│  │              │ │            │ │  Voice   │ │ Voices  │  │
│  └──────┬───────┘ └─────┬──────┘ └────┬─────┘ └────┬────┘  │
└─────────│───────────────│─────────────│──────────────│──────┘
          │               │             │              │
          ▼               ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline Layer                          │
│                                                              │
│  preprocess_audio.py  →  pydub + ffmpeg                      │
│         ↓                                                    │
│  transcribe.py        →  whisperx (faster-whisper)           │
│         ↓                detect_language() → confidence      │
│         ↓                task="transcribe" (no translation)  │
│         ↓                pyannote/speaker-diarization-3.1    │
│         ↓                                                    │
│  voice_recognition.py →  resemblyzer VoiceEncoder           │
│         ↓                256-dim cosine similarity           │
│         ↓                clamped audio slicing               │
│         ↓                                                    │
│  detector.py          →  Pass 1: rapidfuzz bulk matching     │
│                          Pass 2: context window analysis     │
│                          keywords.json (EN + HI + UR)        │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                     SQLite Database                          │
│  calls  │  segments (+ matched_name, match_confidence)       │
│  flags  │  call_summaries  │  voice_profiles                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Detection: How It Works

### Pass 1 — Fuzzy Keyword Matching
Each transcript segment is run through `rapidfuzz.process.extract` against the full keyword list in bulk (vectorized, not a loop). A segment scoring above the fuzzy threshold (default 80/100) becomes a candidate.

This catches misspellings, filler words, and phonetic near-misses:
- `"goli maar"` matches `"goli maar dunga"` ✅
- `"i wil kil you"` matches `"i will kill you"` ✅

### Pass 2 — Context Window Analysis
The candidate segment is analyzed alongside the 4 preceding segments. If the surrounding conversation contains benign signals (sports talk, workplace context, sarcasm markers), the confidence score is dampened.

This is the **swappable interface** — the `analyze_segment()` function has a fixed contract:
```python
def analyze_segment(segment: dict, context_window: list[dict]) -> dict:
    # Returns: {flag: bool, category: str, confidence: float, matched_keyword: str}
```
Drop in any classifier, LLM call, or external moderation API without touching anything else.

### Keyword Coverage — 3 Languages

| Category | What it catches | Languages |
|----------|----------------|-----------|
| `threat` | Direct threats to life, bodily harm | EN, HI, UR |
| `abuse` | Verbal abuse, harassment, degradation | EN, HI, UR |
| `harm_planning` | Coordinated planning to harm or silence someone | EN, HI, UR |
| `disaster` | Bomb threats, arson, mass attack planning | EN, HI, UR |

Hindi is covered in both Devanagari (`मैं तुम्हें मार दूंगा`) and Roman transliteration (`marna padega`) — because Whisper transcribes Hindi speech in Roman script by default.

Urdu is covered in both Nastaliq (`تمہیں ماردوں گا`) and Roman (`tumhe maar dunga`).

---

## Voice Recognition

CallGuard uses [resemblyzer](https://github.com/resemble-ai/Resemblyzer) to generate a 256-dimensional voice embedding from a clean audio sample. These embeddings are stored in the `voice_profiles` SQLite table.

During analysis:
1. Diarization identifies speaker segments (SPEAKER_00, SPEAKER_01, etc.)
2. Audio slices for each speaker are concatenated and embedded
3. Cosine similarity is computed against every enrolled profile
4. If similarity ≥ threshold (default 0.75), the generic label is replaced with the person's name

The matching threshold is tunable via `RECOGNITION_THRESHOLD` in `.env`. Real-world voice recordings typically score between 0.65–0.92 against a clean enrollment sample.

Debug output during analysis:
```
[voice_recognition] Processing speaker: 'SPEAKER_00' (4 segments)
[voice_recognition]   Combined audio: 18.4s
[voice_recognition]   Similarity scores:
[voice_recognition]     vs 'John Smith': 0.8731 — ✅ MATCH (threshold=0.75)
[voice_recognition]     vs 'Jane Doe':  0.3214 — ❌ below threshold
[voice_recognition]   → MATCHED: 'John Smith' (0.873)
```

---

## Project Structure

```
callguard/
│
├── app.py                      # Streamlit dashboard — 4 tabs
│                               #   🎙️ Analyze Call
│                               #   📂 Past Calls
│                               #   🎤 Enroll a Voice
│                               #   👥 Manage Voices
│
├── pipeline/
│   ├── preprocess_audio.py     # Any format → 16kHz mono WAV (pydub + ffmpeg)
│   ├── transcribe.py           # whisperx wrapper — detect_language, transcribe, diarize
│   ├── voice_recognition.py    # resemblyzer enrollment + cosine similarity matching
│   ├── detector.py             # Two-pass threat detection + swappable analyze_segment()
│   └── keywords.json           # Editable threat keywords — EN, HI, UR
│
├── db/
│   ├── models.py               # SQLAlchemy ORM: Call, Segment, Flag, CallSummary, VoiceProfile
│   └── database.py             # Session management + all CRUD helpers
│
├── .streamlit/
│   └── config.toml             # Theme + upload size config
│
├── sample_audio/               # Drop test recordings here
├── requirements.txt
├── packages.txt                # System packages for cloud (ffmpeg, libsndfile1)
├── .env.example                # Full config reference with explanations
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/prithvirajsingh3108-ctrl/Call-guard.git
cd Call-guard

# 2. Python 3.11 venv (required — whisperx not compatible with 3.12+)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. System dependency
brew install ffmpeg          # macOS
# sudo apt install ffmpeg    # Ubuntu/Debian

# 4. PyTorch (CPU — works everywhere)
pip install torch torchvision torchaudio

# 5. All other dependencies
pip install -r requirements.txt

# 6. Configure
cp .env.example .env
# Edit .env — at minimum set HF_TOKEN

# 7. Run
streamlit run app.py
```

Open http://localhost:8501

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | Hugging Face token for pyannote diarization |
| `WHISPER_MODEL` | `base` | Model size: `tiny` / `base` / `small` / `medium` / `large-v2` |
| `WHISPER_LANGUAGE` | *(empty)* | Force language code e.g. `hi`, `ur`, `en`. Leave blank for auto-detect |
| `COMPUTE_DEVICE` | `cpu` | `cpu` or `cuda` |
| `AUDIO_SAMPLE_RATE` | `16000` | Resample target (Hz) |
| `DATABASE_PATH` | `callguard.db` | SQLite file path |
| `KEYWORDS_PATH` | `pipeline/keywords.json` | Threat keyword file |
| `FUZZY_THRESHOLD` | `80` | 0–100 fuzzy match sensitivity (lower = more permissive) |
| `CONTEXT_WINDOW_SIZE` | `4` | Prior segments used for context dampening |
| `RECOGNITION_THRESHOLD` | `0.75` | Voice match cosine similarity cutoff (0.0–1.0) |

### Diarization Setup

Speaker diarization requires accepting two model licenses on Hugging Face:

1. Accept at → https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept at → https://huggingface.co/pyannote/segmentation-3.0
3. Generate token at → https://huggingface.co/settings/tokens
4. Add to `.env`: `HF_TOKEN=hf_xxxxxxxxxxxx`

Without a token, the app runs without speaker separation — all segments show as `UNKNOWN`.

---

## Extending the Classifier

The threat detection logic is deliberately isolated behind a clean interface. To swap the keyword-based analyzer for a real ML model or external API:

```python
# pipeline/detector.py — replace ONLY this function body
def analyze_segment(segment: dict, context_window: list[dict]) -> dict:
    """
    Your custom classifier goes here.
    Input:  segment dict + list of preceding segments
    Output: {flag: bool, category: str, confidence: float, matched_keyword: str}
    """
    # Example: call an external moderation API
    # Example: run a fine-tuned HuggingFace classifier
    # Example: prompt an LLM
    pass
```

Nothing else in the codebase needs to change.

---

## Database Schema

```sql
voice_profiles   -- enrolled speaker identities + 256-dim embeddings
calls            -- one row per processed audio file
segments         -- one row per speaker turn
                 --   + matched_name, match_confidence, match_status (v2)
flags            -- one row per detected threat
                 --   + category, confidence, matched_keyword, context_window
call_summaries   -- aggregate stats per call
```

---

## Known Limitations (Alpha)

- **CPU only** — transcription on CPU takes roughly 1× real time for `base` model (a 2-minute call takes ~2 minutes). GPU reduces this to near real-time.
- **Voice matching requires clean enrollment audio** — a noisy sample will produce a weak embedding. Use 15–30s of clean speech for best results.
- **Short clips (<10s) have unreliable language detection** — use `WHISPER_LANGUAGE` in `.env` to force the language when testing with short samples.
- **Keyword detection is language-script dependent** — Hindi spoken in a call will be transcribed in Roman script by Whisper. The keyword list covers Roman transliterations for this reason.
- **SQLite is not concurrent** — fine for a single-user prototype, not for multi-user production.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Speech-to-text | openai-whisper + whisperx (faster-whisper backend) |
| Speaker diarization | pyannote.audio 3.1 |
| Voice recognition | resemblyzer (256-dim GE2E embeddings) |
| Audio preprocessing | pydub + ffmpeg |
| Fuzzy text matching | rapidfuzz |
| Database ORM | SQLAlchemy + SQLite |
| Dashboard | Streamlit + Plotly |
| Public tunnel | ngrok |
| Version control | Git + GitHub |

---

## Roadmap (Post-Alpha)

- [ ] Replace keyword detector with a fine-tuned multilingual classifier
- [ ] Add real-time streaming analysis (WebRTC input)
- [ ] Expand voice enrollment to handle multiple samples per person
- [ ] Add export: PDF report, CSV flag log
- [ ] Role-based access control for multi-analyst deployments
- [ ] Add Punjabi, Arabic, Bengali keyword coverage
- [ ] GPU support for production-speed transcription

---

## Repository

**GitHub:** https://github.com/prithvirajsingh3108-ctrl/Call-guard

**Version:** Alpha v2.0
**Build date:** July 2026
**Status:** Prototype — not production ready

---

*Built as an intern prototype project. Prioritizes working end-to-end functionality over scale or polish. The architecture is designed for replaceability — every ML component can be swapped without touching the surrounding pipeline.*
