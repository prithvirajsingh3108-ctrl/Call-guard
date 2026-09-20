# 🛡️ CallGuard — Conversation Threat Detection System
### Version 3.0

> **An AI-powered audio analysis platform that hears the risk in every call.**
> CallGuard transcribes recorded conversations, separates speakers, and runs a multi-layer threat detection pipeline across English, Hindi, and Urdu — surfacing every flag in a premium investigative dashboard, entirely on your machine.

---

## What Problem Does This Solve?

In call centers, law enforcement, enterprise security, and personal safety contexts, thousands of hours of audio are recorded every day — and almost none of it gets reviewed. A human analyst can't realistically screen every call for threats, abuse, or coordinated harm planning.

CallGuard automates that first pass. It doesn't replace human judgment — it surfaces the moments that need it, ranked by confidence and severity.

---

## What It Does

Upload a call recording and CallGuard will:

1. **Clean and normalize the audio** — EBU R128 loudness normalization, high-pass filter at 80 Hz, silence trimming, and proper upsampling for telephone (8 kHz) audio
2. **Transcribe with speaker separation** — whisperx + faster-whisper handles speech-to-text; pyannote.audio 3.1 separates who said what
3. **Detect language per segment** — auto-detects en/hi/ur with confidence scoring; falls back gracefully
4. **Normalize text across scripts** — Unicode NFKC, Urdu glyph unification, Devanagari nukta stripping, repeated-character collapse, and Roman↔Devanagari cross-mapping so `maar dunga`, `مار دوں گا`, and `मार दूंगा` all match the same keyword
5. **Run four detection layers:**
   - **Layer 1 — Keyword** (rapidfuzz, three scorers, sliding windows across segment boundaries)
   - **Layer 2 — Semantic** (sentence-transformers multilingual cosine similarity against a curated phrase bank)
   - **Layer 3 — Context** (benign-signal dampening, speaker-escalation bonus)
   - **Layer 4 — LLM Judge** (optional, for borderline scores — off by default)
6. **Score each call 0–100** — fused risk score stored in the database and shown as a colour-coded pill
7. **Match speakers to enrolled identities** — resemblyzer 256-dim embeddings replace SPEAKER_00 labels
8. **Store everything** — every call, segment (with ASR diagnostics), flag (with per-stage scores), and alert is persisted to SQLite with zero-data-loss migrations
9. **Fire alerts** — dashboard banner, Telegram, email, and Twilio on configurable thresholds
10. **Monitor live** — real-time mic capture with faster-whisper (2–6 s latency on CPU)

---

## Version History

| Version | Highlights |
|---------|-----------|
| **v3.0** | 5-phase upgrade: config.py, 4-layer detector, text normalizer, semantic layer, LLM judge, eval harness (93.5% recall), live monitor, multi-channel alerts, password gate, diagnostics toggle, full UI redesign |
| **Alpha v2.0** | Explicit language detection, voice recognition (resemblyzer), model caching, Urdu keywords, DB schema v2 |
| **v1.0** | Initial end-to-end prototype |

---

## v3.0 — What Changed

### New files

| File | Purpose |
|------|---------|
| `config.py` | Single `Config` dataclass — every setting overridable from `.env` or `st.secrets` |
| `pipeline/normalize.py` | Full text normalisation pipeline (Unicode → Roman↔Devanagari overlay) |
| `pipeline/semantic.py` | sentence-transformers semantic similarity; phrase embeddings cached to disk |
| `pipeline/phrases.json` | ~80 curated example threat phrases in en/hi/ur/Hinglish per category |
| `pipeline/scoring.py` | `risk_score()` — 0–100 from top-two flag confidences |
| `pipeline/alerts.py` | Multi-channel alert dispatch with per-category cooldown and deduplication |
| `pipeline/live.py` | Background mic capture thread, chunk queue, live detection loop |
| `eval/labelled_segments.csv` | 92 labelled rows across 5 languages with hard negatives |
| `eval/run_eval.py` | Precision / Recall / F1 harness; saves `eval/report.json`; exits non-zero if recall < target |
| `tests/test_normalize.py` | 13 unit tests for the normalizer |
| `tests/test_detector.py` | 23 unit tests for the detector |
| `tests/test_scoring.py` | 10 unit tests for the risk scorer |
| `ui/theme.py` | All CSS injected via `inject_theme()` — tokens, animations, icon-font protection |
| `ui/components.py` | 7 reusable render components (waveform, transcript, risk_ring, …) |

### DB schema additions (v3, non-breaking migrations)

| Table | New columns |
|-------|------------|
| `segments` | `asr_text`, `asr_language`, `asr_confidence`, `no_speech_prob`, `normalized_text` |
| `flags` | `keyword_score`, `semantic_score`, `emotion_score`, `llm_score`, `llm_reason`, `stage_details`, `severity`, `review_status` |
| `call_summaries` | `risk_score` |
| `alerts` | new table — `level`, `category`, `confidence`, `text`, `channels_sent`, `evidence_path` |

All migrations run automatically via `ALTER TABLE … ADD COLUMN` — **existing `callguard.db` files keep all data**.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                         │
│  Analyze Call · Past Calls · Keyword Library · Live Monitor        │
│  Enroll Voice · Manage Voices                                      │
│  ─ Password gate · Diagnostics toggle · Confirm/False-alarm btns   │
└────────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Pipeline Layer                             │
│                                                                    │
│  preprocess_audio.py   pydub + ffmpeg                              │
│    loudnorm (EBU R128) · high-pass 80Hz · silence trim · upsample  │
│         ↓                                                          │
│  transcribe.py         whisperx / faster-whisper                   │
│    detect_language() · task="transcribe" · pyannote diarize 3.1    │
│         ↓                                                          │
│  normalize.py          NFKC · Urdu glyphs · Devanagari nukta       │
│    repeat-collapse · Roman variant map · cross-script overlay      │
│         ↓                                                          │
│  detector.py  — 4-layer fusion                                     │
│    Layer 1  rapidfuzz  WRatio + partial + token_set  sliding win   │
│    Layer 2  semantic   sentence-transformers cosine similarity     │
│    Layer 3  context    benign dampening + speaker escalation       │
│    Layer 4  LLM judge  ollama / anthropic (off by default)         │
│         ↓                                                          │
│  scoring.py            risk_score = 0.7×top + 0.3×second  (0–100)  │
│         ↓                                                          │
│  alerts.py             cooldown · telegram · email · twilio        │
│                                                                    │
│  live.py               sounddevice capture · faster-whisper tiny   │
│                        warning/emergency escalation                │
└────────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                       SQLite Database                              │
│  calls · segments (ASR diagnostics) · flags (stage scores)         │
│  call_summaries (risk_score) · voice_profiles · alerts             │
└────────────────────────────────────────────────────────────────────┘
```

---

## Detection: How the 4 Layers Work

### Layer 1 — Keyword (rapidfuzz, multi-scorer)

Text is first run through the normalizer (producing 1–3 candidate strings: original, Devanagari overlay, Roman overlay). Each string is matched against `keywords.json` using **three scorers in parallel**:

| Scorer | Good for |
|--------|---------|
| `WRatio` | General short-text matching |
| `partial_ratio` | Keyword inside a longer sentence |
| `token_set_ratio` | Out-of-order words |

A **sliding window** of 2–3 consecutive segments is also tested, catching threats that span a segment boundary.

### Layer 2 — Semantic (sentence-transformers)

A multilingual model (`paraphrase-multilingual-MiniLM-L12-v2` by default) computes cosine similarity between the segment and every phrase in `pipeline/phrases.json`. Phrase embeddings are encoded once and cached to disk — no re-encoding on subsequent runs.

This catches **paraphrases with no exact keyword**, e.g. "you won't be around much longer" matching the `threat` category.

### Layer 3 — Context modifier

Applied multiplicatively to the fused score:
- Benign signals in surrounding turns (sports, work, jokes) → −30%
- Benign signals in the flagged segment itself → −20%
- Same speaker with ≥2 medium-confidence flags in context → +10% (escalation)

### Layer 4 — LLM Judge (off by default)

Only called for borderline scores (default 0.40–0.80). Sends the target segment plus the 4 preceding turns. Requires a structured JSON response: `{is_threat, category, severity, confidence, reason}`. 8-second timeout; falls back to rule-based decision on failure.

Enable with `LLM_JUDGE=ollama` or `LLM_JUDGE=anthropic` in `.env`.

### Fusion

```
final_confidence = (WEIGHT_KEYWORD × kw_score + WEIGHT_SEMANTIC × sem_score)
                   / (WEIGHT_KEYWORD + WEIGHT_SEMANTIC)
```

Then the context modifier is applied. If the LLM judge ran, its confidence overrides.

Defaults: `WEIGHT_KEYWORD=0.45`, `WEIGHT_SEMANTIC=0.35`.

### Severity mapping

| Score | Severity | Meaning |
|-------|---------|---------|
| ≥ 0.80 | **High** | Emergency — alert fired |
| 0.60–0.79 | **Medium** | Warning — alert fired |
| 0.50–0.59 | **Low** | Flagged, no alert |
| < 0.50 | — | Not flagged |

---

## Eval Results (v3.0, keyword-only — semantic layer not installed)

```
Precision : 98.6%
Recall    : 93.5%   (target: 80% ✅)
F1        : 96.0%

TP=72  FP=1  FN=5  TN=14

Per language:
  English      P=96.3%  R=83.9%  F1=89.7%
  Hindi (Dev)  P=100%   R=100%   F1=100%
  Hindi (Rom)  P=100%   R=100%   F1=100%
  Urdu (Nas)   P=100%   R=100%   F1=100%
  Urdu (Rom)   P=100%   R=100%   F1=100%
```

Installing `sentence-transformers` will push the 5 English `harm_planning` false negatives (currently sitting at confidence 0.48–0.50) over the flag threshold.

Run the eval yourself:
```bash
python eval/run_eval.py
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
# sudo apt install ffmpeg    # Ubuntu / Debian

# 4. PyTorch CPU
pip install torch==2.1.0 torchaudio==2.1.0

# 5. All dependencies
pip install -r requirements.txt

# 6. Configure
cp .env.example .env
# Edit .env — at minimum set APP_PASSWORD and HF_TOKEN

# 7. Run
streamlit run app.py
```

Open **http://localhost:8501**

---

## Optional Heavy Features

Install these separately — the app falls back gracefully if they are missing.

```bash
# Semantic layer (strongly recommended — boosts recall on paraphrases)
pip install sentence-transformers==3.0.1

# Live monitor mic capture
pip install sounddevice==0.4.7 soundfile==0.12.1

# Noise reduction in audio preprocessing
pip install noisereduce==3.0.2

# LLM judge via Anthropic
pip install anthropic==0.34.0

# Twilio SMS / WhatsApp alerts
pip install twilio==9.3.0

# Faster file watching during development
pip install watchdog
```

---

## Configuration Reference

All settings live in `config.py` and are overridable from `.env` or `st.secrets` (Streamlit Cloud). See `.env.example` for the full list with explanations.

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_PASSWORD` | *(empty)* | Password gate — leave blank to disable locally |
| `HF_TOKEN` | *(required for diarization)* | Hugging Face token |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `COMPUTE_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `WHISPER_LANGUAGE` | *(empty)* | Force language code e.g. `hi`, `ur`, `en` |
| `ASR_ENGINE` | `whisperx` | `whisperx` / `indic` / `deepgram` / `assemblyai` / `google` |
| `USE_DIARIZATION` | `true` | Speaker separation (requires HF_TOKEN) |

### Audio preprocessing

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_LOUDNORM` | `true` | EBU R128 two-pass loudness normalisation |
| `AUDIO_HIGHPASS_HZ` | `80` | High-pass filter cutoff in Hz |
| `AUDIO_NOISE_REDUCE` | `false` | Spectral noise reduction (requires noisereduce) |
| `AUDIO_TRIM_SILENCE` | `true` | Strip leading/trailing silence |

### Detection thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `FUZZY_THRESHOLD` | `80` | 0–100 rapidfuzz score cutoff |
| `FLAG_THRESHOLD` | `0.50` | Minimum fused confidence to flag |
| `WARNING_THRESHOLD` | `0.60` | Confidence for Warning alert |
| `EMERGENCY_THRESHOLD` | `0.80` | Confidence for Emergency alert |
| `WEIGHT_KEYWORD` | `0.45` | Keyword layer fusion weight |
| `WEIGHT_SEMANTIC` | `0.35` | Semantic layer fusion weight |
| `CONTEXT_WINDOW_SIZE` | `4` | Prior segments used for context analysis |
| `LLM_JUDGE` | `off` | `off` / `ollama` / `anthropic` |

### Alerts

| Variable | Default | Description |
|----------|---------|-------------|
| `ALERT_COOLDOWN_S` | `60` | Per-category cooldown between alerts (seconds) |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | Telegram bot token |
| `TELEGRAM_CHAT_ID` | *(empty)* | Telegram chat/channel ID |
| `SMTP_HOST` | *(empty)* | SMTP server for email alerts |
| `SMTP_TO` | *(empty)* | Alert recipient email address |

### Live monitor

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_MODEL` | `base` | faster-whisper model for live transcription |
| `LIVE_CHUNK_SEC` | `4` | Capture chunk size in seconds |
| `LIVE_OVERLAP_SEC` | `1` | Overlap between chunks |
| `LIVE_WARN_WINDOW_SEC` | `30` | Window for double-warning → emergency escalation |

---

## Project Structure

```
callguard/
│
├── app.py                      # Streamlit dashboard — 6 pages
│                               #   Analyze Call  · Past Calls
│                               #   Keyword Library · Live Monitor
│                               #   Enroll Voice  · Manage Voices
│
├── config.py                   # Centralized Config dataclass
│
├── pipeline/
│   ├── preprocess_audio.py     # loudnorm · highpass · noise reduce · trim
│   ├── transcribe.py           # whisperx + diarization wrapper
│   ├── normalize.py            # Unicode · Urdu glyphs · Roman↔Devanagari
│   ├── detector.py             # 4-layer detection + fusion + stage_details
│   ├── semantic.py             # sentence-transformers cosine scorer
│   ├── scoring.py              # risk_score() 0–100
│   ├── alerts.py               # Telegram · email · Twilio · cooldown
│   ├── live.py                 # Mic capture + live detection loop
│   ├── voice_recognition.py    # resemblyzer enrollment + matching
│   ├── keywords.json           # Editable threat keywords — EN, HI, UR
│   └── phrases.json            # Semantic phrase bank — EN, HI, UR, Hinglish
│
├── db/
│   ├── models.py               # ORM: Call · Segment · Flag · CallSummary
│   │                           #       VoiceProfile · Alert
│   └── database.py             # Session management · CRUD · auto-migrations
│
├── ui/
│   ├── theme.py                # inject_theme() — all CSS, design tokens
│   └── components.py          # waveform · transcript · risk_ring · …
│
├── eval/
│   ├── labelled_segments.csv   # 92 labelled rows for evaluation
│   ├── run_eval.py             # Precision / Recall / F1 harness
│   └── report.json             # Last eval run output
│
├── tests/
│   ├── test_normalize.py       # 13 normalizer tests
│   ├── test_detector.py        # 23 detector tests
│   └── test_scoring.py        # 10 scoring tests
│
├── .streamlit/
│   └── config.toml             # Dark-green theme tokens
│
├── sample_audio/               # Drop test recordings here
├── requirements.txt
├── packages.txt
├── .env.example                # Full config reference with explanations
└── README.md
```

---

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
# 48 passed
```

---

## Extending the Classifier

The swappable `analyze_segment()` interface is unchanged. Drop in any classifier:

```python
# pipeline/detector.py — replace only this function body
def analyze_segment(segment: dict, context_window: list[dict]) -> dict:
    """
    Your custom classifier here.

    Input : segment dict + list of preceding segments
    Output: {
      flag, category, confidence, matched_keyword,
      keyword_score, semantic_score, llm_score, llm_reason,
      severity, stage_details
    }
    """
    pass
```

Nothing else in the codebase needs to change.

---

## Diagnostics Panel

Toggle **Show diagnostics** in the sidebar to reveal per-segment detail for every transcript row:

- Raw ASR text and detected language
- ASR confidence (avg_logprob) and no_speech_prob
- Keyword score, semantic score, fused score, LLM score
- Context modifier reason
- Flag threshold used

Use this to understand exactly why a segment was or was not flagged.

---

## Review Queue & Eval Export

Every flagged segment in the transcript has **Confirm** and **False alarm** buttons. Clicking either:
1. Sets `flags.review_status` in the database
2. Appends the segment to `eval/labelled_segments.csv` automatically

This means the eval dataset grows from real-world use and `python eval/run_eval.py` always reflects your latest reviewed data.

---

## Live Monitor

The **Live Monitor** page captures microphone audio in configurable chunks (default 4 s with 1 s overlap), transcribes each chunk with faster-whisper, and runs the full detection pipeline in a background thread. The UI polls the result queue without blocking.

Alert levels:
- **Warning** — single flag at confidence ≥ 0.60
- **Emergency** — confidence ≥ 0.80, OR two warnings within 30 seconds

⚠ Only monitor audio you have explicit permission to record.

---

## Password Gate

Set `APP_PASSWORD` in `.env` to protect the app when exposed publicly via ngrok. Leave it blank for local development.

---

## Known Limitations (v3)

- **CPU latency** — transcription takes roughly 1× real time for `base` model. Live monitor latency is 2–6 s on CPU.
- **Semantic layer requires install** — `pip install sentence-transformers`. App works without it (keyword-only, 93.5% recall).
- **LLM judge adds latency** — only call it for borderline scores; keep it off unless you have a local Ollama instance.
- **SQLite is single-writer** — fine for one analyst, not for multi-user production.
- **Live monitor requires sounddevice** — `pip install sounddevice soundfile`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Dashboard | Streamlit 1.58 |
| Speech-to-text | whisperx + faster-whisper |
| Speaker diarization | pyannote.audio 3.1 |
| Voice recognition | resemblyzer (256-dim GE2E) |
| Audio preprocessing | pydub + ffmpeg |
| Text normalisation | unicodedata + regex (custom) |
| Keyword matching | rapidfuzz (WRatio + partial + token_set) |
| Semantic matching | sentence-transformers (optional) |
| LLM judge | ollama / anthropic (optional) |
| Live capture | sounddevice + faster-whisper |
| Alerts | Telegram Bot API · smtplib · Twilio (optional) |
| Database | SQLAlchemy + SQLite |
| Evaluation | Custom harness — precision / recall / F1 |
| Tests | pytest (48 tests) |
| Public tunnel | ngrok |

---

## Roadmap (Post v3)

- [ ] Indic ASR engine (AI4Bharat IndicConformer) for higher Hindi/Urdu accuracy
- [ ] Word-level timestamp alignment for precise waveform flag markers
- [ ] Fine-tuned multilingual classifier to replace/augment keyword layer
- [ ] PDF report export + CSV flag log
- [ ] Multi-user support with role-based access control
- [ ] Punjabi, Arabic, Bengali keyword + phrase coverage
- [ ] GPU compute_type float16 for production-speed transcription
- [ ] Streamlit-webrtc for browser-based live monitoring

---

## Repository

**GitHub:** https://github.com/prithvirajsingh3108-ctrl/Call-guard  
**Version:** 3.0  
**Build date:** September 2026  
**Status:** Prototype — not production ready  

---

*Built as an intern prototype project. Prioritises working end-to-end functionality over scale or polish. Every ML component is behind a swappable interface — replace any layer without touching the surrounding pipeline.*
