"""
app.py
──────
CallGuard Dashboard — premium dark-green UI (all phases)

Pages
-----
1. Analyze Call      — upload, pipeline, diagnostics, results
2. Past Calls        — browse / open / delete with risk pills
3. Keyword Library   — chips + add form per language
4. Live Monitor      — real-time mic capture with alerts
5. Enroll a Voice    — voice profile enrollment
6. Manage Voices     — delete enrolled profiles

Run with:
    source .venv/bin/activate && streamlit run app.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# Page config must be the FIRST Streamlit call
st.set_page_config(
    page_title="CallGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import cfg

def _load_secrets():
    """Merge st.secrets into os.environ for Streamlit Cloud."""
    try:
        for k in ["HF_TOKEN","WHISPER_MODEL","COMPUTE_DEVICE","DATABASE_PATH",
                  "KEYWORDS_PATH","FUZZY_THRESHOLD","CONTEXT_WINDOW_SIZE","APP_PASSWORD"]:
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass

_load_secrets()

from ui.theme import inject_theme
inject_theme()

from db.database import (
    init_db, get_session,
    create_call, update_call_status, save_call_results,
    get_all_calls, get_call_by_id,
    get_segments_for_call, get_flags_for_call, get_summary_for_call,
    delete_call, update_flag_review, get_recent_alerts,
    save_voice_profile, get_all_voice_profiles, delete_voice_profile,
)
from pipeline.scoring import risk_score as compute_risk_score
from pipeline.detector import detect_threats, summarize_flags, invalidate_keyword_cache
from pipeline.normalize import normalize

from ui.components import (
    waveform, transcript, risk_ring,
    category_bars, flag_timeline, speaker_share,
    render_pipeline_steps,
    sidebar_brand, sidebar_engine_card, dropzone_empty_state,
    risk_pill, toast, fmt_time,
)

init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Password gate
# ─────────────────────────────────────────────────────────────────────────────

def _password_gate() -> bool:
    """
    Show a login screen if APP_PASSWORD is set.
    Returns True when the user is authenticated (or no password is set).
    """
    pwd = cfg.APP_PASSWORD
    if not pwd:
        return True  # no gate in local dev

    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        '<div style="max-width:380px;margin:6rem auto">'
        '<div class="cg-card" style="text-align:center;padding:2.5rem">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:\'Instrument Serif\',Georgia,serif;font-size:2rem;'
        'color:var(--text);margin-bottom:.25rem">CallGuard</div>'
        '<div style="color:var(--mute);font-size:.85rem;margin-bottom:1.5rem">'
        'Enter your access password to continue.</div>',
        unsafe_allow_html=True,
    )
    entered = st.text_input("Password", type="password", label_visibility="collapsed",
                            placeholder="Password")
    if st.button("Unlock", type="primary", use_container_width=True):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown("</div></div>", unsafe_allow_html=True)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Cached models
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_whisperx_model():
    import whisperx, warnings; warnings.filterwarnings("ignore")
    return whisperx.load_model(
        cfg.WHISPER_MODEL, cfg.COMPUTE_DEVICE, compute_type="float32"
    )

@st.cache_resource(show_spinner=False)
def _get_diarize_model():
    import warnings; warnings.filterwarnings("ignore")
    if not cfg.HF_TOKEN:
        return None
    try:
        from whisperx.diarize import DiarizationPipeline
        return DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=cfg.HF_TOKEN,
            device=cfg.COMPUTE_DEVICE,
        )
    except Exception as e:
        print(f"[app] Diarization model failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Keyword helpers
# ─────────────────────────────────────────────────────────────────────────────

KEYWORDS_PATH = Path(cfg.KEYWORDS_PATH)

def _load_keywords() -> dict:
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_keywords(data: dict) -> None:
    with open(KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    invalidate_keyword_cache()

def _detect_script(word: str) -> str:
    for ch in word:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F: return "hi"
        if 0x0600 <= cp <= 0x06FF: return "ur"
    return "en"


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics panel helper
# ─────────────────────────────────────────────────────────────────────────────

def _render_diagnostics(seg: dict) -> None:
    """Render per-segment diagnostic details in an expander."""
    d = seg.get("stage_details") or {}
    if not d:
        st.caption("No stage details stored.")
        return

    cols = st.columns(4)
    cols[0].metric("Keyword",  f"{d.get('keyword_score', 0):.0%}")
    cols[1].metric("Semantic", f"{d.get('semantic_score', 0):.0%}")
    cols[2].metric("Fused",    f"{d.get('fused_score', 0):.0%}")
    cols[3].metric("Threshold",f"{d.get('flag_threshold', cfg.FLAG_THRESHOLD):.0%}")

    st.markdown(
        f"**Keyword match:** `{d.get('keyword_match','—')}` "
        f"({d.get('keyword_category','—')})  \n"
        f"**Semantic nearest:** _{d.get('nearest_phrase','—')}_ "
        f"({d.get('semantic_category','—')})  \n"
        f"**Context modifier:** `{d.get('context_reason','—')}`  \n"
        f"**LLM called:** {'Yes → ' + str(d.get('llm_score','')) if d.get('llm_called') else 'No'}  \n"
        f"**ASR text:** _{seg.get('asr_text') or seg.get('text','—')}_  \n"
        f"**Language:** `{seg.get('asr_language','—')}`  \n"
        f"**ASR confidence:** `{seg.get('asr_confidence','—')}`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export-to-eval helper
# ─────────────────────────────────────────────────────────────────────────────

EVAL_CSV = Path("eval/labelled_segments.csv")

def _export_to_eval(text: str, language: str, label: int, category: str) -> None:
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EVAL_CSV.exists()
    with open(EVAL_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["text", "language", "label", "category"])
        w.writerow([text, language, label, category])


# ─────────────────────────────────────────────────────────────────────────────
# Results layout (shared by Analyze and Past Calls)
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(
    enriched: list[dict],
    summary: dict,
    duration_sec: float,
    wav_path: str | None = None,
    show_diag: bool = False,
    call_id: int | None = None,
) -> None:
    flags_list = [s for s in enriched if s.get("flag")]
    left, right = st.columns([6, 4], gap="large")

    with left:
        # Waveform card
        st.markdown('<div class="cg-card" style="padding:1rem 1.25rem 0.75rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:8px">Waveform</div>', unsafe_allow_html=True)
        waveform(wav_path, flags_list, duration_sec)
        if duration_sec:
            st.markdown(f'<div style="font-size:.75rem;color:var(--dim);text-align:right;margin-top:2px">{fmt_time(duration_sec)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Transcript card
        st.markdown('<div class="cg-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:12px">Transcript</div>', unsafe_allow_html=True)

        for seg in enriched:
            _render_transcript_row(seg, show_diag=show_diag, call_id=call_id)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        score = compute_risk_score(flags_list)

        # Risk card
        st.markdown('<div class="cg-card" style="padding:1rem 1.25rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:4px">Call Risk</div>', unsafe_allow_html=True)
        risk_ring(score)
        c1, c2, c3 = st.columns(3)
        c1.metric("Segments", summary["total_segments"])
        c2.metric("Flags",    summary["total_flags"])
        c3.metric("Peak",     f"{summary['highest_confidence']:.0%}" if summary["highest_confidence"] else "—")
        st.markdown("</div>", unsafe_allow_html=True)

        # Category bars
        st.markdown('<div class="cg-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:12px">Flags by Category</div>', unsafe_allow_html=True)
        category_bars(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        # Timeline
        st.markdown('<div class="cg-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:12px">Flags Over the Call</div>', unsafe_allow_html=True)
        flag_timeline(flags_list, duration_sec)
        st.markdown("</div>", unsafe_allow_html=True)

        # Speaker share
        if enriched:
            st.markdown('<div class="cg-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:10px">Speaker Share</div>', unsafe_allow_html=True)
            speaker_share(enriched)
            st.markdown("</div>", unsafe_allow_html=True)


def _render_transcript_row(seg: dict, show_diag: bool = False, call_id: int | None = None) -> None:
    """Render one transcript row with optional diagnostics + review buttons."""
    import html as _html
    from ui.components import fmt_time, EM, ALERT, MUTE, DIM, SURFACE, SURFACE2, LINE

    ts   = f"{fmt_time(seg.get('start', seg.get('start_sec', 0)))} – {fmt_time(seg.get('end', seg.get('end_sec', 0)))}"
    ms   = seg.get("match_status", "not_run")
    mn   = seg.get("matched_name", "")
    mc   = seg.get("match_confidence", 0)
    if ms == "matched" and mn:
        spk = f"{mn} ({mc:.0%})"
    else:
        spk = seg.get("speaker_display") or seg.get("speaker", "UNKNOWN")

    text = seg.get("text", "")
    cat  = seg.get("category", "")
    kw   = seg.get("matched_keyword", "")
    conf = seg.get("confidence", 0.0)

    if seg.get("flag"):
        from ui.components import CATEGORY_LABELS
        cat_label = CATEGORY_LABELS.get(cat, cat)
        with st.expander(
            f"⚠  {fmt_time(seg.get('start', 0))}  ·  {spk}  —  {text[:65]}{'…' if len(text)>65 else ''}",
            expanded=False,
        ):
            # Highlight keyword in text
            safe_text = _html.escape(text)
            if kw:
                safe_kw  = _html.escape(kw)
                safe_text = safe_text.replace(safe_kw, f'<span class="cg-kw">{safe_kw}</span>', 1)

            st.markdown(
                f'<div class="cg-seg cg-seg-flagged">'
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">'
                f'<span class="cg-ts">{ts}</span>'
                f'<span class="cg-spk">{spk}</span>'
                f'<span class="cg-pill cg-pill-alert">{cat_label}</span>'
                f'<span class="cg-pill cg-pill-mute">{conf:.0%} conf · {seg.get("severity","Medium")}</span>'
                f'</div>'
                f'<div style="margin-bottom:8px">{safe_text}</div>'
                f'<div style="font-size:.8rem;color:var(--mute)">Keyword: '
                f'<code style="background:var(--surface2);padding:1px 6px;border-radius:4px">{_html.escape(kw)}</code>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Context window
            ctx = seg.get("context_window", [])
            if ctx:
                st.markdown(f'<div style="font-size:.78rem;color:{MUTE};font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-top:10px">Context window</div>', unsafe_allow_html=True)
                for c in ctx[:4]:
                    c_spk  = c.get("speaker_display") or c.get("speaker", "")
                    c_text = _html.escape(c.get("text", ""))
                    st.markdown(f'<div style="padding:6px 10px;border-left:2px solid {LINE};margin:4px 0;font-size:.85rem;color:{MUTE}"><span style="color:{EM};font-weight:600">{c_spk}</span>&ensp;{c_text}</div>', unsafe_allow_html=True)

            # Diagnostics
            if show_diag:
                st.markdown("---")
                st.markdown("**Diagnostics**")
                _render_diagnostics(seg)

            # Review buttons + export
            flag_id = seg.get("flag_id")
            review  = seg.get("review_status", "pending")

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
            confirmed_key   = f"confirm_{flag_id or id(seg)}"
            falsealarm_key  = f"falarm_{flag_id or id(seg)}"
            export_key      = f"export_{flag_id or id(seg)}"

            if btn_col1.button("✓ Confirm", key=confirmed_key,
                               type="primary" if review != "confirmed" else "secondary"):
                if flag_id:
                    with get_session() as s:
                        update_flag_review(s, flag_id, "confirmed")
                _export_to_eval(text, seg.get("asr_language","en"), 1, cat)
                toast("Confirmed and added to eval set.", "success")
                st.rerun()

            if btn_col2.button("✗ False alarm", key=falsealarm_key):
                if flag_id:
                    with get_session() as s:
                        update_flag_review(s, flag_id, "false_alarm")
                _export_to_eval(text, seg.get("asr_language","en"), 0, "")
                toast("Marked as false alarm and added to eval set.", "info")
                st.rerun()

    else:
        import html as _html
        safe_text = _html.escape(text)
        st.markdown(
            f'<div class="cg-seg">'
            f'<span class="cg-ts">{ts}</span>&ensp;'
            f'<span class="cg-spk">{spk}</span>'
            f'<br/>{safe_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if show_diag and seg.get("stage_details"):
            with st.expander("Diagnostics", expanded=False):
                _render_diagnostics(seg)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Analyze
# ─────────────────────────────────────────────────────────────────────────────

def page_analyze():
    show_diag = st.session_state.get("show_diagnostics", cfg.SHOW_DIAGNOSTICS_DEFAULT)

    st.markdown(
        '<h1 style="margin-bottom:4px">Hear the risk in every call.</h1>'
        '<p style="color:var(--mute);font-size:1rem;margin-top:0;margin-bottom:1.5rem">'
        'Upload a recording. CallGuard transcribes it, separates the speakers '
        'and flags threats in English, Hindi and Urdu.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("analysis_done"):
        _render_results(
            st.session_state["analysis_enriched"],
            st.session_state["analysis_summary"],
            st.session_state["analysis_duration"],
            st.session_state.get("analysis_wav_path"),
            show_diag=show_diag,
            call_id=st.session_state.get("analysis_call_id"),
        )
        if st.button("Analyze another call", type="primary"):
            for k in ["analysis_done","analysis_enriched","analysis_summary",
                      "analysis_duration","analysis_wav_path","analysis_call_id"]:
                st.session_state.pop(k, None)
            st.rerun()
        return

    # Cloud ASR warning
    if cfg.cloud_asr_active:
        st.warning(f"⚠ Cloud ASR active ({cfg.ASR_ENGINE}) — audio is sent to a third-party service.")

    # Dropzone
    with st.container(key="dropzone"):
        st.markdown(
            """
<div style="display:flex;align-items:flex-end;justify-content:center;gap:5px;height:52px;margin-bottom:1.25rem" aria-hidden="true">
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:28px;animation:breathe 1.6s ease-in-out infinite;animation-delay:0s"></div>
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:40px;animation:breathe 1.6s ease-in-out infinite;animation-delay:.12s"></div>
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:52px;animation:breathe 1.6s ease-in-out infinite;animation-delay:.24s"></div>
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:44px;animation:breathe 1.6s ease-in-out infinite;animation-delay:.36s"></div>
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:36px;animation:breathe 1.6s ease-in-out infinite;animation-delay:.48s"></div>
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:46px;animation:breathe 1.6s ease-in-out infinite;animation-delay:.60s"></div>
  <div class="cg-bar" style="width:5px;border-radius:3px;background:var(--em);height:30px;animation:breathe 1.6s ease-in-out infinite;animation-delay:.72s"></div>
</div>
<div style="font-size:1rem;color:var(--text);font-weight:500;margin-bottom:4px">Drop an audio file here</div>
<div style="font-size:.82rem;color:var(--mute);margin-bottom:1rem">mp3 &middot; m4a &middot; wav &middot; flac &middot; ogg</div>
""",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
            label_visibility="collapsed",
        )

    if not uploaded:
        return

    st.audio(uploaded)
    use_diarization = st.checkbox(
        "Enable speaker diarization (requires HF_TOKEN)",
        value=cfg.has_hf_token,
    )
    if not st.button("Run Analysis", type="primary"):
        return

    # Save temp file
    tmp_dir  = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / uploaded.name
    tmp_path.write_bytes(uploaded.getbuffer())

    with get_session() as s:
        call    = create_call(s, filename=uploaded.name, file_path=str(tmp_path))
        call_id = call.id

    pipe_ph   = st.empty()
    status_ph = st.empty()
    progress  = st.progress(0)
    render_pipeline_steps(pipe_ph, 0)
    wav_path = None

    try:
        # Step 0 — Preprocess
        status_ph.markdown('<span style="color:var(--mute);font-size:.88rem">Preprocessing audio…</span>', unsafe_allow_html=True)
        with get_session() as s:
            update_call_status(s, call_id, "transcribing")
        from pipeline.preprocess_audio import convert_to_wav
        wav_path = convert_to_wav(str(tmp_path))
        try:
            from pydub import AudioSegment as _AS
            duration_sec = len(_AS.from_file(wav_path)) / 1000.0
        except Exception:
            duration_sec = 0.0
        progress.progress(20)

        # Step 1 — Transcribe
        render_pipeline_steps(pipe_ph, 1)
        status_ph.markdown('<span style="color:var(--mute);font-size:.88rem">Transcribing…</span>', unsafe_allow_html=True)
        os.environ["USE_DIARIZATION"] = "true" if use_diarization else "false"

        if use_diarization:
            import whisperx, warnings; warnings.filterwarnings("ignore")
            from whisperx.diarize import assign_word_speakers
            wx_model = _get_whisperx_model()
            audio    = whisperx.load_audio(wav_path)
            forced_lang = cfg.WHISPER_LANGUAGE.strip() or None
            if forced_lang:
                detected_language, lang_confidence = forced_lang, 1.0
            else:
                try:
                    detect_audio = audio[:30*16000] if len(audio) > 30*16000 else audio
                    lang_code, lang_confidence, _ = wx_model.model.detect_language(detect_audio)
                    detected_language = lang_code
                    if lang_confidence < 0.6 and len(audio) > 30*16000:
                        detected_language, lang_confidence, _ = wx_model.model.detect_language(audio)
                except Exception:
                    detected_language, lang_confidence = "en", 0.0
            result = wx_model.transcribe(audio, batch_size=16, language=detected_language, task="transcribe")
            try:
                dm = _get_diarize_model()
                if dm:
                    result = assign_word_speakers(dm(wav_path), result)
            except Exception as exc:
                print(f"[app] Diarization skipped: {exc}")
            raw = result.get("segments",[]) if isinstance(result,dict) else result.segments
            segments = []
            for seg in raw:
                d = seg if isinstance(seg, dict) else vars(seg)
                segments.append({
                    "speaker":        d.get("speaker","UNKNOWN") or "UNKNOWN",
                    "text":           str(d.get("text","")).strip(),
                    "start":          round(float(d.get("start",0.0)),3),
                    "end":            round(float(d.get("end",0.0)),3),
                    "asr_text":       str(d.get("text","")).strip(),
                    "asr_language":   detected_language,
                    "asr_confidence": d.get("avg_logprob"),
                    "no_speech_prob": d.get("no_speech_prob"),
                })
        else:
            from pipeline.transcribe import transcribe_plain
            raw_segs = transcribe_plain(wav_path)
            segments = [{**s, "asr_text": s["text"], "asr_language": "unknown"} for s in raw_segs]

        progress.progress(55)

        # Voice recognition
        try:
            from pipeline.voice_recognition import identify_speakers, apply_speaker_names
            name_map = identify_speakers(segments, wav_path)
            segments = apply_speaker_names(segments, name_map)
        except Exception:
            for seg in segments:
                seg.setdefault("speaker_display", seg.get("speaker","UNKNOWN"))

        # Step 2 — Detect
        render_pipeline_steps(pipe_ph, 2)
        status_ph.markdown('<span style="color:var(--mute);font-size:.88rem">Analyzing for threats…</span>', unsafe_allow_html=True)
        with get_session() as s:
            update_call_status(s, call_id, "analyzing")
        enriched = detect_threats(segments)
        summary  = summarize_flags(enriched)
        progress.progress(80)

        # Step 3 — Save
        render_pipeline_steps(pipe_ph, 3)
        status_ph.markdown('<span style="color:var(--mute);font-size:.88rem">Saving to database…</span>', unsafe_allow_html=True)
        flags_list = [s for s in enriched if s.get("flag")]
        score      = compute_risk_score(flags_list)
        with get_session() as s:
            save_call_results(s, call_id, enriched, summary, duration_sec, risk_score=score)
            update_call_status(s, call_id, "done")

        # Attach flag IDs so review buttons work
        with get_session() as s:
            db_flags = get_flags_for_call(s, call_id)
            flag_map = {}
            for fl in db_flags:
                flag_map[fl.timestamp_sec] = fl.id
        for seg in enriched:
            if seg.get("flag"):
                seg["flag_id"] = flag_map.get(seg.get("start", 0.0))

        progress.progress(100)
        render_pipeline_steps(pipe_ph, 4)
        status_ph.empty()

        fc = summary["total_flags"]
        toast(
            f"Analysis complete — {fc} flag(s) in {len(segments)} segments. Risk: {score}/100.",
            "info" if fc else "success"
        )

        st.session_state.update({
            "analysis_done":     True,
            "analysis_enriched": enriched,
            "analysis_summary":  summary,
            "analysis_duration": duration_sec,
            "analysis_wav_path": wav_path,
            "analysis_call_id":  call_id,
        })
        st.divider()
        _render_results(enriched, summary, duration_sec, wav_path, show_diag=show_diag, call_id=call_id)

    except Exception as exc:
        progress.empty(); pipe_ph.empty()
        status_ph.error(f"Error: {exc}")
        with get_session() as s:
            update_call_status(s, call_id, "error", str(exc))
        st.exception(exc)
    finally:
        try:
            if tmp_path.exists(): tmp_path.unlink()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Past Calls
# ─────────────────────────────────────────────────────────────────────────────

def _build_enriched_from_db(call_id: int) -> tuple[list[dict], dict, float]:
    """Load call data from DB and return (enriched_segments, summary_dict, duration)."""
    with get_session() as s:
        call     = get_call_by_id(s, call_id)
        summary  = get_summary_for_call(s, call_id)
        segments = get_segments_for_call(s, call_id)
        flags    = get_flags_for_call(s, call_id)

    flag_map: dict[int, object] = {fl.segment_id: fl for fl in flags}
    enriched = []
    for seg in segments:
        fl = flag_map.get(seg.id)
        enriched.append({
            "start":          seg.start_sec,
            "end":            seg.end_sec,
            "speaker":        seg.speaker,
            "speaker_display":seg.matched_name if (seg.match_status=="matched" and seg.matched_name) else seg.speaker,
            "match_status":   seg.match_status or "not_run",
            "matched_name":   seg.matched_name,
            "match_confidence":seg.match_confidence or 0.0,
            "text":           seg.text,
            "asr_text":       seg.asr_text,
            "asr_language":   seg.asr_language,
            "asr_confidence": seg.asr_confidence,
            "no_speech_prob": seg.no_speech_prob,
            "flag":           fl is not None,
            "category":       fl.category          if fl else "",
            "matched_keyword":fl.matched_keyword   if fl else "",
            "confidence":     fl.confidence        if fl else 0.0,
            "keyword_score":  fl.keyword_score     if fl else 0.0,
            "semantic_score": fl.semantic_score    if fl else 0.0,
            "llm_score":      fl.llm_score         if fl else None,
            "llm_reason":     fl.llm_reason        if fl else None,
            "severity":       fl.severity          if fl else "Low",
            "review_status":  fl.review_status     if fl else "pending",
            "stage_details":  fl.stage_details     if fl else {},
            "context_window": fl.context_window    if fl else [],
            "flag_id":        fl.id                if fl else None,
        })

    by_cat  = summary.by_category or {} if summary else {}
    sum_dict = {
        "total_segments":    summary.total_segments if summary else len(segments),
        "total_flags":       summary.total_flags    if summary else 0,
        "by_category":       by_cat,
        "highest_confidence":summary.highest_confidence if summary else 0.0,
        "flagged_segments":  [e for e in enriched if e["flag"]],
    }
    duration = call.duration_sec if call else 0.0
    return enriched, sum_dict, duration or 0.0


def page_past_calls():
    show_diag = st.session_state.get("show_diagnostics", False)

    st.markdown(
        '<h1 style="margin-bottom:4px">Past Calls</h1>'
        '<p style="color:var(--mute);font-size:1rem;margin-top:0;margin-bottom:1.5rem">'
        'Every analyzed recording, with its transcript and flags kept in the database.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("past_call_selected"):
        cid = st.session_state["past_call_selected"]
        if st.button("← Back to all calls"):
            st.session_state.pop("past_call_selected")
            st.rerun()
        st.divider()
        enriched, sum_dict, dur = _build_enriched_from_db(cid)
        with get_session() as s:
            call = get_call_by_id(s, cid)
            summary = get_summary_for_call(s, cid)
        st.markdown(
            f'<h2 style="margin-bottom:2px">{call.filename if call else ""}</h2>'
            f'<div style="color:var(--mute);font-size:.85rem;margin-bottom:1.5rem">'
            f'#{cid} · {call.completed_at.strftime("%d %b %Y, %H:%M") if call and call.completed_at else "—"}'
            f' · {fmt_time(dur)} · {sum_dict["total_flags"]} flag(s) · '
            f'{risk_pill(summary.risk_score or 0 if summary else 0)}'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_results(enriched, sum_dict, dur, show_diag=show_diag, call_id=cid)
        st.markdown('<div style="margin-top:2rem">', unsafe_allow_html=True)
        if st.button("Delete this call", key=f"del_{cid}"):
            with get_session() as s:
                delete_call(s, cid)
            toast(f"Call #{cid} deleted.", "success")
            st.session_state.pop("past_call_selected")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with get_session() as s:
        calls = get_all_calls(s)
        if not calls:
            st.markdown('<div class="cg-card" style="text-align:center;padding:3rem 2rem;color:var(--mute)">No calls analyzed yet.</div>', unsafe_allow_html=True)
            return
        rows = []
        for c in calls:
            sm = get_summary_for_call(s, c.id)
            rows.append({"_id": c.id, "Recording": c.filename,
                         "Analyzed": c.completed_at.strftime("%d %b %Y, %H:%M") if c.completed_at else "—",
                         "Length": fmt_time(c.duration_sec) if c.duration_sec else "—",
                         "Flags": sm.total_flags if sm else 0,
                         "_score": sm.risk_score or 0 if sm else 0})

    for row in rows:
        with st.container():
            st.markdown(
                f'<div class="cg-card" style="padding:1rem 1.5rem;margin-bottom:.75rem">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">'
                f'<div><div style="font-weight:600;font-size:.95rem">{row["Recording"]}</div>'
                f'<div style="font-size:.78rem;color:var(--mute);margin-top:2px">'
                f'{row["Analyzed"]} · {row["Length"]} · {row["Flags"]} flag(s)</div></div>'
                f'<div>{risk_pill(row["_score"])}</div></div></div>',
                unsafe_allow_html=True,
            )
            co, cd, _ = st.columns([1,1,8])
            if co.button("Open",   key=f"open_{row['_id']}", type="primary"):
                st.session_state["past_call_selected"] = row["_id"]
                st.rerun()
            if cd.button("Delete", key=f"del_list_{row['_id']}"):
                with get_session() as s:
                    delete_call(s, row["_id"])
                toast(f"Call #{row['_id']} deleted.", "success")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Keyword Library
# ─────────────────────────────────────────────────────────────────────────────

def page_keyword_library():
    st.markdown(
        '<h1 style="margin-bottom:4px">Keyword Library</h1>'
        '<p style="color:var(--mute);font-size:1rem;margin-top:0;margin-bottom:1.5rem">'
        'The phrases the detector matches, including misspellings and Roman-script Hindi and Urdu.</p>',
        unsafe_allow_html=True,
    )
    kw_data = _load_keywords()
    CATS    = ["threat", "abuse", "harm_planning", "disaster"]

    tab_en, tab_hi, tab_ur = st.tabs(["English", "Hindi", "Urdu"])
    for tab, script in [(tab_en,"en"), (tab_hi,"hi"), (tab_ur,"ur")]:
        with tab:
            for cat in CATS:
                words = [w for w in kw_data.get(cat, []) if _detect_script(w) == script]
                if not words:
                    continue
                st.markdown(
                    f'<div class="cg-card" style="margin-bottom:1rem">'
                    f'<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:10px">'
                    f'{cat.replace("_"," ").title()} <span style="color:var(--dim)">({len(words)})</span></div>'
                    + "".join(f'<span class="cg-kw-chip">{w}</span>' for w in words)
                    + '</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="cg-card" style="margin-top:1.5rem">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:14px">Add Keyword</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 2, 1])
    new_word = c1.text_input("Keyword", placeholder="e.g. khatam kar dunga", label_visibility="collapsed")
    category = c2.selectbox("Category", CATS, label_visibility="collapsed")
    if c3.button("Add", type="primary"):
        nw = new_word.strip()
        if not nw:
            toast("Please enter a keyword.", "warning")
        else:
            kw_data = _load_keywords()
            if nw not in kw_data.get(category, []):
                kw_data.setdefault(category, []).append(nw)
                _save_keywords(kw_data)
                toast(f'Added "{nw}" to {category}.', "success")
            else:
                toast(f'"{nw}" already in {category}.', "info")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Live Monitor
# ─────────────────────────────────────────────────────────────────────────────

def page_live_monitor():
    st.markdown(
        '<h1 style="margin-bottom:4px">Live Monitor</h1>'
        '<p style="color:var(--mute);font-size:1rem;margin-top:0;margin-bottom:1rem">'
        'Real-time threat detection from your microphone. Latency is typically '
        '2–6 seconds on CPU.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "⚠ **Consent notice** — Only monitor audio you have explicit permission to record. "
        "CallGuard processes all audio locally on this machine."
    )

    # Alert banner
    alert_level = st.session_state.get("alert_level")
    if alert_level == "emergency":
        st.error(
            f"🚨 **EMERGENCY** — {st.session_state.get('alert_category','').upper()} detected "
            f"({st.session_state.get('alert_confidence',0):.0%} confidence)  \n"
            f"_{st.session_state.get('alert_text','')[:200]}_"
        )
    elif alert_level == "warning":
        st.warning(
            f"⚠ **Warning** — {st.session_state.get('alert_category','').upper()} "
            f"({st.session_state.get('alert_confidence',0):.0%})"
        )

    # Check dependencies
    try:
        import sounddevice  # noqa
        sd_ok = True
    except ImportError:
        sd_ok = False
        st.error("sounddevice not installed. Run: `pip install sounddevice soundfile`")

    try:
        from faster_whisper import WhisperModel  # noqa
        fw_ok = True
    except ImportError:
        fw_ok = False
        st.warning("faster-whisper not installed — live transcription unavailable. "
                   "Run: `pip install faster-whisper`")

    # Session state
    if "live_session" not in st.session_state:
        st.session_state["live_session"] = None
    if "live_results" not in st.session_state:
        st.session_state["live_results"] = []

    session = st.session_state["live_session"]
    is_running = session is not None and session.is_running

    col_start, col_stop, col_test = st.columns([1, 1, 2])

    with col_start:
        start_disabled = is_running or not (sd_ok and fw_ok)
        if st.button("▶ Start", type="primary", disabled=start_disabled):
            from pipeline.live import start_session
            st.session_state["live_session"] = start_session()
            st.session_state["live_results"] = []
            st.session_state.pop("alert_level", None)
            st.rerun()

    with col_stop:
        if st.button("■ Stop", disabled=not is_running):
            from pipeline.live import stop_session
            stop_session(st.session_state["live_session"])
            st.session_state["live_session"] = None
            st.rerun()

    with col_test:
        if st.button("Send test alert"):
            from pipeline.alerts import send_test_alert
            channels = send_test_alert()
            toast(f"Test alert sent via: {', '.join(channels)}", "info")

    # Status indicator
    if is_running:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:1rem 0">'
            '<span class="cg-engine-dot"></span>'
            '<span style="color:var(--em);font-weight:600">Listening…</span>'
            '<span style="color:var(--mute);font-size:.85rem">(~2–6 s latency on CPU)</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Poll queue
        from pipeline.live import get_next_result
        new_results = []
        for _ in range(20):
            r = get_next_result(session, timeout=0.02)
            if r is None:
                break
            new_results.append(r)
        if new_results:
            st.session_state["live_results"] = (
                st.session_state["live_results"] + new_results
            )[-100:]  # keep last 100

        # Auto-refresh while running
        import time
        time.sleep(0.5)
        st.rerun()

    # Live transcript
    results = st.session_state.get("live_results", [])
    if results:
        st.markdown('<div class="cg-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:12px">Live Transcript</div>', unsafe_allow_html=True)
        for r in reversed(results[-30:]):
            ts_str = datetime.fromtimestamp(r.timestamp).strftime("%H:%M:%S")
            if r.flag:
                from ui.components import CATEGORY_LABELS
                cat_label = CATEGORY_LABELS.get(r.category, r.category)
                st.markdown(
                    f'<div class="cg-seg cg-seg-flagged">'
                    f'<span class="cg-ts">{ts_str}</span>&ensp;'
                    f'<span class="cg-pill cg-pill-alert">{cat_label}</span>&ensp;'
                    f'<span class="cg-pill cg-pill-mute">{r.confidence:.0%}</span>'
                    f'<br/>{r.text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="cg-seg">'
                    f'<span class="cg-ts">{ts_str}</span>&ensp;{r.text}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Recent DB alerts
    with get_session() as s:
        recent_alerts = get_recent_alerts(s, limit=10)
    if recent_alerts:
        st.markdown('<div class="cg-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.75rem;color:var(--mute);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:10px">Recent Alerts</div>', unsafe_allow_html=True)
        for al in recent_alerts:
            level_col = "var(--alert)" if al.level == "emergency" else "#f0b429"
            st.markdown(
                f'<div class="cg-seg" style="margin-bottom:4px">'
                f'<span style="color:{level_col};font-weight:700">{al.level.upper()}</span>&ensp;'
                f'<span class="cg-ts">{al.created_at.strftime("%d %b %H:%M:%S")}</span>&ensp;'
                f'<span class="cg-pill cg-pill-alert">{al.category}</span>&ensp;'
                f'<span class="cg-pill cg-pill-mute">{al.confidence:.0%}</span>'
                f'<br/><span style="color:var(--mute);font-size:.85rem">{al.text[:120]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Enroll Voice
# ─────────────────────────────────────────────────────────────────────────────

def page_enroll_voice():
    st.markdown('<h1 style="margin-bottom:4px">Enroll a Voice</h1><p style="color:var(--mute);font-size:1rem;margin-top:0;margin-bottom:1.5rem">Upload a 10–30 second clean audio sample to generate a voice fingerprint.</p>', unsafe_allow_html=True)
    st.markdown('<div class="cg-card">', unsafe_allow_html=True)
    name     = st.text_input("Person's name", placeholder="e.g. John Smith")
    uploaded = st.file_uploader("Voice sample", type=["wav","mp3","m4a","flac","ogg"])
    if uploaded:
        st.audio(uploaded)
    if st.button("Enroll Voice", type="primary"):
        if not name.strip():
            toast("Please enter a name.", "warning")
        elif not uploaded:
            toast("Please upload an audio sample.", "warning")
        else:
            tmp = Path(tempfile.mkdtemp()) / uploaded.name
            tmp.write_bytes(uploaded.getbuffer())
            with st.spinner(f"Generating embedding for '{name.strip()}'…"):
                try:
                    from pipeline.voice_recognition import generate_embedding
                    from pipeline.preprocess_audio import convert_to_wav
                    wav = convert_to_wav(str(tmp))
                    emb = generate_embedding(wav)
                    with get_session() as s:
                        save_voice_profile(s, name=name.strip(), embedding=emb, audio_file=uploaded.name)
                    toast(f"'{name.strip()}' enrolled.", "success")
                except Exception as exc:
                    toast(f"Enrollment failed: {exc}", "error")
                finally:
                    try: tmp.unlink()
                    except Exception: pass
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Manage Voices
# ─────────────────────────────────────────────────────────────────────────────

def page_manage_voices():
    st.markdown('<h1 style="margin-bottom:4px">Enrolled Voices</h1><p style="color:var(--mute);font-size:1rem;margin-top:0;margin-bottom:1.5rem">View and delete enrolled voice profiles.</p>', unsafe_allow_html=True)
    with get_session() as s:
        profiles = get_all_voice_profiles(s)
    if not profiles:
        st.markdown('<div class="cg-card" style="text-align:center;padding:3rem 2rem;color:var(--mute)">No voices enrolled yet.</div>', unsafe_allow_html=True)
        return
    st.metric("Total enrolled voices", len(profiles))
    for p in profiles:
        st.markdown('<div class="cg-card" style="padding:.9rem 1.25rem;margin-bottom:.6rem">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([4, 4, 1])
        c1.markdown(f"**{p.name}**")
        c2.markdown(f'<span style="color:var(--dim);font-size:.8rem">Added {p.date_added.strftime("%d %b %Y, %H:%M")}</span>', unsafe_allow_html=True)
        if c3.button("Delete", key=f"del_profile_{p.id}"):
            with get_session() as s:
                delete_voice_profile(s, p.id)
            toast(f"Deleted '{p.name}'.", "success")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

NAV = {
    "Analyze call":    page_analyze,
    "Past calls":      page_past_calls,
    "Keyword library": page_keyword_library,
    "Live monitor":    page_live_monitor,
    "Enroll a Voice":  page_enroll_voice,
    "Manage Voices":   page_manage_voices,
}

def main():
    if not _password_gate():
        return

    sidebar_brand()

    # Diagnostics toggle
    show_diag = st.sidebar.toggle(
        "Show diagnostics",
        value=st.session_state.get("show_diagnostics", cfg.SHOW_DIAGNOSTICS_DEFAULT),
        key="_diag_toggle",
    )
    st.session_state["show_diagnostics"] = show_diag

    st.sidebar.divider()

    page_name = st.sidebar.radio(
        "Navigation", list(NAV.keys()), label_visibility="collapsed"
    )

    st.sidebar.markdown('<div style="flex:1;min-height:80px"></div>', unsafe_allow_html=True)
    sidebar_engine_card(
        model  = cfg.WHISPER_MODEL,
        device = cfg.COMPUTE_DEVICE,
        hf_set = cfg.has_hf_token,
    )

    NAV[page_name]()


if __name__ == "__main__":
    main()
