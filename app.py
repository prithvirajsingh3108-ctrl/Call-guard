"""
app.py
──────
CallGuard Streamlit Dashboard

Tabs
----
1. 🎙 Analyze Call   — Upload audio, run the full pipeline, view results
2. 📂 Past Calls     — Browse the SQLite log of previously analyzed calls

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# ── Load secrets from Streamlit Cloud if running in cloud ────────────────────
# Streamlit Cloud stores secrets in st.secrets; locally we use .env
def _load_streamlit_secrets():
    try:
        for key in ["HF_TOKEN", "WHISPER_MODEL", "COMPUTE_DEVICE",
                    "DATABASE_PATH", "KEYWORDS_PATH", "FUZZY_THRESHOLD",
                    "CONTEXT_WINDOW_SIZE"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # st.secrets not available locally — .env handles it

_load_streamlit_secrets()
from dotenv import load_dotenv

load_dotenv()

# Make sure project root is on the path so our modules import correctly
sys.path.insert(0, str(Path(__file__).parent))

from db.database import (
    init_db, get_session, create_call, update_call_status,
    save_call_results, get_all_calls, get_call_by_id,
    get_segments_for_call, get_flags_for_call, get_summary_for_call,
    delete_call,
    save_voice_profile, get_all_voice_profiles, delete_voice_profile,
)

# ── Cache heavy models so they load once and stay in memory across reruns ─────
@st.cache_resource(show_spinner=False)
def _get_whisperx_model():
    import whisperx as _wx
    import warnings; warnings.filterwarnings("ignore")
    print("[app] Loading whisperx model (cached)...")
    return _wx.load_model(
        os.getenv("WHISPER_MODEL", "base"),
        os.getenv("COMPUTE_DEVICE", "cpu"),
        compute_type="float32",
    )

@st.cache_resource(show_spinner=False)
def _get_diarize_model():
    import warnings; warnings.filterwarnings("ignore")
    from whisperx.diarize import DiarizationPipeline
    token = os.getenv("HF_TOKEN", "")
    if not token:
        return None
    try:
        print("[app] Loading diarization model (cached)...")
        return DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=token,
            device=os.getenv("COMPUTE_DEVICE", "cpu"),
        )
    except Exception as e:
        print(f"[app] Diarization model load failed: {e}")
        return None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CallGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise DB on startup ──────────────────────────────────────────────────
init_db()

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .flagged-segment {
        background-color: #ffe0e0;
        border-left: 4px solid #e74c3c;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .safe-segment {
        background-color: #f8f9fa;
        border-left: 4px solid #dee2e6;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .speaker-label {
        font-weight: bold;
        font-size: 0.85em;
        color: #555;
    }
    .timestamp-label {
        font-size: 0.8em;
        color: #888;
        font-family: monospace;
    }
    .flag-badge {
        display: inline-block;
        background: #e74c3c;
        color: white;
        border-radius: 3px;
        padding: 2px 6px;
        font-size: 0.75em;
        margin-left: 8px;
    }
    .confidence-badge {
        display: inline-block;
        background: #e67e22;
        color: white;
        border-radius: 3px;
        padding: 2px 6px;
        font-size: 0.75em;
        margin-left: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper: format seconds as MM:SS ──────────────────────────────────────────
def fmt_time(secs: float) -> str:
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m:02d}:{s:02d}"


# ── Helper: category color badge ─────────────────────────────────────────────
CATEGORY_COLORS = {
    "threat":        "#e74c3c",
    "abuse":         "#e67e22",
    "harm_planning": "#8e44ad",
    "disaster":      "#c0392b",
}

def category_badge(cat: str) -> str:
    color = CATEGORY_COLORS.get(cat, "#555")
    return f'<span style="background:{color};color:white;border-radius:3px;padding:2px 8px;font-size:0.8em;">{cat}</span>'


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyze a new call
# ═════════════════════════════════════════════════════════════════════════════

def render_transcript(enriched_segments: list[dict]):
    """Render the full transcript with flagged segments highlighted."""
    st.subheader("📝 Transcript")

    for seg in enriched_segments:
        ts          = f"{fmt_time(seg['start'])} – {fmt_time(seg['end'])}"
        # Use speaker_display if available (set by voice recognition),
        # otherwise fall back to the raw speaker label
        spk_display = seg.get("speaker_display") or seg.get("speaker", "UNKNOWN")

        if seg.get("flag"):
            with st.expander(
                f"⚠️ {spk_display}  [{ts}]  — {seg['text'][:80]}{'...' if len(seg['text'])>80 else ''}",
                expanded=False,
            ):
                st.markdown(
                    f"**Full text:** {seg['text']}\n\n"
                    f"**Speaker:** {spk_display}  \n"
                    f"**Category:** {seg['category']}  \n"
                    f"**Keyword matched:** `{seg['matched_keyword']}`  \n"
                    f"**Threat confidence:** {seg['confidence']:.0%}"
                )
                if seg.get("context_window"):
                    st.markdown("**Context (preceding turns):**")
                    for ctx in seg["context_window"]:
                        ctx_display = ctx.get("speaker_display", ctx.get("speaker",""))
                        st.markdown(f"> *{ctx_display}:* {ctx['text']}")
        else:
            st.markdown(
                f'<div class="safe-segment">'
                f'<span class="speaker-label">{spk_display}</span> '
                f'<span class="timestamp-label">[{ts}]</span><br/>'
                f'{seg["text"]}'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_summary_panel(summary: dict, call_duration: float):
    """Render the summary metrics and timeline chart."""
    st.subheader("📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Segments",   summary["total_segments"])
    col2.metric("⚠️ Flags",         summary["total_flags"])
    col3.metric("Categories",        len(summary["by_category"]))
    col4.metric("Peak Confidence",   f"{summary['highest_confidence']:.0%}")

    if summary["total_flags"] == 0:
        st.success("✅ No threats detected in this call.")
        return

    # Category breakdown bar chart
    if summary["by_category"]:
        df_cat = pd.DataFrame(
            list(summary["by_category"].items()),
            columns=["Category", "Count"],
        )
        fig = px.bar(
            df_cat, x="Category", y="Count",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            title="Flags by Category",
            height=300,
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Timeline of flags
    flagged = summary.get("flagged_segments", [])
    if flagged and call_duration > 0:
        df_time = pd.DataFrame([
            {
                "Time (s)":   f["start"],
                "Speaker":    f["speaker"],
                "Category":   f["category"],
                "Confidence": f["confidence"],
                "Text":       f["text"][:50],
            }
            for f in flagged
        ])
        fig2 = px.scatter(
            df_time,
            x="Time (s)", y="Category",
            size="Confidence", color="Category",
            color_discrete_map=CATEGORY_COLORS,
            hover_data=["Speaker", "Text", "Confidence"],
            title="Flag Timeline",
            height=280,
        )
        fig2.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)


def page_analyze():
    st.title("🛡️ CallGuard — Analyze a Call")
    st.markdown("Upload an audio recording to transcribe it, detect threats, and log the results.")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose an audio file (.wav, .mp3, .m4a, .flac, .ogg)",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
    )

    if not uploaded:
        st.info("Upload a file above to get started.")
        return

    st.audio(uploaded)

    use_diarization = st.checkbox(
        "Enable speaker diarization (requires HF_TOKEN in .env)",
        value=bool(os.getenv("HF_TOKEN")),
    )

    if not st.button("🚀 Run Analysis", type="primary"):
        return

    # ── Save uploaded file to disk ─────────────────────────────────────────
    tmp_dir  = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # ── Create call record ─────────────────────────────────────────────────
    with get_session() as session:
        call = create_call(session, filename=uploaded.name, file_path=str(tmp_path))
        call_id = call.id

    # ── Progress display ───────────────────────────────────────────────────
    progress  = st.progress(0, text="Starting...")
    status_ph = st.empty()

    def update_status(msg: str, pct: int):
        progress.progress(pct, text=msg)
        status_ph.info(msg)

    try:
        # Step 1: Preprocess audio
        update_status("🔄 Preprocessing audio (converting to 16 kHz mono WAV)...", 10)
        with get_session() as session:
            update_call_status(session, call_id, "transcribing")

        from pipeline.preprocess_audio import convert_to_wav
        wav_path = convert_to_wav(str(tmp_path))

        # Measure duration
        try:
            from pydub import AudioSegment as _AS
            _audio = _AS.from_file(wav_path)
            duration_sec = len(_audio) / 1000.0
        except Exception:
            duration_sec = 0.0

        # Step 2: Transcribe using cached models
        if use_diarization:
            update_status("🎙️ Transcribing + diarizing...", 30)
        else:
            update_status("🎙️ Transcribing with plain Whisper...", 30)

        os.environ["USE_DIARIZATION"] = "true" if use_diarization else "false"

        if use_diarization:
            import whisperx
            import warnings; warnings.filterwarnings("ignore")
            from whisperx.diarize import assign_word_speakers

            # Use Streamlit-cached models — only loads once per session
            wx_model = _get_whisperx_model()
            audio    = whisperx.load_audio(wav_path)

            result   = wx_model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", "en")

            # NOTE: Alignment step intentionally skipped —
            # it downloads a 3-4GB language model per language and is not
            # needed for threat detection (we only need segment text, not word timestamps)

            # Diarize
            try:
                diarize_model = _get_diarize_model()
                if diarize_model:
                    diarize_segs = diarize_model(wav_path)
                    result       = assign_word_speakers(diarize_segs, result)
            except Exception as exc:
                print(f"[app] Diarization skipped: {exc}")

            # Build segments list
            raw = result.get("segments",[]) if isinstance(result,dict) else result.segments
            segments = []
            for seg in raw:
                if isinstance(seg, dict):
                    segments.append({"speaker": seg.get("speaker","UNKNOWN"),
                                     "text": seg.get("text","").strip(),
                                     "start": round(seg.get("start",0.0),3),
                                     "end":   round(seg.get("end",0.0),3)})
                else:
                    segments.append({"speaker": getattr(seg,"speaker","UNKNOWN") or "UNKNOWN",
                                     "text": getattr(seg,"text","").strip(),
                                     "start": round(getattr(seg,"start",0.0),3),
                                     "end":   round(getattr(seg,"end",0.0),3)})
        else:
            from pipeline.transcribe import transcribe_plain
            segments = transcribe_plain(wav_path)

        # Step 2b: Voice recognition — match every speaker against enrolled profiles
        update_status("🔎 Matching speakers against enrolled voice profiles...", 60)
        try:
            from pipeline.voice_recognition import identify_speakers, apply_speaker_names
            name_map = identify_speakers(segments, wav_path)
            segments = apply_speaker_names(segments, name_map)

            # Show recognition result in UI
            matched = [v[0] for v in name_map.values() if v[2] == "matched"]
            no_enrolled = any(v[2] == "no_enrolled_voices" for v in name_map.values())
            if no_enrolled:
                st.info("ℹ️ No voices enrolled yet — all speakers shown as 'Unknown Speaker'. "
                        "Use **Enroll a Voice** to register speakers.")
            elif matched:
                st.success(f"🎙️ Recognized: {', '.join(set(matched))}")
            else:
                st.info("🎙️ No enrolled voices matched in this call.")
        except Exception as exc:
            st.warning(f"Speaker recognition skipped: {exc}")
            # Apply default unknown labeling so segments still have display fields
            for seg in segments:
                seg["matched_name"]     = "Unknown Speaker"
                seg["match_confidence"] = 0.0
                seg["match_status"]     = "not_run"
                seg["speaker_display"]  = seg.get("speaker", "UNKNOWN")
                seg["speaker_original"] = seg.get("speaker", "UNKNOWN")

        update_status("🔍 Analyzing transcript for threats...", 70)
        with get_session() as session:
            update_call_status(session, call_id, "analyzing")

        # Step 3+4: Detect threats
        from pipeline.detector import detect_threats, summarize_flags
        enriched  = detect_threats(segments)
        summary   = summarize_flags(enriched)

        # Step 5: Persist to DB
        update_status("💾 Saving results to database...", 90)
        with get_session() as session:
            save_call_results(session, call_id, enriched, summary, duration_sec)
            update_call_status(session, call_id, "done")

        progress.progress(100, text="✅ Analysis complete!")
        status_ph.success(f"Done! Found {summary['total_flags']} flag(s) in {len(segments)} segments.")

        # ── Display results ────────────────────────────────────────────────
        st.divider()
        render_summary_panel(summary, duration_sec)
        st.divider()
        render_transcript(enriched)

    except Exception as exc:
        progress.empty()
        status_ph.error(f"❌ Error: {exc}")
        with get_session() as session:
            update_call_status(session, call_id, "error", str(exc))
        st.exception(exc)

    finally:
        # Clean up temp files
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            if "wav_path" in locals() and Path(wav_path) != tmp_path and Path(wav_path).exists():
                os.unlink(wav_path)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Past Calls Browser
# ═════════════════════════════════════════════════════════════════════════════

def page_past_calls():
    st.title("📂 Past Calls")
    st.markdown("Browse previously analyzed call recordings from the database.")

    with get_session() as session:
        calls = get_all_calls(session)

        if not calls:
            st.info("No calls have been analyzed yet. Go to **Analyze Call** to process your first recording.")
            return

        # ── Calls table ───────────────────────────────────────────────────────
        table_data = []
        for c in calls:
            summary = get_summary_for_call(session, c.id)
            table_data.append({
                "ID":         c.id,
                "File":       c.filename,
                "Status":     c.status,
                "Duration":   fmt_time(c.duration_sec) if c.duration_sec else "—",
                "Flags":      str(summary.total_flags) if summary else "—",
                "Segments":   str(summary.total_segments) if summary else "—",
                "Analyzed":   c.completed_at.strftime("%Y-%m-%d %H:%M") if c.completed_at else "—",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Select a call to inspect ──────────────────────────────────────────
        st.divider()
        call_ids   = [c.id for c in calls]
        call_names = [f"#{c.id} — {c.filename}" for c in calls]
        choice     = st.selectbox("Select a call to view details:", call_names)
        selected_id = call_ids[call_names.index(choice)]

        selected_call = get_call_by_id(session, selected_id)
        if selected_call is None:
            st.error("Call not found.")
            return

        st.subheader(f"Call #{selected_call.id}: {selected_call.filename}")
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.write(f"**Status:** {selected_call.status}")
        meta_col2.write(f"**Duration:** {fmt_time(selected_call.duration_sec) if selected_call.duration_sec else '—'}")
        meta_col3.write(f"**Analyzed:** {selected_call.completed_at.strftime('%Y-%m-%d %H:%M') if selected_call.completed_at else '—'}")

        if selected_call.status == "error":
            st.error(f"Processing error: {selected_call.error_msg}")
            return

        # Summary panel
        summary = get_summary_for_call(session, selected_id)
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segments",       summary.total_segments)
            col2.metric("⚠️ Flags",       summary.total_flags)
            col3.metric("Categories",     len(summary.by_category) if summary.by_category else 0)
            col4.metric("Peak Confidence", f"{summary.highest_confidence:.0%}" if summary.highest_confidence else "—")

            if summary.by_category:
                df_cat = pd.DataFrame(
                    list(summary.by_category.items()),
                    columns=["Category", "Count"],
                )
                fig = px.bar(
                    df_cat, x="Category", y="Count",
                    color="Category",
                    color_discrete_map=CATEGORY_COLORS,
                    title="Flags by Category",
                    height=280,
                )
                fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Full transcript from DB
        segments = get_segments_for_call(session, selected_id)
        flags    = get_flags_for_call(session, selected_id)

        # Build a lookup: segment_id → list of flags
        flag_map: dict[int, list] = {}
        for fl in flags:
            flag_map.setdefault(fl.segment_id, []).append(fl)

        st.divider()
        st.subheader("📝 Transcript")

        for seg in segments:
            ts    = f"{fmt_time(seg.start_sec)} – {fmt_time(seg.end_sec)}"
            flgs  = flag_map.get(seg.id, [])

            # Build display label from stored match data
            if seg.matched_name and seg.match_status == "matched":
                spk_display = f"{seg.matched_name} ({seg.match_confidence:.0%} match)"
            elif seg.match_status == "no_enrolled_voices":
                spk_display = "Unknown Speaker (no enrolled voices yet)"
            elif seg.match_status == "no_match":
                conf_str = f"{seg.match_confidence:.0%}" if seg.match_confidence else "0%"
                spk_display = f"Unknown Speaker (no match, best: {conf_str})"
            else:
                spk_display = seg.speaker  # fallback to original diarized label

            if flgs:
                fl = flgs[0]
                with st.expander(
                    f"⚠️ {spk_display}  [{ts}]  — {seg.text[:80]}{'...' if len(seg.text)>80 else ''}",
                    expanded=False,
                ):
                    st.markdown(
                        f"**Full text:** {seg.text}\n\n"
                        f"**Speaker:** {spk_display}  \n"
                        f"**Category:** {fl.category}  \n"
                        f"**Keyword:** `{fl.matched_keyword}`  \n"
                        f"**Threat confidence:** {fl.confidence:.0%}"
                    )
                    if fl.context_window:
                        st.markdown("**Context:**")
                        for ctx in fl.context_window:
                            st.markdown(f"> *{ctx.get('speaker','')}:* {ctx['text']}")
            else:
                st.markdown(
                    f'<div class="safe-segment">'
                    f'<span class="speaker-label">{spk_display}</span> '
                    f'<span class="timestamp-label">[{ts}]</span><br/>'
                    f'{seg.text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Delete button
        st.divider()
        if st.button("🗑️ Delete this call from database", type="secondary"):
            with get_session() as del_session:
                delete_call(del_session, selected_id)
            st.success(f"Call #{selected_id} deleted.")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Enroll a Voice
# ═════════════════════════════════════════════════════════════════════════════

def page_enroll_voice():
    st.title("🎤 Enroll a Voice")
    st.markdown(
        "Upload a short, clean audio sample (10–30 seconds) of one person speaking. "
        "CallGuard will generate a voice fingerprint and recognize that person in future calls."
    )

    name = st.text_input("Person's name", placeholder="e.g. John Smith")

    uploaded = st.file_uploader(
        "Upload voice sample (.wav, .mp3, .m4a)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
    )

    if uploaded:
        st.audio(uploaded)

    if not st.button("🔐 Enroll Voice", type="primary"):
        return

    if not name.strip():
        st.error("Please enter a name before enrolling.")
        return
    if not uploaded:
        st.error("Please upload an audio sample.")
        return

    # Save uploaded file to temp
    tmp_dir  = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner(f"Generating voice embedding for '{name.strip()}'..."):
        try:
            from pipeline.voice_recognition import generate_embedding
            from pipeline.preprocess_audio import convert_to_wav

            wav_path  = convert_to_wav(str(tmp_path))
            embedding = generate_embedding(wav_path)

            with get_session() as session:
                save_voice_profile(
                    session,
                    name       = name.strip(),
                    embedding  = embedding,
                    audio_file = uploaded.name,
                )

            st.success(f"✅ '{name.strip()}' enrolled successfully! They will now be recognized in future calls.")

        except Exception as exc:
            st.error(f"Enrollment failed: {exc}")
        finally:
            try:
                tmp_path.unlink()
                if 'wav_path' in locals() and wav_path != str(tmp_path):
                    os.unlink(wav_path)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Manage Enrolled Voices
# ═════════════════════════════════════════════════════════════════════════════

def page_manage_voices():
    st.title("👥 Manage Enrolled Voices")
    st.markdown("View and delete enrolled voice profiles.")

    with get_session() as session:
        profiles = get_all_voice_profiles(session)

        if not profiles:
            st.info("No voices enrolled yet. Go to **Enroll a Voice** to add someone.")
            return

        # Summary count
        st.metric("Total enrolled voices", len(profiles))
        st.divider()

        for profile in profiles:
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.markdown(f"**{profile.name}**")
            col2.markdown(
                f"<span style='color:#888;font-size:0.85em'>"
                f"Added {profile.date_added.strftime('%Y-%m-%d %H:%M')}"
                f"{'  ·  Sample: ' + profile.audio_file if profile.audio_file else ''}"
                f"</span>",
                unsafe_allow_html=True,
            )
            profile_id = profile.id
            if col3.button("🗑️ Delete", key=f"del_profile_{profile_id}"):
                with get_session() as del_session:
                    delete_voice_profile(del_session, profile_id)
                st.success(f"Deleted '{profile.name}'.")
                st.rerun()

        st.divider()
        st.markdown(
            f"**Recognition threshold:** `{os.getenv('RECOGNITION_THRESHOLD', '0.75')}` "
            f"— set `RECOGNITION_THRESHOLD` in `.env` to adjust (0.0–1.0, higher = stricter)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main layout — sidebar navigation
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.sidebar.image(
        "https://img.icons8.com/color/96/000000/shield.png",
        width=64,
    )
    st.sidebar.title("CallGuard")
    st.sidebar.markdown("Conversation Threat Detection")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["🎙️ Analyze Call", "📂 Past Calls", "🎤 Enroll a Voice", "👥 Manage Voices"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Config**")
    st.sidebar.markdown(f"Model: `{os.getenv('WHISPER_MODEL', 'base')}`")
    st.sidebar.markdown(f"Device: `{os.getenv('COMPUTE_DEVICE', 'cpu')}`")
    hf = os.getenv("HF_TOKEN", "")
    st.sidebar.markdown(f"HF Token: {'✅ set' if hf else '❌ not set'}")
    st.sidebar.markdown(f"Fuzzy threshold: `{os.getenv('FUZZY_THRESHOLD', '80')}`")
    st.sidebar.markdown(f"Voice match threshold: `{os.getenv('RECOGNITION_THRESHOLD', '0.75')}`")

    if page == "🎙️ Analyze Call":
        page_analyze()
    elif page == "📂 Past Calls":
        page_past_calls()
    elif page == "🎤 Enroll a Voice":
        page_enroll_voice()
    elif page == "👥 Manage Voices":
        page_manage_voices()



if __name__ == "__main__":
    main()
