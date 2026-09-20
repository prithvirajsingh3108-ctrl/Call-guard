"""
ui/components.py
────────────────
Reusable presentation components for the CallGuard dashboard.

Components
----------
1.  waveform(audio_path, flags, duration)    — canvas waveform with flag markers
2.  transcript(segments, flags)              — styled transcript with flagged rows
3.  risk_ring(score)                         — SVG circular gauge with animation
4.  category_bars(summary)                  — animated horizontal bar per category
5.  flag_timeline(flags, duration)           — SVG scatter timeline
6.  speaker_share(segments)                 — two-color split bar per speaker
7.  pipeline_steps(active_index)            — 4-step pipeline progress bar

Each function is self-contained and only calls st.* / st.components.v1.html.
No imports from pipeline/ or db/ — pure presentation.
"""

import json
import math
import html as _html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ── Design-token constants (mirror CSS variables for Python code) ──────────────
EM      = "#3fe0a0"
ALERT   = "#ff6b5e"
SURFACE = "#0b150f"
SURFACE2= "#111f16"
MUTE    = "#8ba493"
DIM     = "#5b7263"
TEXT    = "#eef5ef"
LINE    = "rgba(214,238,222,.09)"

CATEGORY_LABELS = {
    "threat":        "Direct Threat",
    "abuse":         "Abuse",
    "harm_planning": "Harm Planning",
    "disaster":      "Disaster",
}

STEP_LABELS = ["Preprocessing", "Transcribing", "Analyzing", "Saving"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Waveform
# ─────────────────────────────────────────────────────────────────────────────

def waveform(audio_path: str | None, flags: list[dict], duration: float) -> None:
    """
    Render an interactive canvas waveform.

    Reads real amplitude data from the WAV file (downsampled to ~190 bars)
    via numpy.  Bars before the playhead are emerald; the rest are white at
    20 % opacity.  Flag positions get a thin alert-red vertical line with a
    small diamond marker on top.  Clicking seeks the HTML5 audio element.
    Animates the bars drawing left-to-right once on first render.

    Falls back to a synthetic sine-wave pattern if the file is unavailable.
    """
    # Build amplitude array —————————————————————————————————————————————
    NUM_BARS = 190
    amplitudes = []

    if audio_path and Path(audio_path).exists():
        try:
            import numpy as np
            from scipy.io import wavfile

            sr, data = wavfile.read(audio_path)
            # Flatten to mono if stereo
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(float)
            # Normalise
            mx = max(abs(data).max(), 1)
            data = data / mx

            # Downsample to NUM_BARS
            chunk = max(1, len(data) // NUM_BARS)
            for i in range(NUM_BARS):
                chunk_data = data[i * chunk: (i + 1) * chunk]
                amplitudes.append(float(abs(chunk_data).mean()) if len(chunk_data) else 0.0)

            # Normalise bars to 0-1
            mx2 = max(amplitudes) or 1
            amplitudes = [a / mx2 for a in amplitudes]
        except Exception:
            amplitudes = []

    # Fallback: synthetic waveform
    if not amplitudes:
        import math as _m
        for i in range(NUM_BARS):
            t = i / NUM_BARS
            amplitudes.append(
                0.3 + 0.5 * abs(_m.sin(t * 14)) * (0.5 + 0.5 * _m.sin(t * 3.7))
            )

    # Encode flag positions as fraction of duration ————————————————————
    flag_fractions = []
    if duration > 0:
        for f in flags:
            ts = f.get("timestamp_sec") or f.get("start") or 0.0
            flag_fractions.append(min(ts / duration, 1.0))

    amps_json  = json.dumps(amplitudes)
    flags_json = json.dumps(flag_fractions)

    html_code = f"""
<style>
  #wv-wrap {{
    position: relative;
    background: {SURFACE};
    border-radius: 16px;
    border: 1px solid {LINE};
    padding: 16px 16px 12px;
    user-select: none;
  }}
  #wv-canvas {{
    width: 100%;
    height: 80px;
    cursor: pointer;
    display: block;
  }}
  #wv-time {{
    font-size: 11px;
    color: {MUTE};
    font-family: monospace;
    margin-top: 6px;
    text-align: right;
  }}
</style>

<div id="wv-wrap">
  <canvas id="wv-canvas" height="80"></canvas>
  <div id="wv-time">0:00</div>
</div>

<script>
(function() {{
  const amps       = {amps_json};
  const flagFracs  = {flags_json};
  const duration   = {duration or 0};
  const EM         = "{EM}";
  const ALERT      = "{ALERT}";
  const MUTED      = "rgba(238,245,239,0.20)";

  const canvas  = document.getElementById("wv-canvas");
  const timeEl  = document.getElementById("wv-time");
  const ctx     = canvas.getContext("2d");
  let playhead  = 0;   // 0.0 – 1.0
  let animStep  = 0;
  let animId    = null;

  function fmt(sec) {{
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return String(m).padStart(2,"0") + ":" + String(s).padStart(2,"0");
  }}

  function draw(revealTo) {{
    const W = canvas.offsetWidth * window.devicePixelRatio;
    const H = canvas.height      * window.devicePixelRatio;
    canvas.width = W;

    ctx.clearRect(0, 0, W, H);

    const n    = amps.length;
    const gap  = 2 * window.devicePixelRatio;
    const bw   = (W - gap * (n - 1)) / n;
    const maxH = H * 0.9;
    const cy   = H / 2;

    for (let i = 0; i < n; i++) {{
      if (i / n > revealTo) break;   // animate reveal

      const x   = i * (bw + gap);
      const h   = Math.max(3 * window.devicePixelRatio, amps[i] * maxH);
      const frac = i / n;

      // colour: emerald before playhead, muted after
      ctx.fillStyle = frac <= playhead ? EM : MUTED;

      // rounded bar
      const r = Math.min(bw / 2, 3 * window.devicePixelRatio);
      ctx.beginPath();
      ctx.roundRect(x, cy - h / 2, bw, h, r);
      ctx.fill();
    }}

    // Flag markers
    flagFracs.forEach(frac => {{
      const x = frac * W;
      // vertical line
      ctx.strokeStyle = ALERT;
      ctx.lineWidth   = 1.5 * window.devicePixelRatio;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
      // diamond on top
      const d = 5 * window.devicePixelRatio;
      ctx.fillStyle = ALERT;
      ctx.beginPath();
      ctx.moveTo(x,     d);
      ctx.lineTo(x + d, d * 2);
      ctx.lineTo(x,     d * 3);
      ctx.lineTo(x - d, d * 2);
      ctx.closePath();
      ctx.fill();
    }});
  }}

  // Animate bars drawing left→right on load
  function animate() {{
    animStep += 0.03;
    draw(Math.min(animStep, 1.0));
    if (animStep < 1.0) {{
      animId = requestAnimationFrame(animate);
    }} else {{
      draw(1.0);
    }}
  }}

  animate();

  // Click to seek
  canvas.addEventListener("click", function(e) {{
    const rect = canvas.getBoundingClientRect();
    playhead = (e.clientX - rect.left) / rect.width;
    timeEl.textContent = fmt(playhead * duration);
    draw(1.0);
  }});

  // Redraw on resize
  const ro = new ResizeObserver(() => draw(1.0));
  ro.observe(canvas);
}})();
</script>
"""
    components.html(html_code, height=130)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transcript
# ─────────────────────────────────────────────────────────────────────────────

def fmt_time(secs: float) -> str:
    """Format seconds as MM:SS."""
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m:02d}:{s:02d}"


def transcript(segments: list[dict], flags: list[dict] | None = None) -> None:
    """
    Render the full transcript.

    Each row shows timestamp, speaker label, and text.
    Flagged rows get:
      - soft red background
      - red underline on the matched keyword
      - category badge
    Expanding a flagged row reveals category, matched keyword, confidence %,
    and up to 4 context-window segments.
    Rows where a hit was cleared by context pass get a dashed outline and
    a "Cleared by context" badge.

    Accepts both live enriched-segment dicts (from detector.detect_threats)
    and ORM Segment objects (from past-calls DB query).
    """
    # Normalise ORM objects → plain dicts ——————————————————————————————
    norm = []
    for seg in segments:
        if isinstance(seg, dict):
            norm.append(seg)
        else:
            # SQLAlchemy Segment object
            flag_list = getattr(seg, "flags", [])
            fl = flag_list[0] if flag_list else None
            norm.append({
                "start":           getattr(seg, "start_sec", 0),
                "end":             getattr(seg, "end_sec", 0),
                "speaker":         getattr(seg, "speaker", "UNKNOWN"),
                "matched_name":    getattr(seg, "matched_name", None),
                "match_status":    getattr(seg, "match_status", "not_run"),
                "match_confidence":getattr(seg, "match_confidence", 0),
                "text":            getattr(seg, "text", ""),
                "flag":            fl is not None,
                "category":        getattr(fl, "category", "") if fl else "",
                "matched_keyword": getattr(fl, "matched_keyword", "") if fl else "",
                "confidence":      getattr(fl, "confidence", 0.0) if fl else 0.0,
                "context_window":  getattr(fl, "context_window", []) if fl else [],
                "cleared":         False,
            })

    for seg in norm:
        ts  = f"{fmt_time(seg['start'])} – {fmt_time(seg['end'])}"
        ms  = seg.get("match_status", "not_run")
        mn  = seg.get("matched_name", "")
        mc  = seg.get("match_confidence", 0)

        # Resolve display name
        if ms == "matched" and mn:
            spk = f"{mn} <span style='color:{DIM};font-size:.75em'>({mc:.0%})</span>"
        elif ms == "no_enrolled_voices":
            spk = "Unknown Speaker"
        elif ms == "no_match":
            spk = f"Unknown <span style='color:{DIM};font-size:.75em'>(no match)</span>"
        else:
            spk = seg.get("speaker_display") or seg.get("speaker", "UNKNOWN")

        is_flagged = seg.get("flag", False)
        is_cleared = seg.get("cleared", False)
        cat        = seg.get("category", "")
        kw         = seg.get("matched_keyword", "")
        conf       = seg.get("confidence", 0.0)
        text       = _html.escape(seg.get("text", ""))
        ctx_win    = seg.get("context_window", []) or []

        # Underline matched keyword in text
        if kw and is_flagged:
            safe_kw  = _html.escape(kw)
            text_hi  = text.replace(
                safe_kw,
                f'<span class="cg-kw">{safe_kw}</span>',
                1,
            )
        else:
            text_hi = text

        if is_cleared:
            # Dashed outline — cleared by context pass
            cat_label = CATEGORY_LABELS.get(cat, cat)
            st.markdown(
                f'<div class="cg-seg cg-seg-cleared">'
                f'<span class="cg-ts">{ts}</span>&ensp;'
                f'<span class="cg-spk" style="color:{MUTE}">{spk}</span>'
                f'&ensp;<span class="cg-pill cg-pill-mute">Cleared by context</span>'
                f'<br/><span style="color:{MUTE}">{text_hi}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif is_flagged:
            cat_label = CATEGORY_LABELS.get(cat, cat)
            # Category badge colour
            with st.expander(
                f"⚠️  {fmt_time(seg['start'])}  ·  {spk.split('<')[0].strip()}  —  "
                f"{seg.get('text','')[:70]}{'…' if len(seg.get('text',''))>70 else ''}",
                expanded=False,
            ):
                st.markdown(
                    f'<div class="cg-seg cg-seg-flagged" style="margin:0">'
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">'
                    f'<span class="cg-ts">{ts}</span>'
                    f'<span class="cg-spk">{spk}</span>'
                    f'<span class="cg-pill cg-pill-alert">{cat_label}</span>'
                    f'<span class="cg-pill cg-pill-mute">{conf:.0%} confidence</span>'
                    f'</div>'
                    f'<div style="margin-bottom:8px">{text_hi}</div>'
                    f'<div style="font-size:.8rem;color:{MUTE}">Keyword: '
                    f'<code style="background:{SURFACE2};padding:1px 6px;border-radius:4px">{_html.escape(kw)}</code>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if ctx_win:
                    st.markdown(
                        f'<div style="margin-top:10px;font-size:.78rem;color:{MUTE};'
                        f'font-weight:600;letter-spacing:.05em;text-transform:uppercase">Context window</div>',
                        unsafe_allow_html=True,
                    )
                    for ctx in ctx_win[:4]:
                        ctx_spk  = ctx.get("speaker_display") or ctx.get("speaker", "")
                        ctx_text = _html.escape(ctx.get("text", ""))
                        st.markdown(
                            f'<div style="padding:6px 10px;border-left:2px solid {LINE};'
                            f'margin:4px 0;font-size:.85rem;color:{MUTE}">'
                            f'<span style="color:{EM};font-weight:600">{ctx_spk}</span>'
                            f'&ensp;{ctx_text}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                f'<div class="cg-seg">'
                f'<span class="cg-ts">{ts}</span>&ensp;'
                f'<span class="cg-spk">{spk}</span>'
                f'<br/>{text_hi}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Risk ring
# ─────────────────────────────────────────────────────────────────────────────

def risk_ring(score: int | float) -> None:
    """
    SVG circular gauge.

    Stroke animates from 0 to score over 1.5 s.
    Number counts up in Instrument Serif.
    Colour: emerald under 70, alert-red at 70+.
    Labels: Low < 30 / Medium 30–69 / High ≥ 70.
    """
    score = max(0, min(100, int(score)))
    colour = ALERT if score >= 70 else EM
    label  = "High" if score >= 70 else ("Medium" if score >= 30 else "Low")
    label_colour = ALERT if score >= 70 else (EM if score >= 30 else MUTE)

    R          = 52       # circle radius
    CX = CY    = 64       # centre
    CIRCUMF    = 2 * math.pi * R
    dash_val   = CIRCUMF * (score / 100)

    unique_id  = f"ring_{score}_{id(score)}"

    svg = f"""
<div class="cg-ring-wrap">
  <svg width="128" height="128" viewBox="0 0 128 128" role="img"
       aria-label="Risk score {score} out of 100, {label}">
    <!-- track -->
    <circle cx="{CX}" cy="{CY}" r="{R}"
            fill="none" stroke="{SURFACE2}" stroke-width="10"/>
    <!-- filled arc -->
    <circle id="{unique_id}" cx="{CX}" cy="{CY}" r="{R}"
            fill="none"
            stroke="{colour}"
            stroke-width="10"
            stroke-linecap="round"
            stroke-dasharray="{CIRCUMF:.2f}"
            stroke-dashoffset="{CIRCUMF:.2f}"
            transform="rotate(-90 {CX} {CY})">
      <animate
        attributeName="stroke-dashoffset"
        from="{CIRCUMF:.2f}"
        to="{CIRCUMF - dash_val:.2f}"
        dur="1.5s"
        fill="freeze"
        calcMode="spline"
        keyTimes="0;1"
        keySplines="0.4 0 0.2 1"/>
    </circle>
    <!-- score number -->
    <text x="{CX}" y="{CY + 8}"
          text-anchor="middle"
          font-family="Instrument Serif, Georgia, serif"
          font-size="28"
          fill="{colour}"
          id="{unique_id}_txt">0</text>
  </svg>
  <div style="text-align:center">
    <span class="cg-pill" style="background:{'rgba(255,107,94,.15)' if score>=70 else 'rgba(63,224,160,.12)'};
          color:{label_colour};border:1px solid {'rgba(255,107,94,.3)' if score>=70 else 'rgba(63,224,160,.25)'};
          font-size:.85rem;padding:4px 16px">{label}</span>
  </div>
</div>

<script>
(function() {{
  const el  = document.getElementById("{unique_id}_txt");
  const target = {score};
  const dur    = 1500;
  const start  = performance.now();
  if (!el) return;
  function tick(now) {{
    const t   = Math.min((now - start) / dur, 1);
    const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
    el.textContent = Math.round(ease * target);
    if (t < 1) requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script>
"""
    components.html(svg, height=210)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Category bars
# ─────────────────────────────────────────────────────────────────────────────

def category_bars(summary: dict) -> None:
    """
    Animated horizontal bars per threat category.

    summary["by_category"] = {"threat": 2, "abuse": 1, ...}
    Bars are red when count > 0, muted grey when 0.
    Width animates from 0 on load via CSS keyframes.
    """
    by_cat = summary.get("by_category") or {}
    max_count = max(by_cat.values(), default=1) or 1

    all_cats = ["threat", "abuse", "harm_planning", "disaster"]
    rows_html = ""
    for cat in all_cats:
        count = by_cat.get(cat, 0)
        label = CATEGORY_LABELS.get(cat, cat)
        pct   = (count / max_count * 100) if max_count else 0
        zero_cls = "cg-hbar-zero" if count == 0 else ""
        rows_html += f"""
<div style="margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;
              font-size:.82rem;margin-bottom:4px">
    <span style="color:{'var(--text)' if count > 0 else 'var(--mute)'}">{label}</span>
    <span style="color:{'var(--alert)' if count > 0 else 'var(--dim)'}">
      {count} flag{'s' if count != 1 else ''}
    </span>
  </div>
  <div class="cg-hbar-outer {zero_cls}">
    <div class="cg-hbar-fill" style="width:{max(pct, 2 if count > 0 else 0):.1f}%;
         {'background:var(--alert)' if count > 0 else 'background:var(--surface2)'}">
    </div>
  </div>
</div>
"""

    st.markdown(rows_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Flag timeline
# ─────────────────────────────────────────────────────────────────────────────

def flag_timeline(flags: list[dict], duration: float) -> None:
    """
    Compact SVG scatter plot.

    x-axis = time (seconds),  one row per category,
    dot size proportional to confidence, with time-axis ticks.
    Renders as an inline SVG via st.markdown.
    """
    if not flags or duration <= 0:
        st.markdown(
            f'<div style="color:{MUTE};font-size:.85rem;'
            f'padding:1rem 0;text-align:center">No flags to display.</div>',
            unsafe_allow_html=True,
        )
        return

    cats       = ["threat", "abuse", "harm_planning", "disaster"]
    cat_index  = {c: i for i, c in enumerate(cats)}
    cat_labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    W, H        = 560, 180
    PAD_L       = 110   # left padding for labels
    PAD_R       = 20
    PAD_T       = 20
    PAD_B       = 32
    PLOT_W      = W - PAD_L - PAD_R
    PLOT_H      = H - PAD_T - PAD_B
    ROW_H       = PLOT_H / len(cats)

    def tx(t: float) -> float:
        return PAD_L + (t / duration) * PLOT_W

    def ty(ci: int) -> float:
        return PAD_T + ci * ROW_H + ROW_H / 2

    # Background rows
    rows_svg = ""
    for i in range(len(cats)):
        y   = PAD_T + i * ROW_H
        clr = SURFACE2 if i % 2 == 0 else SURFACE
        rows_svg += f'<rect x="{PAD_L}" y="{y:.1f}" width="{PLOT_W}" height="{ROW_H:.1f}" fill="{clr}"/>'

    # Category labels
    labels_svg = ""
    for i, lbl in enumerate(cat_labels):
        y = ty(i)
        labels_svg += (
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" font-family="Hanken Grotesk,sans-serif" '
            f'font-size="11" fill="{MUTE}">{lbl}</text>'
        )

    # Time ticks
    NUM_TICKS = 6
    ticks_svg = ""
    for i in range(NUM_TICKS + 1):
        t   = duration * i / NUM_TICKS
        x   = tx(t)
        m   = int(t) // 60
        s   = int(t) % 60
        lbl = f"{m}:{s:02d}"
        ticks_svg += (
            f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H - PAD_B}" '
            f'stroke="{LINE}" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{H - PAD_B + 14}" '
            f'text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="{DIM}">{lbl}</text>'
        )

    # Dots
    dots_svg = ""
    for f in flags:
        ts   = f.get("timestamp_sec") or f.get("start") or 0.0
        cat  = f.get("category", "")
        conf = f.get("confidence", 0.5)
        ci   = cat_index.get(cat, 0)
        x    = tx(min(ts, duration))
        y    = ty(ci)
        r    = max(4, min(11, conf * 14))
        tip  = _html.escape(f.get("matched_keyword", cat))
        dots_svg += (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill="{ALERT}" fill-opacity=".85" stroke="{SURFACE}" stroke-width="1.5">'
            f'<title>{tip} ({conf:.0%})</title></circle>'
        )

    svg_html = f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:{W}px;overflow:visible;display:block"
     role="img" aria-label="Flag timeline chart">
  <rect width="{W}" height="{H}" fill="{SURFACE}" rx="12"/>
  {rows_svg}
  {ticks_svg}
  {labels_svg}
  {dots_svg}
</svg>
"""
    st.markdown(svg_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Speaker share
# ─────────────────────────────────────────────────────────────────────────────

def speaker_share(segments: list[dict | object]) -> None:
    """
    Two-color split bar showing speaking-time percentage per speaker.

    Emerald for the first speaker, white (20% opacity) for the second,
    and muted green for any additional speakers.
    Percentages are shown above the bar.
    """
    # Accumulate duration per speaker ——————————————————————————————————
    totals: dict[str, float] = {}
    for seg in segments:
        if isinstance(seg, dict):
            spk   = seg.get("speaker_display") or seg.get("matched_name") or seg.get("speaker", "?")
            start = seg.get("start", 0)
            end   = seg.get("end",   0)
        else:
            ms = getattr(seg, "match_status", "not_run")
            mn = getattr(seg, "matched_name", "")
            if ms == "matched" and mn:
                spk = mn
            else:
                spk = getattr(seg, "speaker", "?")
            start = getattr(seg, "start_sec", 0)
            end   = getattr(seg, "end_sec",   0)

        totals[spk] = totals.get(spk, 0) + max(0, end - start)

    if not totals:
        return

    total_dur = sum(totals.values()) or 1
    speakers  = sorted(totals.items(), key=lambda x: -x[1])

    COLOURS = [EM, "rgba(238,245,239,.22)", MUTE, DIM]

    # Labels row
    labels_html = '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px">'
    for i, (spk, dur) in enumerate(speakers):
        pct   = dur / total_dur * 100
        color = COLOURS[min(i, len(COLOURS) - 1)]
        labels_html += (
            f'<span style="font-size:.82rem;color:{TEXT}">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{color};margin-right:5px;'
            f'vertical-align:middle"></span>'
            f'{_html.escape(spk)}&ensp;'
            f'<span style="color:{MUTE}">{pct:.0f}%</span>'
            f'</span>'
        )
    labels_html += '</div>'

    # Split bar
    bar_html = '<div class="cg-share-bar-outer">'
    for i, (spk, dur) in enumerate(speakers):
        pct   = dur / total_dur * 100
        color = COLOURS[min(i, len(COLOURS) - 1)]
        bar_html += (
            f'<div class="cg-share-bar-fill" '
            f'style="width:{pct:.1f}%;background:{color};float:left;'
            f'animation-delay:{i * 0.1:.1f}s"></div>'
        )
    bar_html += '</div><div style="clear:both"></div>'

    st.markdown(labels_html + bar_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pipeline steps bar
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_steps(active_index: int) -> str:
    """
    Render the 4-step pipeline progress bar as an HTML string
    (caller should pass it to an st.empty().markdown(...)).

    active_index:
      -1  = not started (all grey)
       0  = Preprocessing active
       1  = Transcribing active  (Preprocessing done)
       2  = Analyzing active     (Preprocessing + Transcribing done)
       3  = Saving active        (first three done)
       4  = All done (all emerald)
    """
    steps_html = '<div class="cg-pipeline" role="status" aria-label="Pipeline progress">'
    for i, label in enumerate(STEP_LABELS):
        if i < active_index:
            cls = "cg-step cg-step-done"
            icon = "✓ "
        elif i == active_index:
            cls = "cg-step cg-step-active"
            icon = ""
        else:
            cls = "cg-step"
            icon = ""
        steps_html += f'<div class="{cls}" aria-current="{"step" if i==active_index else "false"}">{icon}{label}</div>'
    steps_html += "</div>"
    return steps_html


def render_pipeline_steps(placeholder, active_index: int) -> None:
    """Convenience wrapper: updates an st.empty() placeholder."""
    placeholder.markdown(pipeline_steps(active_index), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar helpers
# ─────────────────────────────────────────────────────────────────────────────

SHIELD_SVG = """
<svg width="36" height="40" viewBox="0 0 36 40" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <!-- shield body -->
  <path d="M18 2L4 8v12c0 9 6.5 16.5 14 18 7.5-1.5 14-9 14-18V8L18 2Z"
        fill="none" stroke="#3fe0a0" stroke-width="2" stroke-linejoin="round"/>
  <!-- waveform bars inside shield -->
  <rect x="9"  y="17" width="2.2" height="8"  rx="1.1" fill="#3fe0a0" opacity=".5"/>
  <rect x="13" y="13" width="2.2" height="14" rx="1.1" fill="#3fe0a0" opacity=".8"/>
  <rect x="17" y="15" width="2.2" height="10" rx="1.1" fill="#3fe0a0"/>
  <rect x="21" y="12" width="2.2" height="16" rx="1.1" fill="#3fe0a0" opacity=".8"/>
  <rect x="25" y="17" width="2.2" height="8"  rx="1.1" fill="#3fe0a0" opacity=".5"/>
</svg>
"""


def sidebar_brand() -> None:
    """Render the CallGuard wordmark + SVG logo in the sidebar."""
    st.sidebar.markdown(
        f"""
<div style="display:flex;align-items:center;gap:12px;padding:0 4px 20px">
  {SHIELD_SVG}
  <div>
    <div style="font-family:'Instrument Serif',Georgia,serif;
                font-size:1.55rem;color:#eef5ef;line-height:1.1">CallGuard</div>
    <div style="font-size:.72rem;color:{MUTE};letter-spacing:.06em;
                text-transform:uppercase;margin-top:1px">Threat Detection</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar_engine_card(model: str = "base", device: str = "cpu", hf_set: bool = False) -> None:
    """Render the 'Engine ready' status card at the bottom of the sidebar."""
    status_col = EM if hf_set else "#f0b429"
    status_txt = "Engine ready" if hf_set else "Diarization off"
    st.sidebar.markdown(
        f"""
<div class="cg-engine-card">
  <div style="margin-bottom:6px">
    <span class="cg-engine-dot"></span>
    <span style="color:{TEXT};font-weight:600">{status_txt}</span>
  </div>
  <div style="color:{MUTE}">whisperx · pyannote 3.1</div>
  <div style="color:{MUTE}">English, Hindi, Urdu</div>
  <div style="margin-top:6px;color:{DIM};font-size:.72rem">
    model: {model} · device: {device}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def dropzone_empty_state() -> None:
    """Render the animated dropzone empty-state above the file uploader."""
    st.markdown(
        """
<div class="cg-dropzone">
  <div class="cg-bars" aria-hidden="true">
    <div class="cg-bar"></div>
    <div class="cg-bar"></div>
    <div class="cg-bar"></div>
    <div class="cg-bar"></div>
    <div class="cg-bar"></div>
    <div class="cg-bar"></div>
    <div class="cg-bar"></div>
  </div>
  <div style="font-size:1rem;color:var(--text);font-weight:500;margin-bottom:6px">
    Drop an audio file here
  </div>
  <div style="font-size:.82rem;color:var(--mute)">
    mp3 · m4a · wav · flac · ogg &nbsp;·&nbsp; Converted to 16 kHz mono automatically
  </div>
  <div style="font-size:.75rem;color:var(--dim);margin-top:10px">
    Files stay on your machine. Nothing is uploaded to a third party.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def risk_pill(score: int | float) -> str:
    """Return an HTML pill string for inline use in tables / headings."""
    score = int(score)
    if score >= 70:
        label = "High"
        bg, fg, border = "rgba(255,107,94,.15)", ALERT, "rgba(255,107,94,.3)"
    elif score >= 30:
        label = "Medium"
        bg, fg, border = "rgba(240,180,41,.15)", "#f0b429", "rgba(240,180,41,.3)"
    else:
        label = "Low"
        bg, fg, border = "rgba(63,224,160,.12)", EM, "rgba(63,224,160,.25)"
    return (
        f'<span class="cg-pill" style="background:{bg};color:{fg};'
        f'border:1px solid {border}">{label} {score}</span>'
    )


def toast(message: str, kind: str = "success") -> None:
    """
    Show a styled inline toast/banner message.

    kind: "success" | "error" | "info" | "warning"
    """
    colours = {
        "success": (EM,     "rgba(63,224,160,.08)",  "rgba(63,224,160,.25)"),
        "error":   (ALERT,  "rgba(255,107,94,.08)",  "rgba(255,107,94,.25)"),
        "info":    ("#4fa8d5","rgba(79,168,213,.08)", "rgba(79,168,213,.25)"),
        "warning": ("#f0b429","rgba(240,180,41,.08)", "rgba(240,180,41,.25)"),
    }
    text_c, bg, border = colours.get(kind, colours["info"])
    icon = {"success": "✓", "error": "✕", "info": "ℹ", "warning": "⚠"}[kind]
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
        f'padding:12px 16px;margin:8px 0;font-size:.9rem;color:{text_c}">'
        f'{icon}&ensp;{_html.escape(message)}</div>',
        unsafe_allow_html=True,
    )
