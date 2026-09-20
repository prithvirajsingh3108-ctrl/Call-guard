"""
ui/theme.py
───────────
inject_theme() — injects all global CSS into the Streamlit page via
st.markdown(unsafe_allow_html=True).

Design tokens (CSS variables):
  --bg         #040906   page background
  --bg2        #08110b   slightly lighter bg
  --surface    #0b150f   card background
  --surface2   #111f16   elevated card / hover
  --line       rgba(214,238,222,.09)  border colour
  --text       #eef5ef   primary text
  --mute       #8ba493   muted / secondary text
  --dim        #5b7263   dimmed labels
  --em         #3fe0a0   emerald accent
  --alert      #ff6b5e   threat / danger colour

Fonts: "Instrument Serif" (headings) + "Hanken Grotesk" (body) via Google Fonts.
"""

import streamlit as st


def inject_theme() -> None:
    """Call once at the top of app.py, after st.set_page_config()."""
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Hanken+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ─── Design tokens ─────────────────────────────────────────────────────── */
:root {
  --bg:       #040906;
  --bg2:      #08110b;
  --surface:  #0b150f;
  --surface2: #111f16;
  --line:     rgba(214,238,222,.09);
  --text:     #eef5ef;
  --mute:     #8ba493;
  --dim:      #5b7263;
  --em:       #3fe0a0;
  --alert:    #ff6b5e;
  --radius:   22px;
  --shadow:   0 8px 40px rgba(0,0,0,.55), 0 2px 8px rgba(0,0,0,.4);
}

/* ─── Global background / colour ────────────────────────────────────────── */
/* NOTE: font-family is NOT set here — that would override Streamlit's       */
/* Material Symbols icon font and render "upload", "expand_more", etc.       */
html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* ─── App font — applied only to content nodes, never to icon spans ──────── */
/* Bug 1 fix: target specific semantic elements so Material Symbols ligatures */
/* inside <span class="material-symbols-rounded"> are never overridden.       */
html,
body,
[data-testid="stApp"],
[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] p,
label,
input,
textarea,
[data-baseweb="select"],
.stButton > button,
.stTextInput input,
.stSelectbox,
.stRadio,
.stCheckbox {
  font-family: "Hanken Grotesk", system-ui, -apple-system, sans-serif !important;
}

/* ─── Bug 1: Protect Material Symbols icon font — MUST come after the rule  */
/* above so specificity wins. Ensures icon ligatures render as glyphs.        */
[data-testid="stIconMaterial"],
span[class*="material"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined",
               "Material Icons" !important;
  font-feature-settings: "liga" !important;
  -webkit-font-feature-settings: "liga" !important;
  font-variant-ligatures: discretionary-ligatures !important;
}

/* Radial glows */
[data-testid="stApp"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 60% 50% at 0% 0%,    rgba(63,224,160,.08) 0%, transparent 70%),
    radial-gradient(ellipse 55% 45% at 100% 100%, rgba(63,224,160,.06) 0%, transparent 70%);
  z-index: 0;
}

/* ─── Hide default Streamlit chrome ─────────────────────────────────────── */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }

/* ─── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.5rem !important;
}

/* Sidebar nav radio — hide auto-generated label */
[data-testid="stSidebar"] .stRadio > label:first-child {
  display: none !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
  gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio label {
  border-radius: 12px !important;
  padding: 10px 14px !important;
  transition: background .18s ease, color .18s ease !important;
  font-size: 0.93rem !important;
  font-weight: 500 !important;
  color: var(--mute) !important;
  cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: var(--surface2) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] input:checked ~ div {
  color: var(--em) !important;
}

/* ─── Main content area ──────────────────────────────────────────────────── */
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1280px !important;
}

/* ─── Bug 3: Heading font — Instrument Serif ─────────────────────────────── */
/* Target both raw h-tags and Streamlit's stHeading wrapper.                  */
h1, h2, h3,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
  font-family: "Instrument Serif", Georgia, serif !important;
  color: var(--text) !important;
  font-weight: 400 !important;
  letter-spacing: -0.02em !important;
}
h1, [data-testid="stHeading"] h1 { font-size: 2.6rem !important; line-height: 1.15 !important; }
h2, [data-testid="stHeading"] h2 { font-size: 1.8rem !important; }
h3, [data-testid="stHeading"] h3 { font-size: 1.3rem !important; }

/* Body copy — scoped selectors only, no broad * or span/div overrides */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] a {
  font-family: "Hanken Grotesk", system-ui, sans-serif !important;
  color: var(--text);
}

/* ─── Cards ──────────────────────────────────────────────────────────────── */
.cg-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.5rem 1.75rem;
  position: relative;
  overflow: hidden;
}
.cg-card + .cg-card { margin-top: 1.25rem; }

/* ─── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
  color: var(--mute) !important;
  font-size: 0.8rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-family: "Instrument Serif", serif !important;
  font-size: 2rem !important;
}

/* ─── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
  border-radius: 999px !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  transition: all .18s ease !important;
  border: 1px solid var(--line) !important;
  background: var(--surface2) !important;
  color: var(--text) !important;
}
.stButton > button:hover {
  background: var(--surface2) !important;
  border-color: var(--em) !important;
  color: var(--em) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--em) !important;
  color: #040906 !important;
  border-color: transparent !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #5aeab4 !important;
  color: #040906 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(63,224,160,.35) !important;
}

/* ─── Bug 2: Merged dropzone — the animated card IS the only border ──────── */
/* .st-key-dropzone wraps our custom bars + the native file uploader.         */
/* The native uploader's own box is stripped to invisible inside it.          */

/* Outer container = the one dashed border */
.st-key-dropzone {
  border: 1.5px dashed rgba(63,224,160,.35) !important;
  border-radius: 26px !important;
  background: var(--surface) !important;
  padding: 2.5rem 2rem 1.5rem !important;
  text-align: center !important;
  transition: border-color .2s, box-shadow .2s !important;
  position: relative !important;
}
/* Drag-over highlight */
.st-key-dropzone:has([data-testid="stFileUploaderDropzone"]:hover),
.st-key-dropzone:focus-within {
  border-color: var(--em) !important;
  box-shadow: 0 0 0 4px rgba(63,224,160,.12) !important;
}

/* Strip ALL borders/backgrounds from the native uploader inside our card */
.st-key-dropzone [data-testid="stFileUploader"] {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}
.st-key-dropzone [data-testid="stFileUploaderDropzone"] {
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  justify-content: center !important;
  box-shadow: none !important;
}

/* Hide the native "Drag and drop file here" instruction — our custom text
   already says this; the size-limit line stays visible but smaller */
.st-key-dropzone [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
  display: none !important;
}
.st-key-dropzone [data-testid="stFileUploaderDropzoneInstructions"] > div > small,
.st-key-dropzone [data-testid="stFileUploaderDropzoneInstructions"] small {
  font-size: 12px !important;
  color: var(--dim) !important;
}

/* Browse files button → emerald pill with dark text */
.st-key-dropzone [data-testid="stFileUploaderDropzone"] button,
.st-key-dropzone [data-testid="stBaseButton-secondary"] {
  background: var(--em) !important;
  color: #03130b !important;
  border: none !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  padding: 8px 22px !important;
  transition: background .18s, transform .18s, box-shadow .18s !important;
}
.st-key-dropzone [data-testid="stFileUploaderDropzone"] button:hover,
.st-key-dropzone [data-testid="stBaseButton-secondary"]:hover {
  background: #5aeab4 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(63,224,160,.35) !important;
}

/* File uploader OUTSIDE the dropzone (e.g. enroll voice page) — minimal style */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed rgba(63,224,160,.25) !important;
  border-radius: 16px !important;
  padding: 0.75rem !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--em) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: transparent !important;
  border: none !important;
}

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 6px !important;
  border-bottom: 1px solid var(--line) !important;
  padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 10px 10px 0 0 !important;
  color: var(--mute) !important;
  font-weight: 500 !important;
  padding: 10px 20px !important;
  border: 1px solid transparent !important;
  border-bottom: none !important;
  transition: color .2s, background .2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text) !important;
  background: var(--surface) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--em) !important;
  background: var(--surface) !important;
  border-color: var(--line) !important;
  border-bottom: 2px solid var(--em) !important;
}

/* ─── Expanders ──────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  margin-bottom: 6px !important;
  overflow: hidden !important;
}
details[data-testid="stExpander"] summary {
  padding: 12px 16px !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: background .15s !important;
}
details[data-testid="stExpander"] summary:hover {
  background: var(--surface2) !important;
}
details[data-testid="stExpander"][open] summary {
  border-bottom: 1px solid var(--line) !important;
}

/* ─── Dataframe / table ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid var(--line) !important;
}
[data-testid="stDataFrame"] thead th {
  background: var(--surface2) !important;
  color: var(--mute) !important;
  text-transform: uppercase !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.06em !important;
  font-weight: 600 !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
  background: var(--surface2) !important;
}

/* ─── Progress bar ───────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--em), #5aeab4) !important;
  border-radius: 999px !important;
}
[data-testid="stProgress"] > div {
  background: var(--surface2) !important;
  border-radius: 999px !important;
}

/* ─── Alerts / info boxes ────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 14px !important;
  border-left-width: 3px !important;
}
.stSuccess { border-color: var(--em) !important; background: rgba(63,224,160,.08) !important; }
.stInfo    { border-color: #4fa8d5 !important;   background: rgba(79,168,213,.08) !important; }
.stWarning { border-color: #f0b429 !important;   background: rgba(240,180,41,.08) !important; }
.stError   { border-color: var(--alert) !important; background: rgba(255,107,94,.08) !important; }

/* ─── Selectbox / text inputs ────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
  background: var(--surface2) !important;
  border-color: var(--line) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stTextInput"] > div > div > input:focus {
  border-color: var(--em) !important;
  box-shadow: 0 0 0 2px rgba(63,224,160,.15) !important;
}

/* ─── Checkbox ───────────────────────────────────────────────────────────── */
input[type="checkbox"]:checked + div {
  background: var(--em) !important;
  border-color: var(--em) !important;
}

/* ─── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: var(--line) !important; }

/* ─── Custom pill badges ─────────────────────────────────────────────────── */
.cg-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  line-height: 1.6;
}
.cg-pill-em    { background: rgba(63,224,160,.15);  color: var(--em);   border: 1px solid rgba(63,224,160,.3);  }
.cg-pill-alert { background: rgba(255,107,94,.15);  color: var(--alert); border: 1px solid rgba(255,107,94,.3); }
.cg-pill-mute  { background: rgba(139,164,147,.1);  color: var(--mute);  border: 1px solid var(--line);          }

/* ─── Transcript rows ────────────────────────────────────────────────────── */
.cg-seg {
  padding: 10px 14px;
  border-radius: 10px;
  margin-bottom: 5px;
  border: 1px solid var(--line);
  background: var(--surface);
  transition: background .15s;
}
.cg-seg:hover { background: var(--surface2); }
.cg-seg-flagged {
  background: rgba(255,107,94,.07) !important;
  border-color: rgba(255,107,94,.3) !important;
}
.cg-seg-cleared {
  background: rgba(139,164,147,.06) !important;
  border: 1px dashed rgba(139,164,147,.3) !important;
}
.cg-ts   { font-size: 0.75rem; color: var(--dim);  font-family: monospace; }
.cg-spk  { font-size: 0.8rem;  color: var(--em);   font-weight: 600; }
.cg-kw   { text-decoration: underline; text-decoration-color: var(--alert);
           text-decoration-thickness: 2px; font-weight: 600; }

/* ─── Pipeline step bar ──────────────────────────────────────────────────── */
.cg-pipeline {
  display: flex;
  gap: 0;
  margin: 1.5rem 0 1rem;
}
.cg-step {
  flex: 1;
  padding: 12px 16px;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--dim);
  border-bottom: 2px solid var(--line);
  position: relative;
  transition: color .3s;
  text-align: center;
}
.cg-step-done {
  color: var(--em) !important;
  border-bottom-color: var(--em) !important;
}
.cg-step-active {
  color: var(--text) !important;
  border-bottom-color: transparent !important;
}
.cg-step-active::after {
  content: "";
  position: absolute;
  bottom: -2px;
  left: 0;
  height: 2px;
  background: var(--em);
  animation: pipeline-fill 3s ease-in-out infinite;
}
@keyframes pipeline-fill {
  0%   { width: 0%;   opacity: 1; }
  80%  { width: 100%; opacity: 1; }
  100% { width: 100%; opacity: .5; }
}

/* ─── Dropzone / empty-state bars ────────────────────────────────────────── */
/* ─── Animated bars (used inline in the dropzone) ───────────────────────── */
/* .cg-bar is still used by the inline HTML in app.py's dropzone container.  */
.cg-bar {
  width: 5px;
  border-radius: 3px;
  background: var(--em);
  animation: breathe 1.6s ease-in-out infinite;
}
@keyframes breathe {
  0%, 100% { opacity: .35; transform: scaleY(.7);  }
  50%       { opacity: 1;   transform: scaleY(1);   }
}

/* ─── Engine-ready card ──────────────────────────────────────────────────── */
.cg-engine-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 14px;
  font-size: 0.8rem;
  color: var(--mute);
  margin-top: auto;
}
.cg-engine-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--em);
  margin-right: 6px;
  box-shadow: 0 0 6px var(--em);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50%       { opacity: .4; }
}

/* ─── Keyword chips ──────────────────────────────────────────────────────── */
.cg-kw-chip {
  display: inline-block;
  background: var(--surface2);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 3px 12px;
  font-size: 0.78rem;
  color: var(--mute);
  margin: 3px;
  transition: border-color .15s, color .15s;
}
.cg-kw-chip:hover {
  border-color: var(--em);
  color: var(--em);
}

/* ─── Risk ring container ────────────────────────────────────────────────── */
.cg-ring-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 1.25rem 0;
}

/* ─── Speaker share bar ──────────────────────────────────────────────────── */
.cg-share-bar-outer {
  height: 10px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--surface2);
  margin: 6px 0 10px;
}
.cg-share-bar-fill {
  height: 100%;
  border-radius: 999px;
  animation: grow-bar .8s ease-out forwards;
}
@keyframes grow-bar {
  from { width: 0; }
}

/* ─── Horizontal category bars ───────────────────────────────────────────── */
.cg-hbar-outer {
  height: 8px;
  border-radius: 999px;
  background: var(--surface2);
  overflow: hidden;
  margin: 4px 0 10px;
}
.cg-hbar-fill {
  height: 100%;
  border-radius: 999px;
  background: var(--alert);
  animation: grow-bar .7s ease-out forwards;
}
.cg-hbar-zero .cg-hbar-fill {
  background: var(--surface2);
}

/* ─── Audio player ───────────────────────────────────────────────────────── */
audio {
  width: 100% !important;
  border-radius: 12px !important;
  background: var(--surface2) !important;
  outline: none !important;
}

/* ─── Waveform canvas wrapper ────────────────────────────────────────────── */
.cg-waveform-wrap {
  border-radius: 16px;
  overflow: hidden;
  background: var(--surface);
  border: 1px solid var(--line);
}

/* ─── Responsive: stack columns on narrow screens ────────────────────────── */
@media (max-width: 768px) {
  .main .block-container { padding: 1rem 1rem 3rem !important; }
  h1 { font-size: 1.8rem !important; }
  .cg-pipeline { flex-wrap: wrap; }
  .cg-step { font-size: 0.7rem; padding: 8px 10px; }
}

/* ─── Focus / keyboard accessibility ────────────────────────────────────── */
:focus-visible {
  outline: 2px solid var(--em) !important;
  outline-offset: 2px !important;
}
button:focus-visible, a:focus-visible {
  box-shadow: 0 0 0 3px rgba(63,224,160,.4) !important;
}

/* ─── Reduced motion ─────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    transition-duration: .01ms !important;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )
