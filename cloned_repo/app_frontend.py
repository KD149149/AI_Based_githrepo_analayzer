from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from app_backend import app
from config import config

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Contivex by JillaniSofTech",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ENTERPRISE THEME
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --surface-dark: #0f172a;
        --surface-card: #1e293b;
        --surface-hover: #334155;
        --border: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
        --gradient-2: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 30px rgba(0,0,0,0.5);
        --radius: 12px;
        --radius-sm: 8px;
        --radius-lg: 16px;
    }

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .main .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem;
        max-width: 1300px;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* REMOVE: header {visibility:hidden;} */
    .stDeployButton {display: none;}


    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }

    /* Sidebar logo area */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0 1.5rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-logo-icon {
        width: 42px;
        height: 42px;
        background: var(--gradient-1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: white;
        flex-shrink: 0;
    }

    .sidebar-logo-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }

    .sidebar-logo-sub {
        font-size: 0.7rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    /* Sidebar section headers */
    .sidebar-section {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted) !important;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(99,102,241,0.2);
    }

    /* Sidebar blog list item */
    .blog-item {
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        margin-bottom: 6px;
        background: rgba(30,41,59,0.5);
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .blog-item:hover {
        border-color: var(--primary);
        background: rgba(99,102,241,0.08);
    }
    .blog-item-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.3;
        margin-bottom: 2px;
    }
    .blog-item-date {
        font-size: 0.7rem;
        color: var(--text-muted);
    }

    /* Sidebar footer */
    .sidebar-footer {
        margin-top: 2rem;
        padding: 1rem 0;
        border-top: 1px solid var(--border);
        text-align: center;
    }
    .sidebar-footer-text {
        font-size: 0.72rem;
        color: var(--text-muted);
        line-height: 1.6;
    }
    .sidebar-footer a {
        color: var(--primary-light) !important;
        text-decoration: none;
        font-weight: 600;
    }

    /* ===== MAIN CONTENT ===== */

    /* Hero header */
    .hero-header {
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin-top: 4px;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--primary-light);
        margin-top: 8px;
    }
    .hero-right {
        text-align: right;
        padding-top: 0.5rem;
    }
    .hero-right a {
        color: var(--primary-light);
        text-decoration: none;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .hero-right-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 0.75rem 0 1.25rem 0;
        border: none;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 16px;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-1);
    }
    .stat-card-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 6px;
    }
    .stat-card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
    }
    .stat-card-sub {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Metric cards (Streamlit native override) */
    [data-testid="stMetric"] {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
        border-top: 3px solid var(--primary);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface-card);
        border-radius: var(--radius);
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.82rem;
        color: var(--text-secondary);
        background: transparent;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: var(--surface-hover);
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: var(--radius-sm);
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid var(--border);
        padding: 0.5rem 1.25rem;
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--gradient-1) !important;
        border: none !important;
        color: white !important;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    }

    /* Download buttons */
    .stDownloadButton > button {
        border-radius: var(--radius-sm);
        font-weight: 600;
        border: 1px solid var(--border);
        background: var(--surface-card);
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        border-color: var(--primary);
        background: rgba(99,102,241,0.08);
        transform: translateY(-1px);
    }

    /* ===== TEXT INPUTS ===== */
    .stTextArea textarea, .stTextInput input {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        background: rgba(15,23,42,0.6) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }

    /* ===== DATAFRAMES ===== */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: var(--radius-sm);
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: var(--radius-sm);
        border-left-width: 4px;
        font-size: 0.88rem;
    }

    /* ===== PROGRESS ===== */
    .stProgress > div > div {
        background: var(--gradient-1);
        border-radius: 4px;
    }

    /* ===== WELCOME STATE ===== */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    .welcome-desc {
        font-size: 0.95rem;
        color: var(--text-secondary);
        max-width: 500px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    /* Topic cards on welcome */
    .topic-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        max-width: 700px;
        margin: 0 auto;
    }
    .topic-card {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
        text-align: left;
        transition: all 0.2s ease;
    }
    .topic-card:hover {
        border-color: var(--primary);
        background: rgba(99,102,241,0.05);
    }
    .topic-card-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--primary-light);
        font-weight: 700;
        margin-bottom: 8px;
    }
    .topic-card-item {
        font-size: 0.82rem;
        color: var(--text-secondary);
        padding: 3px 0;
        line-height: 1.4;
    }

    /* ===== GENERATION PROGRESS ===== */
    .gen-status {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .gen-status-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 8px;
    }
    .gen-status-node {
        font-size: 1rem;
        font-weight: 700;
        color: var(--accent);
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.25rem;
    }
    .section-header-icon {
        width: 32px;
        height: 32px;
        background: rgba(99,102,241,0.15);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .section-header-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .section-header-count {
        font-size: 0.72rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* ===== EVIDENCE LIST ===== */
    .evidence-item {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin-bottom: 8px;
        transition: all 0.15s ease;
    }
    .evidence-item:hover {
        border-color: var(--primary);
    }
    .evidence-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    .evidence-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    .evidence-snippet {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 6px;
        line-height: 1.5;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }
    .app-footer-brand {
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.8;
    }
    .app-footer a {
        color: var(--primary-light);
        text-decoration: none;
        font-weight: 600;
    }

    /* ===== MISC ===== */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }

    pre {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
    }

    /* Radio clean up */
    .stRadio > div {
        gap: 2px;
    }
    .stRadio > div > label {
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
    }

    /* Date input */
    .stDateInput > div > div {
        border-radius: var(--radius-sm) !important;
    }

    /* Checkbox */
    .stCheckbox label span {
        font-size: 0.85rem !important;
    }

    /* Image styling */
    .element-container img {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-muted);
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        opacity: 0.5;
    }
    .empty-state-text {
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Loaded file indicator */
    .loaded-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--success);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UTILITIES
# =============================================================================
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r'[^a-z0-9 _-]+', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s[:100] or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(images_dir.parent)))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not any(images_dir.iterdir()):
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.name)
    return buf.getvalue()


def extract_latest_state(current: Dict, payload: Any) -> Dict:
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current.update(next(iter(payload.values())))
        else:
            current.update(payload)
    return current


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        final = graph_app.invoke(inputs)
        yield ("final", final)
        return
    except Exception:
        pass
    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        final = graph_app.invoke(inputs)
        yield ("final", final)
        return
    except Exception:
        pass
    final = graph_app.invoke(inputs)
    yield ("final", final)


# =============================================================================
# MARKDOWN RENDERER
# =============================================================================
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def render_markdown_with_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts = []
    last_pos = 0
    for m in matches:
        if m.start() > last_pos:
            parts.append(("text", md[last_pos:m.start()]))
        parts.append(("image", f"{m.group('alt').strip()}|||{m.group('src').strip()}"))
        last_pos = m.end()
    if last_pos < len(md):
        parts.append(("text", md[last_pos:]))

    i = 0
    while i < len(parts):
        kind, content = parts[i]
        if kind == "text":
            st.markdown(content, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = content.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "text":
            next_text = parts[i + 1][1].lstrip()
            if next_text:
                first_line = next_text.split('\n')[0].strip()
                m_cap = _CAPTION_RE.match(first_line)
                if m_cap:
                    caption = m_cap.group("cap")
                    rest = '\n'.join(next_text.split('\n')[1:])
                    parts[i + 1] = ("text", rest)

        if src.startswith(("http://", "https://")):
            st.image(src, caption=caption or alt or None, use_container_width=True)
        else:
            img_path = Path(src.lstrip("./")).resolve()
            if img_path.exists():
                st.image(str(img_path), caption=caption or alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


# =============================================================================
# BLOG MANAGEMENT
# =============================================================================
def list_past_blogs() -> List[Tuple[Path, str, float]]:
    output_dir = config.paths.output_dir
    if not output_dir.exists():
        return []
    blogs = []
    for p in output_dir.glob("*.md"):
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                title = extract_title_from_md(text, p.stem)
                blogs.append((p, title, p.stat().st_mtime))
            except Exception:
                continue
    blogs.sort(key=lambda x: x[2], reverse=True)
    return blogs


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.split('\n'):
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip() or fallback
    return fallback


def load_blog(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return {
            "plan": None, "evidence": [], "image_specs": [],
            "final": path.read_text(encoding="utf-8"),
            "errors": [], "source_file": str(path),
        }
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return None


# =============================================================================
# PROGRESS TRACKER
# =============================================================================
class ProgressTracker:
    def __init__(self, container: DeltaGenerator):
        self.container = container
        self.current_node = None
        self.state = {}
        self.logs = []

    def update(self, node: Optional[str], state: Dict[str, Any]):
        if node and node != self.current_node:
            self.current_node = node
            self.state.update(state)
            self._render()

    def _render(self):
        with self.container:
            st.empty()
            if self.current_node:
                st.markdown(f"""
                <div class="gen-status">
                    <div class="gen-status-label">Current Step</div>
                    <div class="gen-status-node">⚡ {self.current_node}</div>
                </div>
                """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            mode = self.state.get("mode", "—")
            c1.metric("Mode", mode.title() if mode else "—")
            c2.metric("Sources", len(self.state.get("evidence", [])))
            plan = self.state.get("plan")
            tasks = len(plan.get("tasks", [])) if isinstance(plan, dict) else 0
            c3.metric("Sections", tasks)
            sections = len(self.state.get("sections", []))
            c4.metric("Progress", f"{sections}/{tasks}" if tasks else "—")

    def log(self, message: str):
        self.logs.append(message)


# =============================================================================
# API KEY VALIDATION
# =============================================================================
def check_api_keys() -> bool:
    missing = config.get_missing_keys()
    if "openai" in missing:
        with st.sidebar.expander("⚠️ Configuration Required", expanded=True):
            st.error("**OpenAI API key is missing.**")
            st.code("OPENAI_API_KEY=sk-proj-your-key-here", language="bash")
            st.caption("[Get your key →](https://platform.openai.com/api-keys)")
            return False

    optional_missing = [k for k in missing if k != "openai"]
    if optional_missing:
        with st.sidebar.expander("ℹ️ Optional APIs", expanded=False):
            for key in optional_missing:
                if key == "tavily":
                    st.caption("• Tavily — Web research disabled")
                elif key == "google":
                    st.caption("• Google — Image generation disabled")
    return True


# =============================================================================
# RENDER HELPERS
# =============================================================================
def render_section_header(icon: str, title: str, count: str = ""):
    count_html = f'<span class="section-header-count">{count}</span>' if count else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-icon">{icon}</div>
        <div>
            <div class="section-header-text">{title}</div>
            {count_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(icon: str, text: str):
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">✍️</div>
            <div>
                <div class="sidebar-logo-text">Contivex</div>
                <div class="sidebar-logo-sub">AI Blog Engine</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not check_api_keys():
            st.stop()

        st.markdown('<div class="sidebar-section">New Blog</div>', unsafe_allow_html=True)

        topic = st.text_area(
            "Topic",
            height=100,
            placeholder="e.g., 'LangGraph orchestration patterns', 'RAG vs Fine-tuning comparison'",
            label_visibility="collapsed",
        )

        col_date, col_gen = st.columns([1, 1])
        with col_date:
            as_of = st.date_input("Date", value=date.today(), label_visibility="collapsed")
        with col_gen:
            generate_btn = st.button("✨ Generate", type="primary", use_container_width=True, disabled=not topic.strip())

        with st.expander("⚙️ Options", expanded=False):
            enable_research = st.checkbox("Force Web Research", value=False)
            enable_images = st.checkbox(
                "Generate Images",
                value=config.image.enable_generation and bool(config.google_api_key),
            )
            max_tasks = st.slider("Sections", min_value=3, max_value=15, value=config.blog.max_tasks)

        # Past blogs
        st.markdown('<div class="sidebar-section">Library</div>', unsafe_allow_html=True)
        past_blogs = list_past_blogs()

        if not past_blogs:
            st.caption("No saved blogs yet.")
        else:
            st.caption(f"{len(past_blogs)} blog(s)")
            blog_map = {}
            for path, title, mtime in past_blogs[:30]:
                d = date.fromtimestamp(mtime).strftime("%b %d, %Y")
                label = f"{title[:42]}{'…' if len(title) > 42 else ''}"
                key = f"{label}  •  {d}"
                blog_map[key] = path

            selected = st.radio("Select", options=list(blog_map.keys()), label_visibility="collapsed")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("📂 Load", use_container_width=True):
                    loaded = load_blog(blog_map[selected])
                    if loaded:
                        st.session_state.last_output = loaded
                        st.rerun()
            with c2:
                if st.button("🗑 Delete", use_container_width=True):
                    try:
                        blog_map[selected].unlink()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.markdown("""
        <div class="sidebar-footer">
            <div class="sidebar-footer-text">
                <strong>Contivex</strong> v1.0<br/>
                by <a href="https://jillanisoftech.com" target="_blank">JillaniSofTech</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ===== HEADER =====
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("""
        <div class="hero-header">
            <div class="hero-title">Contivex</div>
            <div class="hero-subtitle">AI-Powered Blog Generation Platform</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="hero-right">
            <div class="hero-right-label">Powered by</div>
            <a href="https://jillanisoftech.com" target="_blank">JillaniSofTech</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Session state init
    if "last_output" not in st.session_state:
        st.session_state.last_output = None
    if "execution_logs" not in st.session_state:
        st.session_state.execution_logs = []

    # ===== TABS =====
    tab_plan, tab_research, tab_preview, tab_images, tab_stats, tab_logs = st.tabs([
        "📋 Plan", "🔍 Research", "📝 Preview", "🖼️ Images", "📊 Stats", "📜 Logs"
    ])

    # ===== GENERATION =====
    if generate_btn:
        if not topic.strip():
            st.error("Please enter a topic.")
            st.stop()

        inputs = {
            "topic": topic.strip(), "mode": "", "needs_research": enable_research,
            "queries": [], "evidence": [], "plan": None, "as_of": as_of.isoformat(),
            "recency_days": 7, "sections": [], "merged_md": "", "md_with_placeholders": "",
            "image_specs": [], "final": "", "errors": [],
        }
        config.blog.max_tasks = max_tasks
        config.image.enable_generation = enable_images

        with st.container():
            st.markdown("### ⚡ Generating...")
            progress_container = st.container()
            status_area = st.empty()
            tracker = ProgressTracker(progress_container)
            current_state = {}
            last_node = None

            try:
                for kind, payload in try_stream(app, inputs):
                    if kind in ("updates", "values"):
                        node_name = None
                        if isinstance(payload, dict) and len(payload) == 1:
                            node_name = next(iter(payload.keys()))
                        if node_name and node_name != last_node:
                            status_area.info(f"▶️ {node_name}")
                            last_node = node_name
                        current_state = extract_latest_state(current_state, payload)
                        tracker.update(node_name, current_state)
                        tracker.log(f"[{kind}] {node_name}")
                    elif kind == "final":
                        st.session_state.last_output = payload
                        st.session_state.execution_logs = tracker.logs
                        status_area.success("✅ Blog generated successfully.")
                        tracker.log("[COMPLETE]")
                        errors = payload.get("errors", [])
                        if errors:
                            with st.expander("⚠️ Warnings"):
                                for err in errors:
                                    st.warning(err)
                        st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                tracker.log(f"[ERROR] {str(e)}")
                st.session_state.execution_logs = tracker.logs
                st.stop()

    # ===== OUTPUT =====
    output = st.session_state.get("last_output")

    if not output:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">✍️</div>
            <div class="welcome-title">Ready to create something great?</div>
            <div class="welcome-desc">
                Enter a topic in the sidebar and hit Generate.
                Contivex handles research, planning, writing, and visuals — all automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="topic-grid">
            <div class="topic-card">
                <div class="topic-card-label">Tutorial / Explainer</div>
                <div class="topic-card-item">→ Complete guide to Python async/await</div>
                <div class="topic-card-item">→ Understanding Kubernetes architecture</div>
                <div class="topic-card-item">→ Vector databases explained</div>
            </div>
            <div class="topic-card">
                <div class="topic-card-label">News / Current</div>
                <div class="topic-card-item">→ AI developments this week</div>
                <div class="topic-card-item">→ Latest AWS announcements</div>
                <div class="topic-card-item">→ Best Python frameworks 2026</div>
            </div>
            <div class="topic-card">
                <div class="topic-card-label">Comparison</div>
                <div class="topic-card-item">→ RAG vs Fine-tuning vs MCP</div>
                <div class="topic-card-item">→ FastAPI vs Flask for microservices</div>
                <div class="topic-card-item">→ LangChain vs LlamaIndex</div>
            </div>
            <div class="topic-card">
                <div class="topic-card-label">System Design</div>
                <div class="topic-card-item">→ Building scalable RAG pipelines</div>
                <div class="topic-card-item">→ Multi-agent orchestration patterns</div>
                <div class="topic-card-item">→ Event-driven ML architectures</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if output.get("source_file"):
        st.markdown(f'<div class="loaded-indicator">📂 Loaded: {Path(output["source_file"]).name}</div>', unsafe_allow_html=True)

    # ===== PLAN TAB =====
    with tab_plan:
        render_section_header("📋", "Blog Structure")
        plan_obj = output.get("plan")

        if not plan_obj:
            render_empty_state("📋", "No plan available.")
        else:
            plan = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else (plan_obj if isinstance(plan_obj, dict) else json.loads(json.dumps(plan_obj, default=str)))

            st.markdown(f"#### {plan.get('blog_title', 'Untitled')}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Audience", plan.get("audience", "—"))
            c2.metric("Tone", plan.get("tone", "—"))
            c3.metric("Type", plan.get("blog_kind", "—"))

            constraints = plan.get("constraints", [])
            if constraints:
                with st.expander("📌 Constraints"):
                    for c in constraints:
                        st.markdown(f"• {c}")

            tasks = plan.get("tasks", [])
            if tasks:
                df = pd.DataFrame([{
                    "#": t.get("id"), "Title": t.get("title"),
                    "Words": t.get("target_words"),
                    "Research": "✅" if t.get("requires_research") else "—",
                    "Citations": "✅" if t.get("requires_citations") else "—",
                    "Code": "✅" if t.get("requires_code") else "—",
                } for t in tasks])
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("🔍 Section Details"):
                    for t in tasks:
                        st.markdown(f"**{t.get('id')}. {t.get('title')}**")
                        st.caption(t.get("goal", ""))
                        for b in t.get("bullets", []):
                            st.markdown(f"  • {b}")
                        st.divider()

    # ===== RESEARCH TAB =====
    with tab_research:
        evidence = output.get("evidence", [])
        render_section_header("🔍", "Research Sources", f"{len(evidence)} sources" if evidence else "")

        if not evidence:
            render_empty_state("🔍", "No research performed — closed-book mode.")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append({
                    "Title": e.get("title", "")[:60], "Source": e.get("source", "—"),
                    "Date": e.get("published_at", "—"), "URL": e.get("url", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("📄 Full Details"):
                for i, e in enumerate(evidence, 1):
                    if hasattr(e, "model_dump"):
                        e = e.model_dump()
                    st.markdown(f"**{i}. [{e.get('title')}]({e.get('url')})**")
                    st.caption(f"{e.get('source', '—')} • {e.get('published_at', '—')}")
                    snippet = e.get("snippet", "")
                    if snippet:
                        st.markdown(f"> {snippet[:300]}{'…' if len(snippet) > 300 else ''}")
                    st.divider()

    # ===== PREVIEW TAB =====
    with tab_preview:
        final_md = output.get("final", "")
        if not final_md:
            render_empty_state("📝", "No content available.")
        else:
            words = final_md.split()
            word_count = len(words)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words", f"{word_count:,}")
            c2.metric("Characters", f"{len(final_md):,}")
            c3.metric("Read Time", f"{word_count // 200} min")
            c4.metric("Sections", len([l for l in final_md.split('\n') if l.startswith('## ')]))

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            render_markdown_with_images(final_md)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Downloads
            plan_obj = output.get("plan")
            if plan_obj:
                blog_title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else (plan_obj.get("blog_title", "blog") if isinstance(plan_obj, dict) else extract_title_from_md(final_md, "blog"))
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            filename = f"{safe_slug(blog_title)}.md"

            render_section_header("📥", "Export")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("📄 Markdown", data=final_md.encode("utf-8"), file_name=filename, mime="text/markdown", use_container_width=True)
            with c2:
                bundle = bundle_zip(final_md, filename, config.paths.images_dir)
                st.download_button("📦 Full Package", data=bundle, file_name=f"{safe_slug(blog_title)}.zip", mime="application/zip", use_container_width=True)
            with c3:
                img_zip = images_zip(config.paths.images_dir)
                if img_zip:
                    st.download_button("🖼️ Images Only", data=img_zip, file_name="images.zip", mime="application/zip", use_container_width=True)
                else:
                    st.button("🖼️ No Images", disabled=True, use_container_width=True)

    # ===== IMAGES TAB =====
    with tab_images:
        render_section_header("🖼️", "Generated Images")
        image_specs = output.get("image_specs", [])
        images_dir = config.paths.images_dir

        if not image_specs and not (images_dir.exists() and any(images_dir.iterdir())):
            render_empty_state("🖼️", "No images generated for this blog.")
        else:
            if image_specs:
                for i, spec in enumerate(image_specs, 1):
                    with st.expander(f"Image {i}: {spec.get('alt', 'Untitled')}", expanded=False):
                        st.markdown(f"**File:** `{spec.get('filename')}`")
                        st.markdown(f"**Caption:** {spec.get('caption')}")
                        st.code(spec.get('prompt', ''), language="text")

            if images_dir.exists():
                imgs = sorted([f for f in images_dir.glob("*") if f.is_file()])
                if imgs:
                    st.markdown("#### Preview")
                    cols = st.columns(2)
                    for idx, img in enumerate(imgs):
                        with cols[idx % 2]:
                            st.image(str(img), caption=img.name, use_container_width=True)

    # ===== STATS TAB =====
    with tab_stats:
        render_section_header("📊", "Content Analytics")
        final_md = output.get("final", "")

        if not final_md:
            render_empty_state("📊", "No content to analyze.")
        else:
            words = final_md.split()
            lines = final_md.split('\n')
            h2_sections = [l for l in lines if l.startswith('## ')]
            code_blocks = len([l for l in lines if l.startswith('```')]) // 2

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Words", f"{len(words):,}")
            c2.metric("Sections", len(h2_sections))
            c3.metric("Code Blocks", code_blocks)
            c4.metric("Read Time", f"{len(words) // 200} min")

            plan_obj = output.get("plan")
            if plan_obj and hasattr(plan_obj, "tasks"):
                sections_data = output.get("sections", [])
                if sections_data:
                    task_words = []
                    for task_id, section_md in sorted(sections_data, key=lambda x: x[0]):
                        wc = len(section_md.split())
                        task = next((t for t in plan_obj.tasks if t.id == task_id), None)
                        title = task.title if task else f"Section {task_id}"
                        target = task.target_words if task else 0
                        variance = ((wc - target) / target * 100) if target else 0
                        task_words.append({"Section": title[:40], "Words": wc, "Target": target, "Diff": f"{variance:+.0f}%"})

                    if task_words:
                        st.markdown("#### Section Distribution")
                        st.dataframe(pd.DataFrame(task_words), use_container_width=True, hide_index=True)

    # ===== LOGS TAB =====
    with tab_logs:
        render_section_header("📜", "Execution Log")
        logs = st.session_state.get("execution_logs", [])

        if not logs:
            render_empty_state("📜", "No execution logs yet. Generate a blog to see logs.")
        else:
            filter_text = st.text_input("Filter", placeholder="Search logs...", label_visibility="collapsed")
            filtered = [l for l in logs if filter_text.lower() in l.lower()] if filter_text else logs

            st.text_area(
                f"{len(filtered)}/{len(logs)} entries",
                value="\n".join(filtered[-100:]),
                height=400,
                label_visibility="collapsed",
            )
            st.download_button(
                "📥 Download Log",
                data="\n".join(logs),
                file_name=f"contivex_log_{date.today().isoformat()}.txt",
                mime="text/plain",
            )

    # ===== FOOTER =====
    st.markdown("""
    <div class="app-footer">
        <div class="app-footer-brand">
            <strong>Contivex</strong> — AI Blog Generation Platform<br/>
            Built by <a href="https://jillanisoftech.com" target="_blank">JillaniSofTech</a> •
            AI & Intelligent Automation Solutions
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()