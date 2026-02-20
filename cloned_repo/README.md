<p align="center">
  <img src="https://img.shields.io/badge/Contivex-AI%20Blog%20Engine-6366f1?style=for-the-badge&labelColor=0f172a" alt="Contivex" />
</p>

<h1 align="center">✍️ Contivex</h1>

<p align="center">
  <strong>AI-Powered Blog Generation Platform</strong><br/>
  <em>Built by <a href="https://jillanisoftech.com">JillaniSofTech</a></em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/streamlit-1.54+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/langgraph-0.3+-10B981?style=flat-square" />
  <img src="https://img.shields.io/badge/openai-gpt--4.1-412991?style=flat-square&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/license-Proprietary-ef4444?style=flat-square" />
</p>

<p align="center">
  Transform a single topic into a comprehensive, research-backed, professionally-written blog post — with structured planning, parallel content generation, and AI-powered image creation.
</p>

---

## What is Contivex?

Contivex is a production-grade AI platform that takes a topic and delivers a complete blog post. It intelligently determines whether web research is needed, gathers evidence from authoritative sources, builds a structured outline, generates content across multiple sections in parallel, and creates technical diagrams — all automatically.

**Who it's for:** Technical writers, developer advocates, content teams, agencies, and startups that need consistent, high-quality content at scale.

---

## Features

**Intelligent Pipeline** — Smart routing detects whether your topic needs web research (closed-book, hybrid, or open-book modes). Tavily-powered research gathers real-time sources. Structured planner creates 3–15 section outlines. Parallel workers generate all sections simultaneously.

**Visual Content** — AI image generation via Google Gemini creates technical diagrams, architecture visuals, and flowcharts. Only generates images that add educational value.

**Enterprise Quality** — Pydantic validation throughout. Graceful degradation with detailed error reporting. Full execution logging. Centralized configuration management.

**Export & Library** — Download as Markdown, ZIP bundle (with images), or standalone images. Browse, load, and manage all generated blogs from the built-in library.

**Content Analytics** — Word count, reading time, section distribution, code block count, and target variance tracking.

---

## Architecture

```
START
  │
  ▼
┌──────────┐
│  Router  │ ── Analyzes topic, decides mode
└────┬─────┘
     │
     ├── closed_book ──────────────────┐
     │                                 │
     ▼                                 ▼
┌──────────┐                    ┌──────────────┐
│ Research │ ── Tavily search   │ Orchestrator │ ── Creates plan
└────┬─────┘                    └──────┬───────┘
     │                                 │
     └────────────────┬────────────────┘
                      │
                ┌─────┴─────┐
                │  Fanout   │ ── Parallel dispatch
                └─────┬─────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     ┌────────┐ ┌────────┐ ┌────────┐
     │Worker 1│ │Worker 2│ │Worker N│ ── Parallel section generation
     └────┬───┘ └────┬───┘ └────┬───┘
          └───────────┼───────────┘
                      │
                ┌─────┴─────┐
                │  Reducer  │ ── Subgraph
                │  ┌──────┐ │
                │  │Merge │ │
                │  └──┬───┘ │
                │     ▼     │
                │  ┌──────┐ │
                │  │Images│ │ ── Plan + Generate + Place
                │  └──┬───┘ │
                └─────┼─────┘
                      │
                      ▼
                    END
```

**Router** — Classifies topic into `closed_book` (evergreen), `hybrid` (evergreen + current data), or `open_book` (news-driven). Generates targeted search queries.

**Research** — Executes Tavily searches, synthesizes evidence, deduplicates, filters by recency window.

**Orchestrator** — Creates structured plan with goals, bullet points, word targets, and metadata per section.

**Workers** — Generate sections in parallel with citation management, code examples, and scope control.

**Reducer** — Merges sections, decides on image placements, generates diagrams via Gemini, saves output.

---

## Prerequisites

| Requirement | Purpose | Required |
|---|---|---|
| Python 3.10+ | Runtime | ✅ |
| OpenAI API Key | LLM operations | ✅ |
| Tavily API Key | Web research | Optional |
| Google API Key | Image generation (Gemini) | Optional |

System: 4GB+ RAM, internet connection, ~500MB disk for dependencies.

---

## Installation

**1. Clone and enter directory**

```bash
cd /path/to/workspace
mkdir contivex && cd contivex
```

**2. Create virtual environment**

```bash
# Mac/Linux
python3 -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure API keys**

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-proj-your-key-here
TAVILY_API_KEY=tvly-your-key-here        # optional
GOOGLE_API_KEY=AIza-your-key-here         # optional
```

| Service | Get Key | Free Tier |
|---|---|---|
| OpenAI | https://platform.openai.com/api-keys | $5 credit |
| Tavily | https://tavily.com | 1,000 searches/month |
| Google AI | https://aistudio.google.com/apikey | Rate-limited free |

**5. Launch**

```bash
# Recommended
python run.py

# Direct
streamlit run app_frontend.py
```

Opens at `http://localhost:8501`

---

## Usage

**Generate a blog** — Type your topic in the main composer area, click Generate. The system routes, researches (if needed), plans, writes in parallel, and optionally generates images.

**Review output** — Use tabs to inspect the plan, research sources, full preview, images, stats, and execution logs.

**Export** — Download Markdown, full ZIP (with images), or images separately from the Preview tab.

**Library** — Open the sidebar to browse, load, or delete past blogs.

### Generation Modes

| Mode | When | Research | Example |
|---|---|---|---|
| Closed-book | Evergreen concepts | None | "Python decorators tutorial" |
| Hybrid | Established + current data | Light | "Best Python frameworks 2026" |
| Open-book | Time-sensitive / news | Heavy (7-day window) | "AI developments this week" |

The system auto-detects the right mode. You can force web research in sidebar settings.

---

## Configuration

### Project Structure

```
contivex/
├── config.py           # Configuration management
├── app_backend.py      # LangGraph workflow engine
├── app_frontend.py     # Streamlit UI
├── run.py              # Launcher
├── requirements.txt    # Dependencies
├── .env                # API keys
├── .env.example        # Template
├── output/             # Generated blogs (auto-created)
│   ├── *.md
│   └── images/
└── logs/               # Execution logs
```

### Key Settings (config.py)

```python
# LLM
config.llm.model = "gpt-4.1-mini"   
config.llm.temperature = 0.4

# Blog
config.blog.min_tasks = 5
config.blog.max_tasks = 9
config.blog.min_words = 120
config.blog.max_words = 550

# Research
config.research.max_queries = 10
config.research.max_results_per_query = 6
config.research.open_book_days = 7

# Images
config.image.max_images = 3
config.image.model = "gemini-2.5-flash-image"
```

---

## Cost Estimates

### Per Blog

| Component | gpt-4.1-mini | gpt-4o |
|---|---|---|
| Router + Planning | ~$0.004 | ~$0.015 |
| Workers (5–9 sections) | ~$0.01–0.02 | ~$0.04–0.08 |
| Research (Tavily) | ~$0.005–0.01 | ~$0.005–0.01 |
| Images (Gemini, 0–3) | ~$0–0.09 | ~$0–0.09 |
| **Total** | **$0.03–0.15** | **$0.08–0.25** |

### Monthly

| Usage | Blogs/mo | Est. Cost |
|---|---|---|
| Light | 10 | ~$0.70 |
| Medium | 50 | ~$3.50 |
| Heavy | 200 | ~$14.00 |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| OpenAI key not found | Check `.env` exists, key format: `sk-proj-...`, has credits |
| Tavily search failed | Blog generates in closed-book mode. Add `TAVILY_API_KEY` for research |
| Image generation failed | Placeholder blocks shown. Add `GOOGLE_API_KEY` or disable images |
| Slow generation | Reduce sections in settings. 3–5 min is normal for complex research topics |
| Module not found | Activate venv, run `pip install -r requirements.txt`, check Python 3.10+ |
| Port in use | `streamlit run app_frontend.py --server.port 8502` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph, fan-out/fan-in, subgraphs) |
| LLM Framework | LangChain |
| Language Models | OpenAI gpt-4.1-mini / gpt-4o |
| Image Generation | Google Gemini 2.5 Flash |
| Web Research | Tavily API |r
| UI | Streamlit |
| Validation | Pydantic |

---

## Roadmap

**v1.1** — Multi-language support, custom templates, SEO suggestions, readability scoring

**v1.2** — Team collaboration, version control, direct publishing (Medium, Dev.to, WordPress)

**v2.0** — Fine-tuned industry models, video scripts, social media generation, multi-modal content

---

## License

Proprietary software developed and owned by **JillaniSofTech**. All rights reserved. Unauthorized copying, modification, distribution, or use is strictly prohibited without explicit written permission.

---

## Support

**Docs** — This README + in-app tooltips + execution logs

**Contact** — [JillaniSofTech on LinkedIn](https://linkedin.com/company/jillanisoftech)

**Services** — Custom AI solutions, LangGraph/LangChain consulting, RAG systems, enterprise integrations, cloud infrastructure (AWS/Azure/GCP)

---

<p align="center">
  <strong>Contivex</strong> — Transform Ideas into Professional Blogs<br/>
  <em>Built by <a href="https://jillanisoftech.com">JillaniSofTech</a></em><br/><br/>
  <a href="https://linkedin.com/company/jillanisoftech">
    <img src="https://img.shields.io/badge/LinkedIn-JillaniSofTech-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>

</p>
