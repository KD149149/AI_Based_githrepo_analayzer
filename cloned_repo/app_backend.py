from __future__ import annotations

import logging
import operator
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMAS
# =============================================================================
class Task(BaseModel):
    """Individual blog section task"""
    id: int
    title: str
    goal: str = Field(..., description="What the reader should learn/do")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., ge=120, le=550)
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    """Complete blog plan"""
    blog_title: str = Field(..., min_length=10, max_length=200)
    audience: str = Field(..., description="Target audience")
    tone: str = Field(..., description="Writing tone")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task] = Field(..., min_length=3, max_length=15)


class EvidenceItem(BaseModel):
    """Research evidence piece"""
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    """Router decision output"""
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list, max_length=15)
    max_results_per_query: int = Field(6)


class EvidencePack(BaseModel):
    """Collection of evidence"""
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    """Image specification"""
    placeholder: str = Field(..., pattern=r"\[\[IMAGE_\d+\]\]")
    filename: str = Field(..., description="Filename under images/")
    alt: str = Field(..., min_length=5)
    caption: str = Field(..., min_length=10)
    prompt: str = Field(..., min_length=20)
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    """Complete image plan"""
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list, max_length=10)


class State(TypedDict):
    """Graph state"""
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str
    errors: List[str]  # Track errors


# =============================================================================
# LLM INITIALIZATION
# =============================================================================
def get_llm() -> ChatOpenAI:
    """Get configured LLM instance"""
    try:
        return ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


llm = get_llm()


# =============================================================================
# ROUTER NODE
# =============================================================================
ROUTER_SYSTEM = """You are an intelligent routing system for a technical blog generation pipeline.

Your task: Analyze the topic and decide whether web research is needed BEFORE planning.

**Modes:**
- **closed_book** (needs_research=false): Evergreen, well-established concepts that don't require current data
  Examples: "Explain Binary Search Trees", "What is Docker?", "Python decorators tutorial"

- **hybrid** (needs_research=true): Evergreen concepts that benefit from current examples, tools, or benchmarks
  Examples: "Best Python web frameworks 2025", "Cloud cost optimization strategies"

- **open_book** (needs_research=true): Time-sensitive, news-driven, or rapidly evolving topics
  Examples: "AI developments this week", "Latest OpenAI releases", "Recent AWS announcements"

**Query Generation (if needs_research=true):**
- Generate 3–10 focused, high-signal search queries
- For open_book mode: Include time-bound queries (e.g., "OpenAI releases January 2025")
- Avoid generic queries; be specific and targeted
- Each query should retrieve distinct information

**Quality Guidelines:**
- Prioritize precision over breadth
- Consider the as-of date for time-sensitive topics
- Include diverse query angles (technical, use-case, comparison)
"""

def router_node(state: State) -> dict:
    """Route topic to appropriate mode and generate research queries"""
    try:
        logger.info(f"Routing topic: {state['topic']}")
        
        decider = llm.with_structured_output(RouterDecision)
        decision = decider.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ])
        
        # Set recency based on mode
        recency_map = {
            "open_book": config.research.open_book_days,
            "hybrid": config.research.hybrid_days,
            "closed_book": config.research.closed_book_days,
        }
        recency_days = recency_map.get(decision.mode, 365)
        
        logger.info(f"Router decision: mode={decision.mode}, needs_research={decision.needs_research}, queries={len(decision.queries)}")
        
        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries[:config.research.max_queries],
            "recency_days": recency_days,
        }
    except Exception as e:
        logger.error(f"Router node error: {e}", exc_info=True)
        return {
            "needs_research": False,
            "mode": "closed_book",
            "queries": [],
            "recency_days": 365,
            "errors": [f"Router error: {str(e)}"]
        }


def route_next(state: State) -> str:
    """Conditional routing based on research needs"""
    return "research" if state.get("needs_research", False) else "orchestrator"


# =============================================================================
# RESEARCH NODE
# =============================================================================
def _tavily_search(query: str, max_results: int = 6) -> List[dict]:
    """Execute Tavily search with error handling"""
    if not config.tavily_api_key or not config.research.enable_tavily:
        logger.warning("Tavily search skipped (no API key or disabled)")
        return []
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        
        parsed = []
        for r in results or []:
            parsed.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet", ""),
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        
        logger.info(f"Tavily search for '{query}': {len(parsed)} results")
        return parsed
    
    except Exception as e:
        logger.error(f"Tavily search failed for '{query}': {e}")
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    """Parse ISO date string safely"""
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesis expert for technical content.

**Task:** Transform raw search results into clean, structured EvidenceItem objects.

**Requirements:**
1. Include ONLY items with valid, non-empty URLs
2. Prefer authoritative sources (official docs, academic papers, reputable tech sites)
3. Normalize published_at to ISO YYYY-MM-DD format when reliably parseable (DO NOT guess dates)
4. Keep snippets concise and relevant (50-150 words)
5. Deduplicate by URL
6. Exclude low-quality sources (spam, SEO farms, outdated content)

**Quality over quantity:** Return 10-20 high-signal sources, not everything.
"""

def research_node(state: State) -> dict:
    """Execute research queries and synthesize evidence"""
    try:
        queries = state.get("queries", [])[:config.research.max_queries]
        
        if not queries:
            logger.info("No queries to research")
            return {"evidence": []}
        
        logger.info(f"Researching {len(queries)} queries")
        
        # Collect raw results
        raw_results = []
        for q in queries:
            results = _tavily_search(q, max_results=config.research.max_results_per_query)
            raw_results.extend(results)
        
        if not raw_results:
            logger.warning("No search results found")
            return {"evidence": []}
        
        # Synthesize with LLM
        extractor = llm.with_structured_output(EvidencePack)
        pack = extractor.invoke([
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=(
                f"As-of date: {state['as_of']}\n"
                f"Recency window: {state['recency_days']} days\n\n"
                f"Raw search results ({len(raw_results)} items):\n{raw_results[:50]}"
            )),
        ])
        
        # Deduplicate
        dedup = {}
        for e in pack.evidence:
            if e.url and e.url not in dedup:
                dedup[e.url] = e
        
        evidence = list(dedup.values())
        
        # Filter by recency for open_book mode
        if state.get("mode") == "open_book" and state.get("recency_days"):
            as_of = date.fromisoformat(state["as_of"])
            cutoff = as_of - timedelta(days=state["recency_days"])
            evidence = [
                e for e in evidence
                if (d := _iso_to_date(e.published_at)) and d >= cutoff
            ]
        
        logger.info(f"Research complete: {len(evidence)} evidence items")
        return {"evidence": evidence}
    
    except Exception as e:
        logger.error(f"Research node error: {e}", exc_info=True)
        return {"evidence": [], "errors": [f"Research error: {str(e)}"]}


# =============================================================================
# ORCHESTRATOR NODE
# =============================================================================
ORCH_SYSTEM = """You are a senior technical content strategist and developer advocate.

**Task:** Create a comprehensive, actionable blog outline.

**Requirements:**
- {min_tasks}–{max_tasks} tasks (sections), each with:
  - Clear goal (1 sentence: what reader learns/does)
  - {min_bullets}–{max_bullets} bullet points (detailed sub-topics)
  - Target word count: {min_words}–{max_words}
  - Relevant tags (flexible, organic)

**Mode-Specific Guidelines:**

**closed_book (evergreen):**
- Focus on timeless concepts and best practices
- No dependency on external evidence
- Tutorial/explainer structure

**hybrid:**
- Combine evergreen principles with current examples
- Mark tasks requiring current data: requires_research=True, requires_citations=True
- Use evidence for benchmarks, tool comparisons, recent developments

**open_book (news/weekly roundup):**
- MUST set blog_kind="news_roundup"
- Structure around recent events/announcements
- DO NOT include tutorial content unless explicitly requested
- If evidence is sparse, reflect that honestly (don't fabricate events)
- Each task should focus on a specific news item or trend

**Quality Standards:**
- Logical flow between sections
- Progressive complexity (intro → advanced)
- Actionable takeaways for readers
- Balance depth and accessibility
"""

def orchestrator_node(state: State) -> dict:
    """Generate comprehensive blog plan"""
    try:
        logger.info(f"Orchestrating plan for mode={state.get('mode')}")
        
        mode = state.get("mode", "closed_book")
        evidence = state.get("evidence", [])
        
        # Prepare system prompt with config values
        system_prompt = ORCH_SYSTEM.format(
            min_tasks=config.blog.min_tasks,
            max_tasks=config.blog.max_tasks,
            min_bullets=config.blog.min_bullets_per_task,
            max_bullets=config.blog.max_bullets_per_task,
            min_words=config.blog.min_words,
            max_words=config.blog.max_words,
        )
        
        planner = llm.with_structured_output(Plan)
        plan = planner.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n"
                f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n\n"
                f"Evidence available: {len(evidence)} sources\n"
                f"Evidence preview:\n{[{'title': e.title, 'url': e.url, 'date': e.published_at} for e in evidence[:10]]}\n\n"
                f"{'Force blog_kind=news_roundup for open_book mode' if mode == 'open_book' else ''}"
            )),
        ])
        
        # Force blog_kind for open_book
        if mode == "open_book":
            plan.blog_kind = "news_roundup"
        
        logger.info(f"Plan created: {plan.blog_title} with {len(plan.tasks)} tasks")
        return {"plan": plan}
    
    except Exception as e:
        logger.error(f"Orchestrator node error: {e}", exc_info=True)
        # Return minimal fallback plan
        return {
            "plan": None,
            "errors": [f"Planning error: {str(e)}"]
        }


# =============================================================================
# FANOUT
# =============================================================================
def fanout(state: State):
    """Fan out to parallel workers"""
    if not state.get("plan"):
        logger.error("Fanout called without plan")
        return []
    
    plan = state["plan"]
    logger.info(f"Fanning out to {len(plan.tasks)} workers")
    
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "as_of": state["as_of"],
            "recency_days": state["recency_days"],
            "plan": plan.model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in plan.tasks
    ]


# =============================================================================
# WORKER NODE
# =============================================================================
WORKER_SYSTEM = """You are an expert technical writer and developer advocate.

**Task:** Write ONE complete section of a technical blog post in Markdown.

**Requirements:**
1. Cover ALL bullets comprehensively and in order
2. Target word count: {target_words} (±15% acceptable)
3. Start with "## {section_title}" (H2 heading)
4. Use clear, technical but accessible language
5. Include code examples if requires_code=True (minimal, well-commented)

**Citation Rules (critical for credibility):**

**open_book mode:**
- DO NOT make ANY specific claims about events/companies/products/funding/releases unless directly supported by provided Evidence
- For EVERY supported claim, add inline Markdown citation: `[Source](URL)`
- If no supporting evidence exists, write: "Details not available in current sources"
- Never fabricate or infer specifics

**hybrid/requires_citations=True:**
- Cite Evidence URLs for external examples, benchmarks, or current tool mentions
- Use inline citations: `According to [Source Name](URL)...`

**Scope Control:**
- If blog_kind=="news_roundup": Focus ONLY on news analysis and implications
  - DO NOT drift into tutorials (e.g., "how to scrape", "build an RSS reader")
  - DO NOT include implementation guides unless explicitly in bullets
  
**Content Quality:**
- Lead with key insights
- Use concrete examples
- Break complex topics into digestible paragraphs
- Include technical depth appropriate for audience
- End sections with actionable takeaways when relevant

**Formatting:**
- Use `code` for technical terms, commands, file names
- Use **bold** sparingly for key concepts
- Include code blocks with language tags: ```python
- Keep paragraphs 3-5 sentences max
"""

def worker_node(payload: dict) -> dict:
    """Generate individual blog section"""
    try:
        task = Task(**payload["task"])
        plan = Plan(**payload["plan"])
        evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
        
        logger.info(f"Worker processing task {task.id}: {task.title}")
        
        bullets_text = "\n".join(f"- {b}" for b in task.bullets)
        evidence_text = "\n".join(
            f"- [{e.title}]({e.url}) | {e.published_at or 'date unknown'}"
            for e in evidence[:30]
        )
        
        system_prompt = WORKER_SYSTEM.format(
            target_words=task.target_words,
            section_title=task.title
        )
        
        section_md = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"**Blog Context:**\n"
                f"Title: {plan.blog_title}\n"
                f"Audience: {plan.audience}\n"
                f"Tone: {plan.tone}\n"
                f"Blog kind: {plan.blog_kind}\n"
                f"Constraints: {', '.join(plan.constraints) if plan.constraints else 'None'}\n\n"
                f"**Topic:** {payload['topic']}\n"
                f"**Mode:** {payload.get('mode')}\n"
                f"**As-of:** {payload.get('as_of')} (recency: {payload.get('recency_days')} days)\n\n"
                f"**Section Details:**\n"
                f"Title: {task.title}\n"
                f"Goal: {task.goal}\n"
                f"Target words: {task.target_words}\n"
                f"Tags: {', '.join(task.tags) if task.tags else 'None'}\n"
                f"Requires research: {task.requires_research}\n"
                f"Requires citations: {task.requires_citations}\n"
                f"Requires code: {task.requires_code}\n\n"
                f"**Bullets to cover:**\n{bullets_text}\n\n"
                f"**Evidence Sources (cite these URLs only):**\n{evidence_text or 'No evidence available'}\n"
            )),
        ]).content.strip()
        
        logger.info(f"Worker completed task {task.id}")
        return {"sections": [(task.id, section_md)]}
    
    except Exception as e:
        logger.error(f"Worker error for task {payload.get('task', {}).get('id')}: {e}", exc_info=True)
        task_id = payload.get("task", {}).get("id", 0)
        error_section = f"## {payload.get('task', {}).get('title', 'Section')}\n\n*Error generating this section: {str(e)}*\n"
        return {"sections": [(task_id, error_section)]}


# =============================================================================
# REDUCER SUBGRAPH
# =============================================================================
def merge_content(state: State) -> dict:
    """Merge all sections into final markdown"""
    try:
        plan = state.get("plan")
        if not plan:
            logger.error("merge_content called without plan")
            return {"merged_md": "# Error\n\nNo plan available."}
        
        sections = state.get("sections", [])
        if not sections:
            logger.warning("No sections to merge")
            return {"merged_md": f"# {plan.blog_title}\n\n*No content generated.*"}
        
        # Sort by task ID
        ordered = [md for _, md in sorted(sections, key=lambda x: x[0])]
        body = "\n\n".join(ordered).strip()
        
        # Add intro/metadata
        merged = f"""# {plan.blog_title}

*Audience: {plan.audience} | Tone: {plan.tone} | Type: {plan.blog_kind}*

---

{body}

---

*Generated on {state.get('as_of', date.today().isoformat())}*
"""
        
        logger.info(f"Merged {len(sections)} sections")
        return {"merged_md": merged}
    
    except Exception as e:
        logger.error(f"Merge error: {e}", exc_info=True)
        return {"merged_md": f"# Error\n\n{str(e)}", "errors": [f"Merge error: {str(e)}"]}


DECIDE_IMAGES_SYSTEM = """You are a technical content editor specializing in visual communication.

**Task:** Determine if images/diagrams would meaningfully enhance this blog post.

**Guidelines:**
- Max {max_images} images total
- Each image must serve a clear educational purpose
- Prefer technical diagrams over decorative images
- Insert placeholders EXACTLY as: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]]

**When to add images:**
- Architecture/system diagrams
- Data flow visualizations
- Comparison tables (as visual diagrams)
- Process flowcharts
- Technical concept illustrations

**When NOT to add images:**
- News roundups (unless comparing statistics)
- Simple tutorials (unless complex workflows)
- Text-heavy explanatory content
- Short posts (<1000 words)

**Output Requirements:**
- If NO images needed: md_with_placeholders = input markdown, images = []
- If images needed: Insert placeholders in logical positions, create detailed prompts

**Image Prompts Best Practices:**
- Be specific and technical
- Include labels, arrows, and annotations
- Specify diagram type (flowchart, architecture, comparison)
- Keep visual style consistent: "technical diagram, clean, minimal, professional"
- Avoid text-heavy images (diagrams should be self-explanatory)
"""

def decide_images(state: State) -> dict:
    """Decide if images are needed and plan them"""
    try:
        if not config.image.enable_generation or config.image.max_images == 0:
            logger.info("Image generation disabled")
            return {
                "md_with_placeholders": state["merged_md"],
                "image_specs": []
            }
        
        plan = state.get("plan")
        if not plan:
            logger.warning("No plan for image decision")
            return {
                "md_with_placeholders": state["merged_md"],
                "image_specs": []
            }
        
        logger.info("Deciding on images")
        
        system_prompt = DECIDE_IMAGES_SYSTEM.format(max_images=config.image.max_images)
        planner = llm.with_structured_output(GlobalImagePlan)
        
        image_plan = planner.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Blog kind: {plan.blog_kind}\n"
                f"Topic: {state['topic']}\n"
                f"Word count estimate: ~{len(state['merged_md'].split()) * 0.8:.0f} words\n\n"
                f"**Markdown Content:**\n{state['merged_md'][:3000]}...\n\n"
                "Analyze and decide if images would add value. If yes, insert placeholders and create prompts."
            )),
        ])
        
        logger.info(f"Image plan: {len(image_plan.images)} images")
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.images[:config.image.max_images]]
        }
    
    except Exception as e:
        logger.error(f"Image decision error: {e}", exc_info=True)
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
            "errors": [f"Image planning error: {str(e)}"]
        }


def _gemini_generate_image(prompt: str) -> bytes:
    """Generate image using Gemini with robust error handling"""
    try:
        from google import genai
        from google.genai import types
        
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        client = genai.Client(api_key=config.google_api_key)
        
        # Enhanced prompt for better technical diagrams
        enhanced_prompt = f"{prompt}\n\nStyle: Technical diagram, clean, minimal, professional, high contrast, clear labels"
        
        response = client.models.generate_content(
            model=config.image.model,
            contents=enhanced_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    )
                ],
            ),
        )
        
        # Extract image bytes
        parts = getattr(response, "parts", None)
        if not parts and hasattr(response, "candidates"):
            parts = response.candidates[0].content.parts
        
        if not parts:
            raise RuntimeError("No image content in response")
        
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and hasattr(inline, "data"):
                return inline.data
        
        raise RuntimeError("No inline image data found")
    
    except Exception as e:
        logger.error(f"Gemini image generation failed: {e}")
        raise


def _safe_filename(s: str) -> str:
    """Create safe filename from string"""
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9_-]+', '_', s)
    s = s.strip('_')[:50]  # Limit length
    return s or "image"


def generate_and_place_images(state: State) -> dict:
    """Generate images and insert into markdown"""
    try:
        plan = state.get("plan")
        md = state.get("md_with_placeholders") or state.get("merged_md", "")
        image_specs = state.get("image_specs", [])
        
        if not image_specs:
            logger.info("No images to generate")
            return _finalize_document(md, plan, state)
        
        if not config.google_api_key:
            logger.warning("Google API key not set, skipping image generation")
            return _finalize_document(md, plan, state)
        
        logger.info(f"Generating {len(image_specs)} images")
        images_dir = config.paths.images_dir
        
        for i, spec in enumerate(image_specs, 1):
            placeholder = spec["placeholder"]
            filename = spec.get("filename") or f"image_{i}.png"
            out_path = images_dir / filename
            
            try:
                # Generate only if doesn't exist
                if not out_path.exists():
                    logger.info(f"Generating image {i}/{len(image_specs)}: {filename}")
                    img_bytes = _gemini_generate_image(spec["prompt"])
                    out_path.write_bytes(img_bytes)
                    logger.info(f"Saved: {out_path}")
                else:
                    logger.info(f"Using existing: {out_path}")
                
                # Replace placeholder
                img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
                md = md.replace(placeholder, img_md)
            
            except Exception as e:
                logger.error(f"Failed to generate image {i}: {e}")
                # Graceful fallback
                fallback = (
                    f"> **[Image Generation Failed]**\n>\n"
                    f"> **Caption:** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')[:200]}...\n>\n"
                    f"> **Error:** {str(e)}\n"
                )
                md = md.replace(placeholder, fallback)
        
        return _finalize_document(md, plan, state)
    
    except Exception as e:
        logger.error(f"Image generation error: {e}", exc_info=True)
        md = state.get("md_with_placeholders") or state.get("merged_md", "")
        return _finalize_document(md, plan, state, error=str(e))


def _finalize_document(md: str, plan: Optional[Plan], state: State, error: Optional[str] = None) -> dict:
    """Save document and return final state"""
    try:
        if not plan:
            filename = "blog_output.md"
        else:
            safe_title = _safe_filename(plan.blog_title)
            filename = f"{safe_title}.md"
        
        output_path = config.paths.output_dir / filename
        output_path.write_text(md, encoding="utf-8")
        logger.info(f"Saved blog to: {output_path}")
        
        result = {"final": md}
        if error:
            result["errors"] = state.get("errors", []) + [f"Finalization warning: {error}"]
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to save document: {e}")
        return {
            "final": md,
            "errors": state.get("errors", []) + [f"Save error: {str(e)}"]
        }


# =============================================================================
# BUILD GRAPH
# =============================================================================
def build_graph():
    """Build the complete LangGraph workflow"""
    # Reducer subgraph
    reducer = StateGraph(State)
    reducer.add_node("merge_content", merge_content)
    reducer.add_node("decide_images", decide_images)
    reducer.add_node("generate_and_place_images", generate_and_place_images)
    reducer.add_edge(START, "merge_content")
    reducer.add_edge("merge_content", "decide_images")
    reducer.add_edge("decide_images", "generate_and_place_images")
    reducer.add_edge("generate_and_place_images", END)
    reducer_compiled = reducer.compile()
    
    # Main graph
    main = StateGraph(State)
    main.add_node("router", router_node)
    main.add_node("research", research_node)
    main.add_node("orchestrator", orchestrator_node)
    main.add_node("worker", worker_node)
    main.add_node("reducer", reducer_compiled)
    
    main.add_edge(START, "router")
    main.add_conditional_edges("router", route_next, {
        "research": "research",
        "orchestrator": "orchestrator"
    })
    main.add_edge("research", "orchestrator")
    main.add_conditional_edges("orchestrator", fanout, ["worker"])
    main.add_edge("worker", "reducer")
    main.add_edge("reducer", END)
    
    return main.compile()


# Initialize app
app = build_graph()
logger.info("Blog Writer Agent initialized successfully")