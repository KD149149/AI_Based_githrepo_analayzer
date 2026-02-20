from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """LLM Configuration"""
    model: str = Field(default="gpt-4.1-mini", description="OpenAI model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for generation")


class ResearchConfig(BaseModel):
    """Research/Search Configuration"""
    max_results_per_query: int = Field(default=6, ge=1, le=20)
    max_queries: int = Field(default=10, ge=1, le=20)
    enable_tavily: bool = Field(default=True)
    
    # Recency windows
    open_book_days: int = Field(default=7, description="Days for open_book/news roundup")
    hybrid_days: int = Field(default=45, description="Days for hybrid mode")
    closed_book_days: int = Field(default=3650, description="Days for closed_book")


class ImageConfig(BaseModel):
    """Image Generation Configuration"""
    max_images: int = Field(default=3, ge=0, le=10)
    default_size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    default_quality: Literal["low", "medium", "high"] = "medium"
    model: str = Field(default="gemini-2.5-flash-image")
    enable_generation: bool = Field(default=True)


class BlogConfig(BaseModel):
    """Blog Generation Configuration"""
    min_tasks: int = Field(default=5, ge=3, le=15)
    max_tasks: int = Field(default=9, ge=5, le=20)
    min_bullets_per_task: int = Field(default=3, ge=2, le=10)
    max_bullets_per_task: int = Field(default=6, ge=3, le=15)
    min_words: int = Field(default=120, ge=50)
    max_words: int = Field(default=550, ge=200)


class PathConfig(BaseModel):
    """File Path Configuration"""
    output_dir: Path = Field(default=Path("output"))
    images_dir: Path = Field(default=Path("output/images"))
    logs_dir: Path = Field(default=Path("logs"))
    
    def ensure_dirs(self):
        """Create directories if they don't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    """Main Application Configuration"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    blog: BlogConfig = Field(default_factory=BlogConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    # API Keys validation
    openai_api_key: Optional[str] = Field(default=None)
    tavily_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Load from environment if not provided
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.tavily_api_key:
            self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
    
    def validate_keys(self) -> dict[str, bool]:
        """Validate API keys are present"""
        return {
            "openai": bool(self.openai_api_key),
            "tavily": bool(self.tavily_api_key),
            "google": bool(self.google_api_key),
        }
    
    def get_missing_keys(self) -> list[str]:
        """Get list of missing API keys"""
        validation = self.validate_keys()
        return [k for k, v in validation.items() if not v]


# Global configuration instance
config = AppConfig()
config.paths.ensure_dirs()