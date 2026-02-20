from crewai import Agent
from tools import scan_repository, find_hardcoded_secrets
from config import MODEL_NAME
from langchain_openai import ChatOpenAI

from langchain_community.chat_models import ChatOllama

def get_llm():
    return ChatOllama(model="llama3")

# def get_llm():
#     return ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.2
#     )


def create_scanner_agent():
    return Agent(
        role="Code Scanner",
        goal="Scan repository and extract structured metadata",
        backstory="Expert at understanding repository structure.",
        verbose=True,
        allow_delegation=False
    )


def create_architecture_agent():
    return Agent(
        role="Architecture Analyst",
        goal="Determine architecture type and module structure",
        backstory="Senior architect analyzing code structure.",
        verbose=True
    )


def create_security_agent():
    return Agent(
        role="Security Auditor",
        goal="Identify vulnerabilities and security risks",
        backstory="Cybersecurity expert scanning code.",
        verbose=True
    )


def create_performance_agent():
    return Agent(
        role="Performance Engineer",
        goal="Detect bottlenecks and scalability issues",
        backstory="Performance optimization specialist.",
        verbose=True
    )


def create_roadmap_agent():
    return Agent(
        role="Roadmap Planner",
        goal="Create prioritized engineering roadmap",
        backstory="Engineering manager prioritizing improvements.",
        verbose=True
    )
