"""
Data Copilot - Insurance Data Query Agent

A domain-specific AI agent that allows users to query millions of rows
of insurance loss data using natural language.

Project Structure:
==================

data_copilot/
├── training/           # 1. TRAINING MODULE
│   ├── domain/         # Insurance formulas & terminology
│   ├── data/           # Training data generators
│   └── scripts/        # Training scripts
│
├── agents/             # 2. INFERENCE MODULE (LangGraph)
│   ├── state.py        # AgentState definition
│   ├── graph.py        # State machine
│   └── nodes/          # Graph nodes
│
├── evaluation/         # 3. EVALUATION MODULE
│   ├── test_types.py   # Test case definitions
│   ├── metrics.py      # Scoring metrics
│   └── evaluator.py    # Main evaluator
│
├── models/             # Model interfaces
├── tools/              # SQL execution
├── api/                # FastAPI backend
├── ui/                 # Streamlit frontend
└── observability/      # MLFlow tracing
"""

__version__ = "0.1.0"
