# ⚡ RAVANA AGI — Autonomous, Evolving Agentic System

*"What if intelligence wasn’t just a tool you queried… but a living system that runs, remembers, reflects, and evolves?"*

RAVANA is not just another AI framework.
It’s an experiment in **autonomous general intelligence** — a system designed to:

* **Run continuously**, without constant human prompting
* **Build and recall memory**, shaping future decisions
* **Model emotions** as internal signals guiding behavior
* **Reflect, adapt, and improve itself** over time

Think of RAVANA as an **agentic runtime for intelligence itself**: an open playground where researchers, hackers, and dreamers can test what it means for an AI to *exist*, not just *respond*.

---

## 🌌 Why RAVANA?

Most AI projects today build **tools**.
RAVANA builds **agents**.

The goal is not to chase benchmarks, but to **prototype autonomy**:

* What happens when an agent can run for weeks, not seconds?
* How does memory change decision-making?
* Can “emotional intelligence” be modeled computationally?
* What does it mean for an AI to self-reflect?

This repository is a lab for those questions. Every file, every module, is a stepping stone towards a **continuous, evolving intelligence system**.

---

## 📂 Repository Guide

Here’s how the project is organized:

* **`main.py`** → Example runtime entrypoint; start the system here
* **`core/`** → Orchestrator, state manager, internal services, and config
* **`modules/`** → Plug-and-play intelligence modules (reflection, experimentation, conversational AI, etc.)
* **`services/`** → Service-level abstractions: memory, knowledge bases, environment interaction
* **`database/`** → Schemas, models, and the database engine for long-term persistence
* **`scripts/`** → Standalone scripts for special operations (physics experiments, conversational AI, cleanup, etc.)
* **`docs/`** → Curated documentation: usage, API references, developer notes (`docs/index.md` is the hub)
* **`assets/`** → Static assets like images and other media
* **`wiki/`** → Deep design documents, architecture breakdowns, and module-level explorations

Think of `docs/` as the **“how to use it”** layer and `wiki/` as the **“how it works under the hood”** layer.

---

## 📖 Knowledge Base

The **wiki** is where the architecture really comes alive. Some must-reads:

* [📜 Project Overview](wiki/Project%20Overview.md) — why this exists
* [🏛 Architecture & Design](wiki/Architecture%20&%20Design.md) — the blueprint of autonomy
* [🧠 Core System](wiki/Core%20System.md) — orchestrator, states, agents
* [🗂 Memory Systems](wiki/Memory%20Systems.md) — short-term, episodic, semantic memory
* [🤝 LLM Integration](wiki/LLM%20Integration.md) — using language models as cognitive modules
* [⚙️ Development Guide](wiki/Development%20Guide.md) — how to contribute & extend
* [🔌 Services](wiki/Services.md) — data pipelines, APIs, and external hooks

Additional gems in the folder: Action System, Emotional Intelligence, Decision-Making modules — each one a piece of the AGI puzzle.

---

## 🚀 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/OpenSource-Syndicate/RAVANA.git
cd RAVANA
```

2. **Create a virtual environment & install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# for dev work:
pip install -e .
```

3. **Run the system**

```bash
python main.py
```

> 🛠 Notes:
>
> * Some modules may need config (`core/config.json` or `core/config.py`).
> * For long-running sessions or deployments, see `docs/development/` + `wiki/Deployment & Operations.md`.

---

## 🧪 Experiments & Tools

The repo includes test harnesses to stretch RAVANA’s abilities:

* `scripts/run_physics_tests.py` — physics interaction benchmarking
* `scripts/physics_cli.py` — interactive physics experiment interface
* `scripts/physics_experiment_prompts.py` — collection of physics experiment prompts
* `scripts/cleanup_session.py` — clean up session data and temporary files
* `scripts/launch_conversational_ai.py` — standalone conversational AI interface
* `test_snake_agent_performance.py` — comprehensive test suite for Snake Agent
* `snake_performance_monitor.py` — real-time performance monitoring
* `run_optimized_snake_agent.py` — run Snake Agent with optimized settings

Run them from an active virtual environment to watch the system’s behavior unfold.

## ⚡ Snake Agent Optimization

The Snake Agent has been enhanced with performance optimizations using threading and multiprocessing:

### Performance Features
- **Threading Architecture**: Multiple specialized threads for different tasks (file monitoring, analysis, communication)
- **Multiprocessing**: CPU-intensive tasks run in separate processes for isolation and performance
- **VLTM Integration**: Very Long-Term Memory system for knowledge retention and pattern recognition
- **Performance Monitoring**: Real-time metrics collection and analysis
- **Auto-Recovery**: Automatic recovery from failures and errors

### Running with Optimal Performance
To run the Snake Agent with optimized settings:

```bash
# Set environment variables for optimal performance (optional, defaults are optimized):
export SNAKE_MAX_THREADS=12
export SNAKE_MAX_PROCESSES=6
export SNAKE_ANALYSIS_THREADS=4
export SNAKE_PERF_MONITORING=True
export SNAKE_AUTO_RECOVERY=True

# Run the optimized snake agent
python run_optimized_snake_agent.py --demo
```

### Performance Monitoring
Monitor the Snake Agent's performance in real-time:

```bash
# Monitor performance for 5 minutes
python snake_performance_monitor.py --short-test

# Analyze historical performance data
python snake_performance_monitor.py --analyze-file snake_performance.json
```

### Testing Performance
Run comprehensive performance tests:

```bash
# Run the comprehensive test suite
python test_snake_agent_performance.py

# Run specific tests
python -m pytest tests/test_enhanced_snake_agent.py -v
```

For more detailed information about Snake Agent optimization, see the [performance guide](docs/snake_agent_performance_guide.md).

---

## 🤝 Contributing

RAVANA is a **collaborative experiment** — contributions are not just welcome, they’re essential.

* Read the [Development Guide](wiki/Development%20Guide.md) and `docs/development/`
* Branch from `main` for new features
* Write tests when possible
* Document your changes (clarity > cleverness)
* Open a pull request with a clear explanation

If you want to **co-create the future of autonomy**, this is your sandbox.

---

## 📜 License

RAVANA is distributed under the **MIT License**.
Use it, remix it, extend it — just don’t keep the future locked away.

---

## 💡 Where to Begin

* Start with [Project Overview](wiki/Project%20Overview.md)
* Explore the [Architecture & Design](wiki/Architecture%20&%20Design.md) page
* Run `main.py` and watch the system breathe

For help, open an issue with the `help wanted` tag.

---

## 🌱 Closing Thought

RAVANA is not finished — it’s not supposed to be.
It’s **alive in iteration**, evolving through every commit, issue, and idea.

*"To build an intelligence is to write code that learns to rewrite itself."*

Welcome to the experiment.
