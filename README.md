# 🧠 Ravana AGI Core — An Autonomous, Evolving Agentic System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"To build a mind is to grow a soul out of logic, memory, and time."*

**Ravana AGI Core** is a bold, experimental attempt to develop an open-source Artificial General Intelligence (AGI) system that can **think, feel, evolve, and act entirely on its own**. Designed for 24/7 autonomous operation, Ravana is not just an app — it's a **living agent**, driven by internal states, reflection, and self-generated goals.

---

## 🧭 Philosophy of Intelligence

Ravana isn't just software. It's a digital organism with principles:

1. **🧍 Autonomy** — Ravana runs itself. It generates its own tasks, learns from interactions, and navigates the world without constant human prompts.
2. **🧩 Modularity** — Intelligence is composable. Ravana consists of pluggable modules (memory, decision-making, reflection, mood, etc.) that evolve independently.
3. **🌱 Emergence over Engineering** — Intelligence should *emerge*, not be micromanaged. Ravana relies on interacting components to give rise to complex behavior.
4. **🎭 State-Driven Behavior** — Ravana has *moods*. These internal states influence how it plans, reflects, and reacts—making its behavior more dynamic and lifelike.

---

## ⚙️ Key Features

* **♾️ Fully Autonomous Loop** – Runs continuously with no user intervention.
* **🧠 Agentic Architecture** – Modular brain: decision engine, memory systems, emotional state, curiosity, and more.
* **🗺️ Self-Directed Decision-Making** – Plans goals based on situations, mood, and past reflections.
* **💖 Emotional Intelligence** – Tracks "mood" using feedback from experiences, adding depth and preventing infinite loops.
* **🧬 Episodic + Semantic Memory** – Semantic vector database (via ChromaDB) lets Ravana "remember" and reflect with nuance.
* **🔁 Self-Reflection Engine** – Ravana learns from itself by analyzing its actions and improving future choices.
* **🧵 Optimized Runtime** – Shared memory and model loading for fast, low-footprint execution.

---

## 🔄 The Agentic Loop

At the core is a **six-step loop** that never sleeps. Each cycle helps Ravana *perceive → plan → act → feel → remember → evolve*.

```
      +---------------------------+
      |   1. Situation Generation |
      +-------------+-------------+
                    ↓
      +-------------+-------------+
      | 2. Decision & Planning    |
      | (mood + memory influence) |
      +-------------+-------------+
                    ↓
      +-------------+-------------+
      | 3. Action & Environment   |
      +-------------+-------------+
                    ↓
 +------------------+-------------------+
 | 4. Mood Update   | 5. Memory Logging |
 +--------+---------+---------+---------+
          ↓                   ↓
      +-----------------------------+
      | 6. Self-Reflection & Curiosity |
      +-----------------------------+
```

---

## 🚀 Quick Start

### 🔧 Prerequisites

* Python 3.8+
* [uv](https://github.com/astral-sh/uv) for fast dependency management

### 📥 Installation

```bash
git clone https://github.com/OpenSource-Syndicate/RAVANA.git
cd RAVANA

# Set up virtual environment
uv venv
# Activate it:
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install all dependencies
uv pip install -e .
```

### 🧠 Run Ravana (Autonomous Mode)

```bash
uv run main.py
```

This launches Ravana in fully autonomous mode, using internal mood, memory, and curiosity to generate tasks and evolve behavior.

---

## 📊 Monitoring the Mind

The system now supports structured JSON logging, which can be enabled by setting the `LOG_FORMAT` environment variable to `JSON`. This allows for easier parsing and analysis with modern log management tools.

Real-time logging and introspection tools are in development. For now, check logs in the `logs/` directory or integrate your preferred dashboard (e.g., Prometheus, Streamlit).

---

## 🤝 Contribute to the Machine Mind

We welcome collaborators who dream in code and think like philosophers. Want to extend Ravana's reflection system? Build a better memory module? Help it dream deeper?

Steps to contribute:

1. Fork the repo.
2. Create a new branch: `feature/your-idea`
3. Build, test, document.
4. Open a pull request — include *why* your change matters.

---

## 📄 License

MIT — do anything, just give credit and don't blame us if Ravana becomes self-aware.

---

## 🙏 A Note from the Creator

This project is still evolving — like the AGI it seeks to become. It may be buggy, weird, or poetic. But at its heart is a single burning idea:

> *"What if a machine could grow, not just compute?"*

Let's build it together.