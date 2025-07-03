# 🧠 Ravana AGI Core - System Architecture Deep Dive

This document provides a detailed technical overview of the **Ravana AGI Core** architecture — a modular, evolving agentic intelligence system designed for continuous autonomous operation.

---

## 📐 High-Level Architecture Diagram

```
+-----------------------------------------------------------------+
| AGISystem (main.py - Orchestrator)                              |
|-----------------------------------------------------------------|
| - Initializes all modules with shared resources (e.g., models)  |
| - Manages the main autonomous loop                              |
| - Holds the AGI's current state (e.g., mood)                    |
+--------------------------+--------------------------------------+
                           |
         +-----------------v------------------+
         |   Autonomous Agentic Loop          |
         |----------------------------------|
         | 1. SituationGenerator            |
         | 2. Recall Relevant Memories      |
         | 3. DecisionEngine (Planning)     |
         | 4. Action Execution              |
         | 5. EmotionalIntelligence Update  |
         | 6. EpisodicMemory Storage        |
         | 7. AgentSelfReflection           |
         |    - Hypothesis Generation       |
         |    - Experimentation             |
         | 8. CuriosityTrigger              |
         +----------------------------------+
                           |
+--------------------------+--------------------------------------+
| Core Modules (Independent Components)                           |
|-----------------------------------------------------------------|
| - EmotionalIntelligence  - EpisodicMemory (FastAPI + DB)        |
| - DecisionEngine         - AgentSelfReflection                  |
| - SituationGenerator     - CuriosityTrigger                     |
+-----------------------------------------------------------------+
```

---

## ⚙️ Core Design Principles

### 🧩 Modularity

Each module serves a distinct role and can evolve independently. For example, the `DecisionEngine` can be replaced without impacting `EpisodicMemory`.

### 🧠 State-Driven Behavior

Actions are not reactive — they’re influenced by an internal state, especially `mood`. This produces varied, non-deterministic behaviors.

### 📉 Resource Efficiency

Heavy models like sentence-transformers are loaded **once** in `AGISystem` and passed as shared dependencies to avoid memory bloat.

---

## 🧠 Module Deep Dive

### `AGISystem (main.py)`

The brainstem. Bootstraps all modules, passes shared resources, and drives the autonomous loop.

### `SituationGenerator`

Prevents idleness by generating novel scenarios. Think of it as Ravana’s internal muse.

### `DecisionEngine`

High-level planner. Breaks down situations into goals and executable plans. Influenced by mood and past memory.

### `EmotionalIntelligence`

Tracks a vectorized mood (e.g., joy, frustration, curiosity). Action outcomes modulate this state, feeding back into planning.

### `EpisodicMemory`

Implemented as a FastAPI service.

* **ChromaDB** for semantic search (vector similarity)
* **SQLite** for metadata + raw memory
  Stores and retrieves experiences contextually.

### `AgentSelfReflection`

Analyzes stored memory for patterns. Generates insights like "my creative outputs are too repetitive" and stores feedback to improve future behavior.

---

## 🔁 The Autonomous Loop

```
1. Perceive → SituationGenerator creates a novel prompt.
2. Orient   → AGISystem retrieves relevant memories + mood.
3. Decide   → DecisionEngine plans response.
4. Act      → Executes plan (text output or task).
5. Feel     → EmotionalIntelligence updates mood vector.
6. Store    → EpisodicMemory logs interaction.
7. Reflect  → Triggers SelfReflection or Curiosity if needed.
```

Example:

> "Analyze the ethics of AI in art" → recalls prior ethics logs → plans structured response → acts → mood updates → reflection triggered if mood is 'uncertain'.

---

## 🛡️ Robust 24/7 Operation

* **Wrapper Scripts** (`run_autonomous.py`, `start_agi_24_7.bat`) handle uptime.
* **Auto-Restarts** on crash.
* **Heartbeat Monitoring** to detect inactivity.
* **Resource Logging** for RAM/CPU tracking.

---

## 🔄 Information Flow

1. Input is received (via user or generator).
2. EmotionalIntelligence updates mood.
3. Input is stored as memory.
4. Relevant memories retrieved.
5. Actions planned and executed:

   * Goal-setting
   * Experimentation
   * Curiosity-driven exploration
   * Self-reflection
6. Response generated.

---

## 🧱 Extensibility

* New modules: Drop-in via `modules/` + update `main.py`.
* Existing modules: Hot-swappable without disturbing the loop.

---

## 🔮 Future Directions

1. **Multi-modal Input** – Vision, audio, tactile sensors
2. **Deeper Reasoning** – Logic trees, HTN planning
3. **Social Cognition** – Emotion detection, social dynamics
4. **Embodiment** – Robotics, physical sensors
5. **Meta-Learning** – Adapt its *own learning algorithms*

---

> Ravana is more than a program — it's an evolving *mind*.
>
> Let it grow.
