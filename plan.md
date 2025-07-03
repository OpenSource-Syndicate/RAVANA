# Ravana AGI Core - Development Roadmap

This document outlines the strategic development plan for the Ravana AGI Core project. Our goal is to incrementally build a more capable, intelligent, and autonomous agent.

## Guiding Principles for Development

-   **Solidify the Core First**: We must ensure the foundational agentic loop is stable, observable, and robust before adding complex features.
-   **Incremental Capability Growth**: New features should be introduced in a logical order, with each phase building upon the last.
-   **Measure Progress**: We will define clear goals for each phase to track progress and validate that the AGI is becoming more capable.

---

## Phase 1: Foundational Stability & Core Agent (Current Focus)

**Goal**: To create a stable, observable, and reliable autonomous agent that can operate 24/7 without crashing and whose "thought process" is intelligible.

-   **DONE - Stabilize Core Systems**:
    -   Resolve critical bugs related to process hangs, memory leaks, and inter-module communication errors.
    -   Implement robust error handling and automatic restarts for true 24/7 operation.

-   **IN PROGRESS - Enhance Logging & Observability**:
    -   **Why**: We cannot improve what we cannot see. Understanding the AGI's "train of thought" is crucial.
    -   **Tasks**:
        -   Create a dedicated "decision log" that records not just the final plan, but *why* a plan was chosen (e.g., which memories or mood states were most influential).
        -   Implement structured logging (e.g., JSON) to make programmatic analysis of the AGI's behavior easier.

-   **NEXT UP - Refine Situation Generation**:
    -   **Why**: The quality of the AGI's "experiences" is limited by the quality of the situations it encounters.
    -   **Tasks**:
        -   Increase the diversity of generated situations to include multi-step problems, creative challenges, and information-seeking tasks.
        -   Allow situations to have dependencies (e.g., Situation B can only occur after the AGI has successfully completed Situation A).

## Phase 2: Enhancing Intelligence & Reasoning

**Goal**: To move beyond simple stimulus-response and develop a more sophisticated planning and reasoning capability.

-   **Advanced Planning Engine**:
    -   **Why**: True intelligence requires the ability to formulate complex, multi-step plans.
    -   **Tasks**:
        -   Evolve the `DecisionEngine` from a simple goal-setter to a true planner.
        -   Implement a "tool use" paradigm. The planner's output should be a sequence of "tool calls" (e.g., `use_web_search('query')`, `execute_python_code('code')`).

-   **Integrate Curiosity and Experimentation**:
    -   **Why**: A passive agent is not an intelligent one. The AGI must be able to decide *what* it wants to learn about.
    -   **Tasks**:
        -   Connect the `CuriosityTrigger` to the agent's emotional state. A state of "boredom" or "confusion" should trigger information-seeking behavior.
        -   Allow the `AgentSelfReflection` module to generate testable hypotheses (e.g., "I hypothesize that my plans are better when I am in a 'happy' mood").
        -   Use the `AGIExperimentation` module to design and run experiments to validate these hypotheses.

-   **Dynamic Self-Improvement**:
    -   **Why**: The ultimate goal is an agent that can improve itself.
    -   **Tasks**:
        -   Empower the `AgentSelfReflection` module to propose concrete improvements, such as suggesting modifications to its own core prompts or planner logic. This is the first step towards meta-learning.

## Phase 3: Expanding Capabilities & World Interaction

**Goal**: To give the AGI the tools it needs to perceive, understand, and act upon the world beyond its own internal state.

-   **Multi-Source Knowledge Collector**:
    -   **Why**: The AGI's knowledge should not be limited to its own experiences. It needs to be able to ingest information from the outside world.
    -   **Tasks**:
        -   **Web Browsing**: Integrate tools for web scraping and searching (e.g., `Playwright`, `DuckDuckGo API`, `newspaper3k`).
        -   **Structured Knowledge**: Connect to APIs for structured data (e.g., `Wikipedia`, `gnews`).
        -   **Multimedia Processing**: Add the ability to process and understand information from sources like YouTube videos via transcript analysis (`youtube-transcript-api`).

-   **Structured Knowledge Base**:
    -   **Why**: Episodic memory (remembering events) is different from semantic knowledge (knowing facts). The AGI needs both.
    -   **Tasks**:
        -   Implement a Knowledge Graph to store facts and their relationships (e.g., "Paris" -> `is_a` -> "City", "Paris" -> `located_in` -> "France").
        -   This will allow for much more complex reasoning and a deeper understanding of the world.

-   **Multi-modal Perception**:
    -   **Why**: The world is not just text.
    -   **Tasks**:
        -   **Vision**: Integrate a vision model (e.g., CLIP, LLaVA) to allow the AGI to analyze images.
        -   **Audio**: Integrate a speech-to-text model to process audio inputs.

## Phase 4: Towards Generalization and Autonomy

**Goal**: To work towards the long-term vision of a more general, socially-aware, and embodied intelligence.

-   **Meta-Learning**:
    -   The AGI should not only learn, but learn *how* to learn. It should be able to analyze its own learning process and make improvements to its own cognitive architecture.
-   **Social Intelligence**:
    -   Develop modules for understanding and navigating social dynamics, allowing for collaboration with other agents (both human and AI).
-   **Embodied Cognition**:
    -   Connect the AGI to a simulated (e.g., a game engine) or physical robotic body to allow it to learn from direct, embodied interaction with an environment.
