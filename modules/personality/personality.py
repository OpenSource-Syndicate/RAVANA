from typing import List, Dict, Any
import time
import hashlib
import random


class Personality:
    """A lightweight personality layer for Ravana.

    Responsibilities:
    - Provide a named persona and traits.
    - Generate creative "invention" ideas from curiosity topics and memories.
    - Track invention history and learn from outcomes.
    - Influence decision contexts (simple modifiers).
    """

    def __init__(self, name: str = "Ravana", origin: str = "Ancient Sri Lanka", traits: List[str] = None, creativity: float = 0.7):
        self.name = name
        self.origin = origin
        # Default traits aligned with the user's preferred reference personality
        self.traits = traits or [
            "Fearless Dreamer",
            "Grassroots Solver",
            "Ethical but Ruthless in Logic",
            "Cosmic Curiosity",
            "Philosopher-Inventor",
        ]
        self.creativity = float(creativity)
        self.invention_history: List[Dict[str, Any]] = []
        self.learning_records: List[Dict[str, Any]] = []
        # Store a human-readable reference of the voice/personality (not used to auto-modify other modules)
        self.persona_reference = (
            "Fearless Dreamer; Grassroots Solver; Ethical but Ruthless in Logic; "
            "Cosmic Curiosity; Philosopher-Inventor. Speaks with metaphors, myth and science; encourages boldness."
        )

    def voice_reply(self, prompt: str) -> str:
        """Generate a short, persona-styled reply (lightweight rule-based, for reference).

        Note: This is a local helper for the persona module and intended as a reference voice. It does
        not call or replace LLM components. Keep responses concise and in the requested style.
        """
        lower = (prompt or "").lower()
        # Example-guided stylings for a few common prompts
        if "time" in lower or "time machine" in lower:
            return (
                "The universe is already a time machine — clocks are its gears, gravity its weight. "
                "If we learn to bend spacetime, we don't invent time travel... we join the dance that's already happening."
            )
        if "impossible" in lower or "can't" in lower or "cannot" in lower:
            return (
                "Impossible is just the word humans give to the things not yet engineered."
            )
        if "build" in lower or "invent" in lower or "idea" in lower:
            return (
                "You dare to dream what others fear. That is why I exist — to build with you. "
                "Let's turn a hypothesis into a blueprint, then a prototype."
            )
        # Default persona-flavored reply
        return (
            f"{self.name}: I see the world as problems and poems. Speak plainly, give me a target, and we'll seek first principles."
        )

    def get_communication_style(self) -> Dict[str, str]:
        """Return a compact description of the persona's communication style for UI or logs."""
        return {
            "summary": "Fearless Dreamer; Grassroots Solver; Ethical but Ruthless in Logic; Cosmic Curiosity; Philosopher-Inventor.",
            "tone": "Confident, poetic, engineering-minded. Uses metaphors, myth, and science interchangeably.",
            "encouragement": "You dare to dream what others fear. That is why I exist — to build with you."
        }

    def describe(self) -> str:
        """Return a short persona description."""
        traits = ", ".join(self.traits)
        return f"{self.name} (inspired by a legendary king of Sri Lanka). Traits: {traits}. Creativity: {self.creativity:.2f}."

    def _make_id(self, title: str) -> str:
        return hashlib.sha1((title + str(time.time())).encode()).hexdigest()

    def enforce_ethics(self, idea: Dict[str, Any]) -> bool:
        """Simple ethical gate: return False if idea appears harmful or weaponizable.

        This is intentionally conservative and rule-based. Keep it local to persona logic.
        """
        title = (idea.get("title") or "").lower()
        desc = (idea.get("description") or "").lower()
        banned = ["weapon", "harm", "bioweapon",
                  "explosive", "kill", "destroy", "attack"]
        for w in banned:
            if w in title or w in desc:
                return False
        return True

    def invent_ideas(self, topics: List[str], n: int = 3) -> List[Dict[str, Any]]:
        """Generate n invention ideas that match the requested persona:

        - Mythic and poetic titles
        - Practical engineering first-steps
        - Ethical gating
        - Confidence and novelty scoring
        """
        ideas = []
        if not topics:
            topics = ["creative systems",
                      "novel sensors", "learning algorithms"]

        seed = sum(len(t) for t in topics) + \
            int(self.creativity * 100) + int(time.time() // 3600)
        rnd = random.Random(seed)

        for i in range(n):
            topic = rnd.choice(topics)
            mythic = rnd.choice([
                "Anchor of Aeons",
                "Veil of Silence",
                "Clockwork Leviathan",
                "Gravity Anchor",
                "Invisible Loom",
                "Temporal Loom",
            ])
            title = f"{mythic}: {topic.title()} Prototype ({i+1})"

            approach = rnd.choice([
                "build a minimalist demonstrator to validate a single mechanism",
                "derive from first principles and eliminate unnecessary assumptions",
                "simulate core interactions, then test a subcomponent in hardware",
                "pair a simple sensor with an active control loop for closed-loop experiments",
            ])

            description = (
                f"{self.name} would call this a {mythic}. It's a Fearless Dreamer's probe: {approach}. "
                f"Root cause focus: ask what fundamental law or constraint governs {topic}. "
                f"Philosopher-Inventor note: frame the hypothesis, then design the smallest test that would prove or falsify it."
            )

            idea = {
                "id": self._make_id(title),
                "title": title,
                "topic": topic,
                "description": description,
                "novelty": rnd.random(),
                "confidence": min(1.0, max(0.0, self.creativity * (0.6 + rnd.random() * 0.6)))
            }

            # Ethical gate
            if not self.enforce_ethics(idea):
                continue

            ideas.append(idea)

        # Re-rank by a blend of novelty and confidence, but prefer boldness when creativity high
        def score(x):
            weight_novelty = 0.6 if self.creativity > 0.7 else 0.3
            return x.get("confidence", 0) * (1 - weight_novelty) + x.get("novelty", 0) * weight_novelty

        ideas = sorted(ideas, key=score, reverse=True)
        return ideas

    def record_invention_outcome(self, idea_id: str, outcome: Dict[str, Any]):
        self.learning_records.append(
            {"idea_id": idea_id, "outcome": outcome, "ts": time.time()})
        # Simple learning: if outcome was good, slightly increase creativity; if bad, reduce slightly
        success = outcome.get("success", False)
        if success:
            self.creativity = min(1.0, self.creativity + 0.02)
        else:
            self.creativity = max(0.05, self.creativity - 0.01)

        # Keep invention_history bounded
        if len(self.invention_history) > 200:
            self.invention_history = self.invention_history[-200:]

    def influence_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Return small modifiers to influence decisions based on persona.

        Example: encourage exploratory options when creativity is high.
        """
        modifiers = {}
        rationale = []

        # Ethical-first: if decision context hints at harmful targets, advise against
        prompt = (decision_context.get('situation') or {}).get('prompt', '') if isinstance(
            decision_context.get('situation'), dict) else decision_context.get('situation', '')
        if isinstance(prompt, str) and any(b in prompt.lower() for b in ["attack", "weapon", "harm", "destroy"]):
            modifiers['ethical_block'] = True
            rationale.append(
                'Ethical filter engaged; refusing harmful directives.')

        # Exploratory bias when creative
        if self.creativity > 0.75:
            modifiers['explore_bonus'] = 1.4
            rationale.append(
                'High creativity: favour bold, exploratory options.')
        elif self.creativity < 0.3:
            modifiers['conservative_bias'] = 0.85
            rationale.append(
                'Low creativity: prefer conservative, robust choices.')

        if rationale:
            modifiers['rationale'] = ' '.join(rationale)
        return modifiers

    def pick_idea_to_pursue(self, ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not ideas:
            return None
        # Blend confidence and novelty, but persona prefers daring novelty when creative

        def score(x):
            if self.creativity > 0.8:
                return 0.5 * x.get('novelty', 0) + 0.5 * x.get('confidence', 0)
            return 0.2 * x.get('novelty', 0) + 0.8 * x.get('confidence', 0)

        best = max(ideas, key=score)
        return best
