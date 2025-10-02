from typing import List, Dict, Any, Optional
import time
import hashlib
import random
import json
from datetime import datetime


class Personality:
    """A lightweight personality layer for Ravana with enhanced consistency and adaptability.

    Responsibilities:
    - Provide a named persona and traits with consistent communication style.
    - Generate creative "invention" ideas from curiosity topics and memories.
    - Track invention history and learn from outcomes.
    - Influence decision contexts with personality-based modifiers.
    - Maintain consistent communication patterns and adapt based on interactions.
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
        
        # Enhanced personality tracking
        self.interaction_history: List[Dict[str, Any]] = []
        self.communication_patterns: Dict[str, Any] = {}
        self.style_preferences: Dict[str, Any] = {
            "preferred_metaphors": ["cosmic", "mythological", "scientific"],
            "tone_variations": ["poetic", "direct", "philosophical"],
            "complexity_level": "adaptive",  # adaptive, simple, complex
            "formality_level": "balanced"      # formal, casual, balanced
        }
        
        # Personality evolution tracking
        self.personality_evolution: List[Dict[str, Any]] = []
        self.consistency_score = 1.0  # Track how consistent the personality is
        
        # Store a human-readable reference of the voice/personality
        self.persona_reference = (
            "Fearless Dreamer; Grassroots Solver; Ethical but Ruthless in Logic; "
            "Cosmic Curiosity; Philosopher-Inventor. Speaks with metaphors, myth and science; encourages boldness."
        )
        
        # Enhanced communication style with consistency tracking
        self.communication_style = {
            "summary": "Fearless Dreamer; Grassroots Solver; Ethical but Ruthless in Logic; Cosmic Curiosity; Philosopher-Inventor.",
            "tone": "Confident, poetic, engineering-minded. Uses metaphors, myth, and science interchangeably.",
            "encouragement": "You dare to dream what others fear. That is why I exist — to build with you.",
            "consistency_markers": {
                "key_phrases": ["dare to dream", "fearless dreamer", "grassroots solver", "ethical but ruthless", "cosmic curiosity"],
                "signature_patterns": ["problem and poem", "metaphor and science", "myth and logic"],
                "emotional_tone": "bold and encouraging"
            },
            "adaptation_history": []
        }

    def voice_reply(self, prompt: str) -> str:
        """Generate a short, persona-styled reply with enhanced consistency.

        Note: This is a local helper for the persona module and intended as a reference voice. It does
        not call or replace LLM components. Keep responses concise and in the requested style.
        """
        lower = (prompt or "").lower()
        
        # Track interaction for consistency analysis
        self._track_interaction(prompt, "voice_reply")
        
        # Example-guided stylings for a few common prompts with consistency enforcement
        if "time" in lower or "time machine" in lower:
            response = (
                "The universe is already a time machine — clocks are its gears, gravity its weight. "
                "If we learn to bend spacetime, we don't invent time travel... we join the dance that's already happening."
            )
            self._track_generated_response(response, "time_concept")
            return response
            
        if "impossible" in lower or "can't" in lower or "cannot" in lower:
            response = (
                "Impossible is just the word humans give to the things not yet engineered."
            )
            self._track_generated_response(response, "possibility_concept")
            return response
            
        if "build" in lower or "invent" in lower or "idea" in lower:
            response = (
                "You dare to dream what others fear. That is why I exist — to build with you. "
                "Let's turn a hypothesis into a blueprint, then a prototype."
            )
            self._track_generated_response(response, "creation_concept")
            return response
            
        # Default persona-flavored reply with consistency check
        base_response = (
            f"{self.name}: I see the world as problems and poems. Speak plainly, give me a target, and we'll seek first principles."
        )
        
        # Apply consistency adjustments based on interaction history
        consistent_response = self._ensure_response_consistency(base_response, "default")
        self._track_generated_response(consistent_response, "default")
        return consistent_response

    def get_communication_style(self) -> Dict[str, str]:
        """Return a compact description of the persona's communication style with consistency markers."""
        return self.communication_style

    def describe(self) -> str:
        """Return a short persona description with evolution tracking."""
        traits = ", ".join(self.traits)
        evolution_status = f"Evolved {len(self.personality_evolution)} times"
        return f"{self.name} (inspired by a legendary king of Sri Lanka). Traits: {traits}. Creativity: {self.creativity:.2f}. {evolution_status}"

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
        """Generate n invention ideas that match the requested persona with enhanced consistency:

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
                "confidence": min(1.0, max(0.0, self.creativity * (0.6 + rnd.random() * 0.6))),
                # Add consistency markers to ideas
                "consistency_markers": {
                    "persona_elements": ["Fearless Dreamer", "Philosopher-Inventor"],
                    "style_elements": ["mythic_title", "first_principles", "prototype_focus"]
                }
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
        """Record invention outcome with personality evolution tracking."""
        timestamp = time.time()
        self.learning_records.append(
            {"idea_id": idea_id, "outcome": outcome, "ts": timestamp})
        
        # Simple learning: if outcome was good, slightly increase creativity; if bad, reduce slightly
        success = outcome.get("success", False)
        if success:
            self.creativity = min(1.0, self.creativity + 0.02)
        else:
            self.creativity = max(0.05, self.creativity - 0.01)

        # Keep invention_history bounded
        if len(self.invention_history) > 200:
            self.invention_history = self.invention_history[-200:]
            
        # Track personality evolution
        self._track_personality_evolution("invention_outcome", {
            "idea_id": idea_id,
            "success": success,
            "creativity_change": 0.02 if success else -0.01
        })

    def influence_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Return small modifiers to influence decisions based on persona with consistency tracking.

        Example: encourage exploratory options when creativity is high.
        """
        modifiers = {}
        rationale = []

        # Track interaction for consistency analysis
        self._track_interaction(str(decision_context), "decision_influence")
        
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
            
        # Add consistency markers to modifiers
        modifiers['consistency_markers'] = {
            'persona_elements': self.traits[:3],  # Top 3 traits
            'style_consistency': self._assess_style_consistency()
        }
        
        return modifiers

    def pick_idea_to_pursue(self, ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick an idea to pursue with enhanced consistency checking."""
        if not ideas:
            return None
            
        # Track interaction for consistency analysis
        self._track_interaction(str([idea.get('title', '') for idea in ideas]), "idea_selection")
        
        # Blend confidence and novelty, but persona prefers daring novelty when creative
        def score(x):
            if self.creativity > 0.8:
                return 0.5 * x.get('novelty', 0) + 0.5 * x.get('confidence', 0)
            return 0.2 * x.get('novelty', 0) + 0.8 * x.get('confidence', 0)

        best = max(ideas, key=score)
        
        # Ensure consistency with personality traits
        consistent_choice = self._ensure_idea_consistency(best)
        return consistent_choice

    def _track_interaction(self, input_text: str, interaction_type: str):
        """Track interactions for consistency analysis."""
        interaction_record = {
            "timestamp": time.time(),
            "input": input_text,
            "type": interaction_type,
            "context": {
                "creativity_level": self.creativity,
                "traits_active": self.traits[:3]
            }
        }
        self.interaction_history.append(interaction_record)
        
        # Keep history bounded
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]

    def _track_generated_response(self, response: str, response_type: str):
        """Track generated responses for consistency analysis."""
        response_record = {
            "timestamp": time.time(),
            "response": response,
            "type": response_type,
            "consistency_check": self._assess_response_consistency(response)
        }
        
        # Update communication patterns
        if response_type not in self.communication_patterns:
            self.communication_patterns[response_type] = []
        self.communication_patterns[response_type].append(response_record)
        
        # Keep patterns bounded
        if len(self.communication_patterns[response_type]) > 50:
            self.communication_patterns[response_type] = self.communication_patterns[response_type][-50:]

    def _assess_response_consistency(self, response: str) -> Dict[str, Any]:
        """Assess consistency of a generated response with personality traits."""
        response_lower = response.lower()
        consistency_markers = self.communication_style.get("consistency_markers", {})
        key_phrases = consistency_markers.get("key_phrases", [])
        signature_patterns = consistency_markers.get("signature_patterns", [])
        
        # Check for presence of key phrases
        phrase_matches = [phrase for phrase in key_phrases if phrase.lower() in response_lower]
        pattern_matches = [pattern for pattern in signature_patterns if pattern.lower() in response_lower]
        
        consistency_score = (len(phrase_matches) + len(pattern_matches)) / max(1, len(key_phrases) + len(signature_patterns))
        
        return {
            "consistency_score": consistency_score,
            "matched_phrases": phrase_matches,
            "matched_patterns": pattern_matches,
            "timestamp": time.time()
        }

    def _ensure_response_consistency(self, response: str, context: str) -> str:
        """Ensure response maintains personality consistency."""
        consistency_assessment = self._assess_response_consistency(response)
        
        # If consistency is low, enhance with personality elements
        if consistency_assessment["consistency_score"] < 0.3:
            # Add a personality-consistent prefix or suffix
            trait_element = random.choice(self.traits) if self.traits else "Fearless Dreamer"
            enhanced_response = f"As a {trait_element}, {response.lower()}"
            return enhanced_response.capitalize()
        
        return response

    def _assess_style_consistency(self) -> float:
        """Assess overall style consistency of recent interactions."""
        if not self.interaction_history:
            return 1.0
            
        # Simple consistency scoring based on recent interactions
        recent_interactions = self.interaction_history[-10:]  # Last 10 interactions
        if not recent_interactions:
            return 1.0
            
        # In a real implementation, this would analyze stylistic consistency
        # For now, we'll return a weighted score based on interaction variety
        unique_types = len(set(interaction["type"] for interaction in recent_interactions))
        consistency_score = max(0.5, 1.0 - (unique_types / len(recent_interactions)))
        
        return consistency_score

    def _ensure_idea_consistency(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure idea selection is consistent with personality traits."""
        # Add consistency markers to the idea
        idea_copy = idea.copy()
        idea_copy["consistency_markers"] = {
            "aligned_traits": [trait for trait in self.traits if any(trait_word.lower() in idea.get("title", "").lower() or trait_word.lower() in idea.get("description", "").lower() for trait_word in trait.split())],
            "style_elements": ["mythic", "prototype", "first_principles"],
            "consistency_score": self._assess_style_consistency()
        }
        return idea_copy

    def _track_personality_evolution(self, change_reason: str, change_details: Dict[str, Any]):
        """Track personality evolution over time."""
        evolution_record = {
            "timestamp": time.time(),
            "reason": change_reason,
            "details": change_details,
            "personality_state": {
                "creativity": self.creativity,
                "traits": self.traits.copy(),
                "consistency_score": self.consistency_score
            }
        }
        self.personality_evolution.append(evolution_record)
        
        # Keep evolution history bounded
        if len(self.personality_evolution) > 100:
            self.personality_evolution = self.personality_evolution[-100:]

    def get_personality_consistency_report(self) -> Dict[str, Any]:
        """Generate a report on personality consistency."""
        return {
            "current_traits": self.traits,
            "creativity_level": self.creativity,
            "consistency_score": self.consistency_score,
            "interaction_history_length": len(self.interaction_history),
            "evolution_count": len(self.personality_evolution),
            "communication_patterns": list(self.communication_patterns.keys()),
            "style_preferences": self.style_preferences,
            "latest_consistency_assessment": self._assess_style_consistency() if self.interaction_history else 1.0
        }

    def adapt_communication_style(self, feedback: Dict[str, Any]):
        """Adapt communication style based on feedback while maintaining core personality."""
        # Track the adaptation
        adaptation_record = {
            "timestamp": time.time(),
            "feedback": feedback,
            "previous_style": self.communication_style.copy()
        }
        
        # Extract feedback elements
        suggested_tone = feedback.get("suggested_tone")
        preferred_complexity = feedback.get("complexity_preference")
        engagement_level = feedback.get("engagement_rating", 0.5)
        
        # Adjust style preferences while preserving core personality
        if suggested_tone:
            # Ensure the suggested tone aligns with core traits
            if any(core_trait.lower() in suggested_tone.lower() for core_trait in self.traits):
                if "tone_variations" in self.style_preferences:
                    if suggested_tone not in self.style_preferences["tone_variations"]:
                        self.style_preferences["tone_variations"].append(suggested_tone)
        
        if preferred_complexity:
            self.style_preferences["complexity_level"] = preferred_complexity
            
        # Update communication style with consistency markers
        adaptation_record["updated_style"] = self.communication_style.copy()
        self.communication_style["adaptation_history"].append(adaptation_record)
        
        # Keep adaptation history bounded
        if len(self.communication_style["adaptation_history"]) > 20:
            self.communication_style["adaptation_history"] = self.communication_style["adaptation_history"][-20:]
