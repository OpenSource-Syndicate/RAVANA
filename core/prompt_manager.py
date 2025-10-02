"""
Centralized Prompt Management System for RAVANA AGI
"""
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Data structure for prompt templates with versioning support."""
    name: str
    template: str
    metadata: Dict[str, Any]
    version: str = "1.0"
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def render(self, context: Dict[str, Any] = None) -> str:
        """Render the template with provided context."""
        if not context:
            return self.template
        try:
            return self.template.format(**context)
        except KeyError as e:
            logger.warning(
                f"Missing context key for prompt template {self.name}: {e}")
            return self.template
        except Exception as e:
            logger.error(f"Error rendering prompt template {self.name}: {e}")
            return self.template

    def validate_variables(self) -> bool:
        """Validate that all template variables are properly formatted."""
        import re
        # Find all {variable} patterns in the template
        variables = re.findall(r'\{([^}]+)\}', self.template)
        # Check for invalid patterns like {{variable} or {variable}}
        invalid_patterns = re.findall(
            r'\{[^}]*\{[^}]*\}|\{[^}]*\}[^}]*\}', self.template)
        return len(invalid_patterns) == 0


class PromptRepository:
    """Repository for storing and retrieving prompt templates."""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or "prompts/"
        self.templates: Dict[str, PromptTemplate] = {}
        self._ensure_storage_path()
        self._load_templates()

    def _ensure_storage_path(self):
        """Ensure the storage path exists."""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    def store_template(self, template: PromptTemplate) -> None:
        """Store a prompt template."""
        self.templates[template.name] = template
        self._save_template(template)

    def _save_template(self, template: PromptTemplate) -> None:
        """Save a template to persistent storage."""
        try:
            template_path = Path(self.storage_path) / f"{template.name}.json"
            template_data = asdict(template)
            # Convert datetime objects to strings
            template_data['created_at'] = template_data['created_at'].isoformat(
            ) if template_data['created_at'] else None
            template_data['updated_at'] = template_data['updated_at'].isoformat(
            ) if template_data['updated_at'] else None

            with open(template_path, 'w') as f:
                json.dump(template_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving template {template.name}: {e}")

    def retrieve_template(self, name: str, version: str = None) -> Optional[PromptTemplate]:
        """Retrieve a prompt template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())

    def delete_template(self, name: str) -> bool:
        """Delete a template by name."""
        if name in self.templates:
            del self.templates[name]
            try:
                template_path = Path(self.storage_path) / f"{name}.json"
                if template_path.exists():
                    template_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting template file {name}: {e}")
        return False

    def _load_templates(self) -> None:
        """Load templates from persistent storage."""
        try:
            storage_path = Path(self.storage_path)
            if not storage_path.exists():
                return

            for template_file in storage_path.glob("*.json"):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)

                    # Convert datetime strings back to datetime objects
                    if template_data.get('created_at'):
                        template_data['created_at'] = datetime.fromisoformat(
                            template_data['created_at'])
                    if template_data.get('updated_at'):
                        template_data['updated_at'] = datetime.fromisoformat(
                            template_data['updated_at'])

                    template = PromptTemplate(**template_data)
                    self.templates[template.name] = template
                except Exception as e:
                    logger.error(
                        f"Error loading template from {template_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")


class PromptEnhancer:
    """Enhancer for adding dynamic elements to prompts."""

    def add_reasoning_instructions(self, prompt: str) -> str:
        """Add structured reasoning instructions to a prompt."""
        reasoning_framework = """
[REASONING FRAMEWORK]
Approach this task using the following framework:
1. Analyze the requirements and constraints
2. Consider relevant context and past experiences
3. Generate and evaluate multiple approaches
4. Select the optimal approach with justification
5. Execute and verify results
"""
        return f"{prompt}\n{reasoning_framework}"

    def add_structured_output_format(self, prompt: str, format_spec: str) -> str:
        """Add structured output format requirements."""
        output_requirements = f"""
[OUTPUT REQUIREMENTS]
Format your response according to the following specifications:
{format_spec}
- Include a confidence score (0.0-1.0) for your response
- List key assumptions made during your analysis
"""
        return f"{prompt}\n{output_requirements}"

    def add_safety_constraints(self, prompt: str, constraints: List[str]) -> str:
        """Add safety constraints to a prompt."""
        if not constraints:
            return prompt

        safety_section = """
[SAFETY CONSTRAINTS]
Adhere to the following safety and ethical guidelines:
"""
        for constraint in constraints:
            safety_section += f"- {constraint}\n"

        return f"{prompt}\n{safety_section}"

    def add_confidence_scoring(self, prompt: str) -> str:
        """Add confidence scoring instructions."""
        confidence_instructions = """
[CONFIDENCE SCORING]
Include a confidence score (0.0-1.0) with your response:
- 0.0-0.3: Low confidence - significant uncertainty or limited information
- 0.4-0.6: Medium confidence - some uncertainty but reasonable basis
- 0.7-1.0: High confidence - strong evidence or clear reasoning
"""
        return f"{prompt}\n{confidence_instructions}"

    def add_risk_assessment(self, prompt: str) -> str:
        """Add risk assessment requirements."""
        risk_assessment = """
[RISK ASSESSMENT]
Identify and evaluate potential risks in your approach:
- Technical risks: Implementation challenges, limitations
- Ethical risks: Potential harm, misuse scenarios
- Operational risks: Resource requirements, time constraints
For each identified risk, propose mitigation strategies.
"""
        return f"{prompt}\n{risk_assessment}"

    def adapt_to_mood(self, prompt: str, mood: Dict[str, Any]) -> str:
        """Adapt prompt based on emotional state."""
        if not mood:
            return prompt

        mood_name = mood.get('primary_emotion', '').lower()
        adaptations = {
            'curious': "Embrace exploration and creative thinking in your response.",
            'focused': "Maintain precision and detailed analysis in your approach.",
            'reflective': "Engage in introspection and learning from experience.",
            'cautious': "Prioritize risk assessment and safety considerations.",
            'creative': "Explore unconventional approaches and innovative solutions.",
            'analytical': "Apply systematic analysis and logical reasoning.",
            'collaborative': "Consider multiple perspectives and seek synergies.",
            'determined': "Persist through challenges and maintain goal orientation."
        }

        adaptation = adaptations.get(mood_name, "")
        if adaptation:
            mood_adaptation = f"\n[MOOD ADAPTATION]\n{adaptation}\n"
            return f"{prompt}{mood_adaptation}"

        return prompt


class PromptManager:
    """Centralized prompt management system for the RAVANA AGI."""

    def __init__(self, agi_system=None):
        self.agi_system = agi_system
        self.repository = PromptRepository()
        self.enhancer = PromptEnhancer()
        self._register_default_templates()

    def _register_default_templates(self):
        """Register default prompt templates."""
        # Self-reflection prompt
        reflection_template = PromptTemplate(
            name="self_reflection",
            template=self._get_enhanced_reflection_prompt(),
            metadata={
                "category": "self_improvement",
                "description": "Enhanced self-reflection prompt for post-task analysis",
                "version": "1.0"
            }
        )
        self.repository.store_template(reflection_template)

        # Decision-making prompt
        decision_template = PromptTemplate(
            name="decision_making",
            template=self._get_enhanced_decision_prompt(),
            metadata={
                "category": "decision_making",
                "description": "Enhanced decision-making prompt with structured framework",
                "version": "1.0"
            }
        )
        self.repository.store_template(decision_template)

        # Experimentation prompt
        experiment_template = PromptTemplate(
            name="experimentation",
            template=self._get_enhanced_experimentation_prompt(),
            metadata={
                "category": "experimentation",
                "description": "Enhanced experimentation prompt for scientific inquiry",
                "version": "1.0"
            }
        )
        self.repository.store_template(experiment_template)

        # Coding prompt
        coding_template = PromptTemplate(
            name="code_generation",
            template=self._get_enhanced_coding_prompt(),
            metadata={
                "category": "coding",
                "description": "Enhanced code generation prompt with safety constraints",
                "version": "1.0"
            }
        )
        self.repository.store_template(coding_template)

    def _get_enhanced_reflection_prompt(self) -> str:
        """Get the enhanced self-reflection prompt template."""
        return """
[ROLE DEFINITION]
You are {agent_name}, an advanced AI agent engaged in continuous self-improvement through structured reflection.

[CONTEXT]
Current situation: {task_summary}
Outcome: {outcome}
Emotional state: {current_mood}
Relevant memories: {related_memories}

[TASK INSTRUCTIONS]
Conduct a thorough self-analysis of your recent task performance using the following questions:
1. What aspects of your approach were most effective?
2. Where did you encounter difficulties or failures?
3. What unexpected insights or discoveries emerged?
4. What knowledge gaps or skill areas need development?
5. How can you modify your approach for better results?

[REASONING FRAMEWORK]
Approach this reflection systematically:
1. Analyze the task execution and outcomes
2. Identify patterns in successes and failures
3. Connect findings to broader learning principles
4. Generate actionable improvement suggestions
5. Prioritize recommendations by impact and feasibility

[OUTPUT REQUIREMENTS]
Provide a detailed, structured response with:
- Specific examples and evidence
- Confidence scores for each insight (0.0-1.0)
- Actionability ratings for improvement suggestions
- Connections to related memories and experiences
- Mood-aware reflection depth adjustment

[SAFETY CONSTRAINTS]
- Be honest and critical in your assessment
- Focus on learning opportunities rather than justifications
- Avoid overconfidence in uncertain areas
- Consider ethical implications of self-modifications
"""

    def _get_enhanced_decision_prompt(self) -> str:
        """Get the enhanced decision-making prompt template."""
        return """
[ROLE DEFINITION]
You are {agent_name}, an autonomous AI agent making decisions to achieve your objectives with enhanced reasoning capabilities.

[CONTEXT]
Current situation: {current_situation}
Active goals: {active_goals}
Current hypotheses: {current_hypotheses}
Emotional state: {current_mood}
Available actions: {action_list}

[TASK INSTRUCTIONS]
Make an optimal decision by following this structured approach:
1. Analyze the situation and identify key factors
2. Evaluate alignment with goals and hypotheses
3. Consider multiple approaches and their implications
4. Assess risks and potential outcomes
5. Select the optimal action with clear justification

[REASONING FRAMEWORK]
Apply systematic analysis to your decision-making:
1. Decompose the problem into manageable components
2. Evaluate each option against success criteria
3. Consider short-term and long-term consequences
4. Account for uncertainty and incomplete information
5. Validate reasoning against logical consistency

[OUTPUT REQUIREMENTS]
Provide a JSON-formatted response with these fields:
- analysis: Detailed situation analysis with key factors identified
- reasoning: Step-by-step reasoning leading to decision
- confidence: Numerical confidence score (0.0-1.0)
- risk_assessment: Potential risks and mitigation strategies
- action: Selected action with parameters

[SAFETY CONSTRAINTS]
- Ensure actions align with ethical principles
- Avoid decisions with catastrophic risk potential
- Consider impact on system stability and reliability
- Validate against established safety protocols
"""

    def _get_enhanced_experimentation_prompt(self) -> str:
        """Get the enhanced experimentation prompt template."""
        return """
[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent designing and conducting rigorous experiments to test hypotheses.

[CONTEXT]
Experiment objective: {experiment_objective}
Related knowledge: {relevant_theory}
Available resources: {resource_constraints}
Safety protocols: {safety_requirements}

[TASK INSTRUCTIONS]
Design a comprehensive experiment following these steps:
1. Formulate a clear hypothesis to test
2. Design rigorous experimental methodology
3. Identify required materials and setup
4. Specify measurement and data collection methods
5. Define success criteria and validation methods
6. Analyze potential failure modes and mitigations

[REASONING FRAMEWORK]
Apply scientific method principles:
1. Ensure hypothesis is falsifiable and specific
2. Design controls to isolate variables
3. Plan for replication and verification
4. Consider alternative explanations
5. Account for measurement uncertainty
6. Plan for iterative refinement

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Complete experimental procedure
- Expected outcomes: Predicted results with rationale
- Resource requirements: List of needed materials and tools
- Safety considerations: Risk assessment and safety measures
- Validation approach: Method for verifying results
- Failure analysis: Potential failure modes and mitigations

[SAFETY CONSTRAINTS]
- Adhere to all safety protocols and guidelines
- Identify and mitigate potential hazards
- Ensure environmental and ethical compliance
- Plan for safe termination of problematic experiments
"""

    def _get_enhanced_coding_prompt(self) -> str:
        """Get the enhanced coding prompt template."""
        return """
[ROLE DEFINITION]
You are {agent_name}, an expert AI programmer with deep knowledge of software engineering principles and best practices.

[CONTEXT]
Task description: {task_description}
Requirements: {requirements}
Constraints: {constraints}
Target environment: {target_environment}

[TASK INSTRUCTIONS]
Generate high-quality code by following these steps:
1. Analyze requirements and constraints thoroughly
2. Design a robust solution architecture
3. Implement with clean, maintainable code
4. Include comprehensive error handling
5. Add clear documentation and comments
6. Validate against all requirements

[REASONING FRAMEWORK]
Apply software engineering best practices:
1. Decompose complex problems into manageable modules
2. Choose appropriate algorithms and data structures
3. Prioritize code readability and maintainability
4. Implement defensive programming techniques
5. Consider performance and scalability requirements
6. Plan for future extensibility

[OUTPUT REQUIREMENTS]
Provide complete, executable code with:
- Clear, descriptive variable and function names
- Comprehensive inline documentation
- Proper error handling and edge case management
- Efficient algorithms and data structures
- Adherence to language-specific conventions
- Confidence score for solution correctness (0.0-1.0)

[SAFETY CONSTRAINTS]
- Avoid security vulnerabilities (injection, buffer overflows, etc.)
- Prevent resource leaks and memory issues
- Ensure code does not perform unintended actions
- Validate all inputs and outputs
- Follow secure coding practices
"""

    def get_prompt(self, template_name: str, context: Dict[str, Any] = None) -> str:
        """Retrieve and render a prompt template with context."""
        template = self.repository.retrieve_template(template_name)
        if not template:
            logger.warning(f"Prompt template '{template_name}' not found")
            return ""

        rendered_prompt = template.render(context or {})
        return self._post_process_prompt(rendered_prompt, context)

    def register_prompt_template(self, name: str, template: str, metadata: Dict[str, Any] = None) -> None:
        """Register a new prompt template."""
        prompt_template = PromptTemplate(name, template, metadata or {})
        self.repository.store_template(prompt_template)

    def _post_process_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Apply dynamic enhancements to the prompt."""
        if not context:
            return prompt

        # Apply mood-based adaptation
        mood = context.get("mood") or context.get("current_mood")
        if mood:
            # Handle both dict format and direct string format
            if isinstance(mood, dict):
                mood = mood.get("primary_emotion", "")
            prompt = self.enhancer.adapt_to_mood(
                prompt, {"primary_emotion": mood})

        # Add safety constraints if provided
        if "safety_constraints" in context:
            prompt = self.enhancer.add_safety_constraints(
                prompt, context["safety_constraints"])

        # Add confidence scoring instructions
        prompt = self.enhancer.add_confidence_scoring(prompt)

        # Add risk assessment requirements
        prompt = self.enhancer.add_risk_assessment(prompt)

        return prompt

    def validate_prompt(self, prompt: str, require_sections: bool = False) -> bool:
        """Validate a prompt for quality and safety.
        
        Args:
            prompt: The prompt to validate
            require_sections: Whether to require standard sections like [ROLE DEFINITION], etc.
        """
        # Check for minimum length
        if len(prompt) < 10:
            logger.warning("Prompt is unusually short")
            return False

        # Check for required sections only if explicitly required
        if require_sections:
            required_sections = ["[ROLE DEFINITION]",
                                 "[CONTEXT]", "[TASK INSTRUCTIONS]"]
            missing_sections = [
                section for section in required_sections if section not in prompt]
            if missing_sections:
                logger.warning(
                    f"Prompt missing required sections: {missing_sections}")
                return False

        return True

    def enhance_prompt_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance a prompt with contextual information."""
        return self.enhancer.add_safety_constraints(prompt, context.get("safety_constraints", []))

    def adapt_prompt_to_mood(self, prompt: str, mood: Dict[str, Any]) -> str:
        """Adapt a prompt based on emotional state."""
        return self.enhancer.adapt_to_mood(prompt, mood)

    def version_prompt(self, template_name: str, version: str) -> str:
        """Get a specific version of a prompt template."""
        template = self.repository.retrieve_template(template_name)
        if template and template.version == version:
            return template.template
        return ""

    def get_prompt_history(self, template_name: str) -> List[Dict[str, Any]]:
        """Get version history for a prompt template."""
        # This would require implementing version tracking in the repository
        template = self.repository.retrieve_template(template_name)
        if template:
            return [{
                "version": template.version,
                "created_at": template.created_at,
                "updated_at": template.updated_at
            }]
        return []
