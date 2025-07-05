from core.actions.action import Action
from typing import Any, Dict, List

class ProposeAndTestInventionAction(Action):
    """
    An action that allows the AGI to propose a novel idea, concept, or 'invention'
    and formally submit it to the experimentation and learning loop for testing.
    """
    @property
    def name(self) -> str:
        return "propose_and_test_invention"

    @property
    def description(self) -> str:
        return (
            "Propose a novel idea, invention, or concept and design an experiment "
            "to test its validity or properties. This is for truly novel ideas that "
            "are not based on analyzing existing performance, but on creative thought."
        )

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "invention_description",
                "type": "string",
                "description": "A clear and concise description of the novel idea, concept, or hypothesis to be tested.",
                "required": True,
            },
            {
                "name": "test_plan_suggestion",
                "type": "string",
                "description": "A brief suggestion on how this invention could be tested.",
                "required": True,
            }
        ]

    async def execute(self, **kwargs: Any) -> Any:
        """
        Executes the action by formatting the invention into a hypothesis
        and returning a directive to the main loop to initiate the experiment.
        """
        invention = kwargs.get("invention_description")
        test_plan = kwargs.get("test_plan_suggestion")

        # Frame the invention as a formal hypothesis
        hypothesis = (
            f"It is hypothesized that the following concept is viable: '{invention}'. "
            f"It can be tested by attempting the following: '{test_plan}'."
        )

        print(f"Action triggered: Proposing new invention for testing. Hypothesis: {hypothesis}")

        # Return a special directive for the AGISystem main loop to handle.
        # This will trigger the experimentation workflow.
        return {
            "action": "initiate_experiment",
            "hypothesis": hypothesis,
            "reason": "A novel invention or concept was proposed for experimental validation via the dedicated action."
        } 