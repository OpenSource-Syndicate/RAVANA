from core.actions.action import Action
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime circular imports
    from core.system import AGISystem
    from services.data_service import DataService

class ProposeAndTestInventionAction(Action):
    def __init__(self, system: 'AGISystem', data_service: 'DataService'):
        super().__init__(system, data_service)
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
        and running it through the advanced experimentation engine.
        """
        invention = kwargs.get("invention_description")
        test_plan = kwargs.get("test_plan_suggestion")

        # Frame the invention as a formal hypothesis
        hypothesis = (
            f"It is hypothesized that the following concept is viable: '{invention}'. "
            f"It can be tested by attempting the following: '{test_plan}'."
        )

        print(f"Action triggered: Proposing new invention for testing. Hypothesis: {hypothesis}")

        # Import the experimentation engine
        from core.llm import agi_experimentation_engine
        import asyncio
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Run the experiment using the advanced experimentation engine
            logger.info(f"Starting advanced experimentation for: {invention[:100]}...")
            
            # Modify the hypothesis to ensure plots are saved properly
            experiment_prompt = f"{hypothesis}\n\nIMPORTANT: If generating Python code with plots, use plt.savefig() to save plots as PNG files instead of plt.show() to avoid blocking execution."
            
            # Run in a thread to avoid blocking the async loop
            result = await asyncio.to_thread(
                agi_experimentation_engine,
                experiment_idea=experiment_prompt,
                llm_model=None,
                use_chain_of_thought=True,
                online_validation=True,
                sandbox_timeout=20,
                verbose=True
            )
            
            # Log the experiment to the database
            await asyncio.to_thread(
                self.data_service.save_experiment_log,
                invention,
                test_plan,
                result.get('final_verdict', 'Experiment completed'),
                result.get('execution_result', 'No execution result')
            )
            
            # Prepare detailed response
            response = {
                "action": "experiment_completed",
                "hypothesis": hypothesis,
                "experiment_successful": result.get('final_verdict', '').lower().find('success') != -1,
                "simulation_type": result.get('simulation_type', 'unknown'),
                "code_generated": bool(result.get('generated_code')),
                "execution_successful": bool(result.get('execution_result')) and not bool(result.get('execution_error')),
                "scientific_validity": "high" if "success" in result.get('final_verdict', '').lower() else "medium",
                "summary": f"Experiment '{invention}' completed. " + 
                          f"Type: {result.get('simulation_type', 'unknown')}. " +
                          f"Result: {result.get('final_verdict', 'No verdict')[:100]}...",
                "detailed_results": result
            }
            
            logger.info(f"Experiment completed successfully: {invention}")
            return response
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            
            # Log the failed experiment
            await asyncio.to_thread(
                self.data_service.save_experiment_log,
                invention,
                test_plan,
                f"FAILED: {str(e)}",
                "Experiment execution failed"
            )
            
            return {
                "action": "experiment_failed",
                "hypothesis": hypothesis,
                "error": str(e),
                "reason": "The experimentation engine encountered an error during execution."
            } 