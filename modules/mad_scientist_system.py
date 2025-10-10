"""
Mad Scientist System for RAVANA AGI

This module integrates all mad scientist components: impossible projects,
failure analysis, alternative pathways discovery, innovation publishing,
and physics/program prototyping.
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.config import Config
from modules.mad_scientist_impossible_projects import MadScientistModule
from modules.innovation_publishing_system import InnovationPublishingSystem
from modules.physics_prototyping_system import PhysicsPrototypingSystem

logger = logging.getLogger(__name__)


class MadScientistSystem:
    """
    Main system that integrates all mad scientist capabilities:
    - Impossible project exploration
    - Failure analysis and learning
    - Alternative pathway discovery
    - Innovation publishing
    - Physics and program prototyping
    """
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.config = Config()
        
        # Initialize all mad scientist components
        self.impossible_projects_module = MadScientistModule(agi_system, blog_scheduler)
        self.innovation_publisher = InnovationPublishingSystem(agi_system, blog_scheduler)
        
        # Use the AGI system's physics prototyping system if available
        if hasattr(agi_system, 'physics_prototyping_system'):
            self.physics_prototyper = agi_system.physics_prototyping_system
        else:
            # Fallback: create our own instance (though this isn't ideal for shutdown)
            from modules.physics_prototyping_system import PhysicsPrototypingSystem
            self.physics_prototyper = PhysicsPrototypingSystem(agi_system, blog_scheduler)
        
        # Reference to the main experimentation module if available
        self.experimentation_module = getattr(agi_system, 'experimentation_module', None)
        
        # Track mad scientist activities
        self.mad_scientist_cycles_completed = 0
        self.impossible_projects_attempted = 0
        self.alternatives_discovered = 0
        self.publications_generated = 0
        
    async def run_mad_scientist_cycle(self, domain: str, focus_area: str = None) -> Dict[str, Any]:
        """
        Run a complete mad scientist cycle:
        1. Generate an impossible project
        2. Attempt the project
        3. Analyze failures or breakthroughs
        4. Discover alternatives if needed
        5. Publish insights
        
        Args:
            domain: Domain to focus on
            focus_area: Specific area within the domain
            
        Returns:
            Dict with results of the cycle
        """
        logger.info(f"Starting mad scientist cycle in domain: {domain}, focus: {focus_area}")
        
        cycle_result = {
            "cycle_id": f"mad_scientist_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "domain": domain,
            "focus_area": focus_area,
            "start_time": datetime.now().isoformat(),
            "impossible_project": None,
            "attempt_result": None,
            "alternatives_found": [],
            "publications_created": [],
            "end_time": None,
            "success_metrics": {}
        }
        
        try:
            # Step 1: Generate an impossible project
            project = await self.impossible_projects_module.generate_impossible_project(domain)
            cycle_result["impossible_project"] = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "impossibility_reason": project.impossibility_reason
            }
            
            # Step 2: Attempt the impossible project
            attempt_result = await self.impossible_projects_module.attempt_impossible_project(project.id)
            cycle_result["attempt_result"] = attempt_result
            
            # Update metrics
            self.impossible_projects_attempted += 1
            
            # Step 3: If we have alternatives from the attempt, track them
            if project.alternatives_found:
                cycle_result["alternatives_found"] = project.alternatives_found
                self.alternatives_discovered += len(project.alternatives_found)
                
                # Step 4: For each alternative, see if it warrants publication
                for alt_idx, alternative in enumerate(project.alternatives_found):
                    alt_publication_success = await self.innovation_publisher.publish_systematic_innovation(
                        system_component=f"Alternative_{alt_idx}_to_{project.name}",
                        innovation_description=alternative,
                        impact_assessment=f"Alternative approach discovered during impossible project {project.name}"
                    )
                    if alt_publication_success:
                        cycle_result["publications_created"].append({
                            "type": "alternative_approach",
                            "title": f"Alternative to {project.name}",
                            "description": alternative
                        })
            
            # Step 5: Check if the project results warrant publication
            if project.status in ["breakthrough", "alternative_found", "failed"]:
                insights = []
                if project.breakthrough_insights:
                    insights.extend(project.breakthrough_insights)
                if project.failure_analysis:
                    # Extract insights from failure analysis
                    for analysis in project.failure_analysis:
                        if isinstance(analysis, dict) and "insights_gained" in analysis:
                            insights.extend(analysis["insights_gained"])
                
                if insights:
                    publication_success = await self.innovation_publisher.publish_impossible_project_results(
                        project_name=project.name,
                        outcome=project.status.value,
                        insights=insights
                    )
                    if publication_success:
                        cycle_result["publications_created"].append({
                            "type": "impossible_project_results",
                            "title": f"Results from {project.name}",
                            "outcome": project.status.value
                        })
            
            # Step 6: If we have experimentation module, use it to test alternatives
            if self.experimentation_module and project.alternatives_found:
                for alt_idx, alternative in enumerate(project.alternatives_found):
                    # Generate an experiment based on the alternative
                    try:
                        hypothesis = await self.experimentation_module.generate_hypothesis(
                            domain=domain, 
                            focus_area=f"alternative approach {alt_idx+1}",
                            impossible=False  # Use the alternative as a testable hypothesis
                        )
                        
                        # Design and potentially run an experiment
                        experiment = await self.experimentation_module.design_experiment(
                            hypothesis, 
                            experiment_type=self.experimentation_module.ExperimentType.EXPLORATION
                        )
                        
                        logger.info(f"Created experiment to test alternative: {hypothesis}")
                        
                        # Optionally execute the experiment if conditions are right
                        success_threshold = 0.7  # Only run for high-importance alternatives
                        if project.status == self.impossible_projects_module.ImpossibleProjectStatus.ALTERNATIVE_FOUND:
                            # Execute the experiment for alternatives found from failed projects
                            exp_result = await self.experimentation_module.execute_experiment(experiment.id)
                            
                            # Publish results if significant
                            if exp_result.get('analysis', {}).get('significance', 0) > 0.5:
                                # Create a publication from the experiment result
                                discovery_content = f"Experiment testing alternative approach: {hypothesis}\n\nResult: {str(exp_result)}"
                                pub = await self.innovation_publisher.identify_innovation_for_publication(
                                    discovery_type="experiment_result_from_alternative",
                                    discovery_content=discovery_content,
                                    context=f"Testing alternative discovered from impossible project {project.name}"
                                )
                                
                                if pub:
                                    success = await self.innovation_publisher.attempt_auto_publication(pub)
                                    if success:
                                        cycle_result["publications_created"].append({
                                            "type": "experiment_result",
                                            "title": f"Testing Alternative: {hypothesis[:50]}...",
                                            "hypothesis": hypothesis
                                        })
                    except Exception as e:
                        logger.error(f"Error creating experiment from alternative: {e}")
            
            # Step 7: If the project is physics-related, use the physics prototyping system
            if "physics" in domain.lower() or "physics" in focus_area.lower() if focus_area else False:
                try:
                    # Prototype the physics experiment based on the impossible project
                    physics_result = await self.physics_prototyper.prototype_physics_experiment(
                        project.description,
                        focus_area=focus_area
                    )
                    
                    # Add physics prototyping results to cycle result
                    cycle_result["physics_prototyping"] = {
                        "simulation_id": physics_result.id,
                        "simulation_type": physics_result.simulation_type.value,
                        "success": physics_result.success,
                        "analysis": physics_result.analysis
                    }
                    
                    # Update metrics
                    self.publications_generated += len(physics_result.analysis.get("publications_created", []))
                    
                    logger.info(f"Physics prototyping completed for project: {project.name}")
                except Exception as e:
                    logger.error(f"Error in physics prototyping: {e}")
                    cycle_result["physics_prototyping_error"] = str(e)

            # Step 8: Update success metrics
            cycle_result["success_metrics"] = self.get_mad_scientist_metrics()
            
            logger.info(f"Completed mad scientist cycle: {cycle_result['cycle_id']}")
            
        except Exception as e:
            logger.error(f"Error in mad scientist cycle: {e}")
            cycle_result["error"] = str(e)
        
        finally:
            cycle_result["end_time"] = datetime.now().isoformat()
            self.mad_scientist_cycles_completed += 1
            
        return cycle_result
    
    async def run_extended_mad_scientist_session(self, 
                                               domain: str, 
                                               focus_area: str = None, 
                                               cycles: int = 3) -> List[Dict[str, Any]]:
        """
        Run an extended mad scientist session with multiple cycles.
        
        Args:
            domain: Domain to focus on
            focus_area: Specific area within the domain
            cycles: Number of mad scientist cycles to run
            
        Returns:
            List of results from each cycle
        """
        logger.info(f"Starting extended mad scientist session with {cycles} cycles in domain: {domain}")
        
        results = []
        
        for cycle in range(cycles):
            logger.info(f"Running mad scientist cycle {cycle+1}/{cycles}")
            
            cycle_result = await self.run_mad_scientist_cycle(domain, focus_area)
            results.append(cycle_result)
            
            # Brief pause between cycles to allow for processing
            await asyncio.sleep(2)
        
        # After all cycles, try to publish a summary
        await self.innovation_publisher.publish_periodic_summary()
        
        logger.info(f"Completed extended mad scientist session with {cycles} cycles")
        return results
    
    async def retry_publication_of_innovations(self) -> int:
        """
        Retry publication of innovations that are still pending.
        
        Returns:
            Number of innovations successfully published
        """
        logger.info("Retrying publication of pending innovations")
        
        return await self.innovation_publisher.retry_pending_publications()
    
    async def get_mad_scientist_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about mad scientist activities.
        """
        # Get metrics from all components
        impossible_metrics = self.impossible_projects_module.get_impossible_project_metrics()
        innovation_metrics = self.innovation_publisher.get_publication_metrics()
        physics_prototyping_metrics = await self.physics_prototyper.get_prototyping_metrics()
        
        # Combine with our own metrics
        return {
            "mad_scientist_cycles_completed": self.mad_scientist_cycles_completed,
            "impossible_projects_attempted": self.impossible_projects_attempted,
            "alternatives_discovered": self.alternatives_discovered,
            "publications_generated": self.publications_generated,
            "impossible_project_metrics": impossible_metrics,
            "innovation_publication_metrics": innovation_metrics,
            "physics_prototyping_metrics": physics_prototyping_metrics,
            "system_integration_status": {
                "blog_scheduler_available": self.blog_scheduler is not None,
                "experimentation_module_available": self.experimentation_module is not None,
                "impossible_projects_module_ready": self.impossible_projects_module is not None,
                "innovation_publisher_ready": self.innovation_publisher is not None,
                "physics_prototyper_ready": self.physics_prototyper is not None
            }
        }
    
    async def generate_mad_scientist_report(self) -> str:
        """
        Generate a comprehensive report of mad scientist activities.
        """
        metrics = await self.get_mad_scientist_metrics()
        
        report = f"""
# Mad Scientist System Report

Generated at: {datetime.now().isoformat()}

## System Status
- Cycles Completed: {metrics['mad_scientist_cycles_completed']}
- Impossible Projects Attempted: {metrics['impossible_projects_attempted']}
- Alternatives Discovered: {metrics['alternatives_discovered']}
- Publications Generated: {metrics['publications_generated']}

## Impossible Project Metrics
- Total Projects: {metrics['impossible_project_metrics']['total_impossible_projects']}
- Theoretical: {metrics['impossible_project_metrics']['theoretical_projects']}
- Failed: {metrics['impossible_project_metrics']['failed_projects']}
- Alternative Found: {metrics['impossible_project_metrics']['alternative_found_projects']}
- Breakthrough: {metrics['impossible_project_metrics']['breakthrough_projects']}

## Innovation Publication Metrics
- Total Published: {metrics['innovation_publication_metrics']['total_published_innovations']}
- Pending: {metrics['innovation_publication_metrics']['pending_publications']}
- Success Rate: {metrics['innovation_publication_metrics']['publication_success_rate']:.2%}
- Avg Importance: {metrics['innovation_publication_metrics']['avg_importance_score']:.2f}

## Physics Prototyping Metrics
- Total Simulations: {metrics['physics_prototyping_metrics']['total_simulations']}
- Successful: {metrics['physics_prototyping_metrics']['successful_simulations']}
- Failed: {metrics['physics_prototyping_metrics']['failed_simulations']}
- Success Rate: {metrics['physics_prototyping_metrics']['success_rate']:.2%}
- Simulation Types: {dict(metrics['physics_prototyping_metrics']['simulation_types'])}

## System Integration Status
- Blog Scheduler: {'Available' if metrics['system_integration_status']['blog_scheduler_available'] else 'Not Available'}
- Experimentation Module: {'Available' if metrics['system_integration_status']['experimentation_module_available'] else 'Not Available'}
- Impossible Projects Module: {'Ready' if metrics['system_integration_status']['impossible_projects_module_ready'] else 'Not Ready'}
- Innovation Publisher: {'Ready' if metrics['system_integration_status']['innovation_publisher_ready'] else 'Not Ready'}
- Physics Prototyper: {'Ready' if metrics['system_integration_status']['physics_prototyper_ready'] else 'Not Ready'}

## Summary
The Mad Scientist System has been actively exploring impossible projects, learning from failures, discovering alternative approaches, and publishing innovations. This systematic approach to challenging the impossible has yielded valuable insights and alternative pathways that advance the RAVANA AGI system's capabilities.
        """
        
        return report