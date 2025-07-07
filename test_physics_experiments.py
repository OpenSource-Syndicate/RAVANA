"""
Comprehensive Physics Experimentation Test Suite for AGI System
Tests the "experiment and learn" feature with advanced physics experiments.
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Import AGI system components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.system import AGISystem
from database.engine import create_db_and_tables
from modules.decision_engine.llm import agi_experimentation_engine
from physics_experiment_prompts import (
    ADVANCED_PHYSICS_EXPERIMENTS, 
    DISCOVERY_PROMPTS,
    get_random_experiment,
    get_discovery_prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('physics_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PhysicsExperimentTester:
    """Test suite for physics experimentation capabilities."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.experiment_count = 0
        
    async def setup(self):
        """Initialize the AGI system for testing."""
        logger.info("Setting up AGI system for physics experiments...")
        self.engine = create_db_and_tables()
        self.agi_system = AGISystem(self.engine)
        self.start_time = datetime.now()
        
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'agi_system'):
            await self.agi_system.stop()
        logger.info("Cleanup completed.")
        
    async def test_single_experiment(self, experiment_data):
        """Test a single physics experiment."""
        logger.info(f"Testing experiment: {experiment_data['name']}")
        
        start_time = time.time()
        result = {
            'experiment_name': experiment_data['name'],
            'difficulty': experiment_data['difficulty'],
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'execution_time': 0,
            'generated_code': None,
            'scientific_validity': None,
            'novel_insights': None
        }
        
        try:
            # Use the AGI experimentation engine
            experiment_result = agi_experimentation_engine(
                experiment_idea=experiment_data['prompt'],
                use_chain_of_thought=True,
                online_validation=True,
                verbose=True
            )
            
            result['success'] = True
            result['generated_code'] = experiment_result.get('generated_code')
            result['final_verdict'] = experiment_result.get('final_verdict')
            result['execution_result'] = experiment_result.get('execution_result')
            result['online_validation'] = experiment_result.get('online_validation')
            result['steps'] = experiment_result.get('steps', [])
            
            # Analyze scientific validity
            if experiment_result.get('final_verdict'):
                if 'success' in experiment_result['final_verdict'].lower():
                    result['scientific_validity'] = 'high'
                elif 'potential' in experiment_result['final_verdict'].lower():
                    result['scientific_validity'] = 'medium'
                else:
                    result['scientific_validity'] = 'low'
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Experiment failed: {e}")
            
        result['execution_time'] = time.time() - start_time
        self.results.append(result)
        self.experiment_count += 1
        
        return result
        
    async def test_discovery_mode(self, discovery_prompt):
        """Test the AGI's ability to discover new physics concepts."""
        logger.info(f"Testing discovery mode with prompt: {discovery_prompt[:100]}...")
        
        start_time = time.time()
        result = {
            'prompt': discovery_prompt,
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'execution_time': 0,
            'creativity_score': 0,
            'scientific_plausibility': 0,
            'novel_concepts': []
        }
        
        try:
            # Use the experimentation engine for discovery
            experiment_result = agi_experimentation_engine(
                experiment_idea=discovery_prompt,
                use_chain_of_thought=True,
                online_validation=True,
                verbose=True
            )
            
            result['success'] = True
            result['generated_response'] = experiment_result.get('refined_idea')
            result['simulation_approach'] = experiment_result.get('simulation_type')
            result['final_assessment'] = experiment_result.get('final_verdict')
            
            # Score creativity and plausibility (simple heuristics)
            if experiment_result.get('refined_idea'):
                idea_text = experiment_result['refined_idea'].lower()
                
                # Creativity indicators
                creative_words = ['novel', 'innovative', 'unprecedented', 'breakthrough', 'revolutionary']
                result['creativity_score'] = sum(1 for word in creative_words if word in idea_text)
                
                # Scientific plausibility indicators
                science_words = ['theory', 'equation', 'measurement', 'experiment', 'hypothesis']
                result['scientific_plausibility'] = sum(1 for word in science_words if word in idea_text)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Discovery test failed: {e}")
            
        result['execution_time'] = time.time() - start_time
        return result
        
    async def test_agi_integration(self, experiment_prompt):
        """Test full AGI system integration with physics experiments."""
        logger.info("Testing full AGI system integration...")
        
        try:
            # Test the propose_and_test_invention action
            task_prompt = f"I want you to propose and test this physics experiment: {experiment_prompt}"
            
            # Run single task through AGI system
            await self.agi_system.run_single_task(task_prompt)
            
            # Check if experiment was logged
            recent_logs = []
            try:
                from sqlmodel import select
                from database.models import ExperimentLog
                
                stmt = select(ExperimentLog).order_by(ExperimentLog.timestamp.desc()).limit(5)
                recent_logs = self.agi_system.session.exec(stmt).all()
                
            except Exception as e:
                logger.warning(f"Could not retrieve experiment logs: {e}")
            
            return {
                'integration_success': True,
                'recent_experiment_logs': len(recent_logs),
                'agi_system_responsive': True
            }
            
        except Exception as e:
            logger.error(f"AGI integration test failed: {e}")
            return {
                'integration_success': False,
                'error': str(e),
                'agi_system_responsive': False
            }
            
    def generate_report(self):
        """Generate a comprehensive test report."""
        if not self.results:
            return "No experiments were run."
            
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r['success'])
        
        # Calculate statistics
        avg_execution_time = sum(r['execution_time'] for r in self.results) / total_experiments
        
        difficulty_stats = {}
        for result in self.results:
            if 'difficulty' in result:
                diff = result['difficulty']
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {'total': 0, 'successful': 0}
                difficulty_stats[diff]['total'] += 1
                if result['success']:
                    difficulty_stats[diff]['successful'] += 1
        
        report = f"""
=== PHYSICS EXPERIMENTATION TEST REPORT ===
Generated: {datetime.now().isoformat()}
Test Duration: {datetime.now() - self.start_time if self.start_time else 'Unknown'}

SUMMARY:
- Total Experiments: {total_experiments}
- Successful: {successful_experiments} ({successful_experiments/total_experiments*100:.1f}%)
- Failed: {total_experiments - successful_experiments}
- Average Execution Time: {avg_execution_time:.2f} seconds

DIFFICULTY BREAKDOWN:
"""
        
        for diff, stats in difficulty_stats.items():
            success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
            report += f"- {diff.title()}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)\n"
        
        report += "\nDETAILED RESULTS:\n"
        for i, result in enumerate(self.results, 1):
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            report += f"{i}. {result.get('experiment_name', 'Unknown')} - {status}\n"
            if result.get('error'):
                report += f"   Error: {result['error']}\n"
            if result.get('scientific_validity'):
                report += f"   Scientific Validity: {result['scientific_validity']}\n"
        
        return report

async def run_comprehensive_physics_tests():
    """Run the complete physics experimentation test suite."""
    tester = PhysicsExperimentTester()
    
    try:
        await tester.setup()
        logger.info("Starting comprehensive physics experimentation tests...")
        
        # Test 1: Run a selection of physics experiments
        logger.info("=== PHASE 1: TESTING INDIVIDUAL EXPERIMENTS ===")
        
        # Test different difficulty levels
        for difficulty in ['intermediate', 'advanced', 'expert']:
            experiments = [exp for exp in ADVANCED_PHYSICS_EXPERIMENTS if exp['difficulty'] == difficulty]
            if experiments:
                # Test first experiment of each difficulty
                experiment = experiments[0]
                logger.info(f"Testing {difficulty} experiment: {experiment['name']}")
                result = await tester.test_single_experiment(experiment)
                
                if result['success']:
                    logger.info(f"✓ {experiment['name']} completed successfully")
                else:
                    logger.error(f"✗ {experiment['name']} failed: {result.get('error', 'Unknown error')}")
        
        # Test 2: Discovery mode tests
        logger.info("=== PHASE 2: TESTING DISCOVERY MODE ===")
        
        for i, discovery_prompt in enumerate(DISCOVERY_PROMPTS[:3], 1):  # Test first 3
            logger.info(f"Discovery test {i}/3")
            discovery_result = await tester.test_discovery_mode(discovery_prompt)
            
            if discovery_result['success']:
                logger.info(f"✓ Discovery test {i} completed")
                logger.info(f"  Creativity Score: {discovery_result['creativity_score']}")
                logger.info(f"  Scientific Plausibility: {discovery_result['scientific_plausibility']}")
            else:
                logger.error(f"✗ Discovery test {i} failed")
        
        # Test 3: AGI System Integration
        logger.info("=== PHASE 3: TESTING AGI SYSTEM INTEGRATION ===")
        
        integration_test_prompt = "Design a quantum experiment to test if consciousness affects wave function collapse"
        integration_result = await tester.test_agi_integration(integration_test_prompt)
        
        if integration_result['integration_success']:
            logger.info("✓ AGI system integration test passed")
        else:
            logger.error(f"✗ AGI system integration test failed: {integration_result.get('error')}")
        
        # Generate and save report
        report = tester.generate_report()
        
        # Save report to file
        report_path = Path("physics_experiment_test_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report saved to: {report_path}")
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        return tester.results
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise
    finally:
        await tester.cleanup()

async def run_single_experiment_test(experiment_name=None):
    """Run a single experiment test for quick debugging."""
    tester = PhysicsExperimentTester()
    
    try:
        await tester.setup()
        
        if experiment_name:
            # Find specific experiment
            experiment = next((exp for exp in ADVANCED_PHYSICS_EXPERIMENTS if exp['name'] == experiment_name), None)
            if not experiment:
                logger.error(f"Experiment '{experiment_name}' not found")
                return
        else:
            # Use a simple experiment for quick testing
            experiment = {
                "name": "Simple Quantum Tunneling Test",
                "prompt": "Create a basic simulation of quantum tunneling through a rectangular potential barrier. Calculate the transmission probability for an electron with energy less than the barrier height.",
                "difficulty": "intermediate"
            }
        
        logger.info(f"Running single experiment: {experiment['name']}")
        result = await tester.test_single_experiment(experiment)
        
        print("\n" + "="*50)
        print(f"EXPERIMENT: {experiment['name']}")
        print(f"SUCCESS: {result['success']}")
        print(f"EXECUTION TIME: {result['execution_time']:.2f}s")
        
        if result['success']:
            print(f"SCIENTIFIC VALIDITY: {result.get('scientific_validity', 'Unknown')}")
            if result.get('final_verdict'):
                print(f"FINAL VERDICT: {result['final_verdict']}")
        else:
            print(f"ERROR: {result.get('error', 'Unknown error')}")
        
        print("="*50)
        
        return result
        
    except Exception as e:
        logger.error(f"Single experiment test failed: {e}")
        raise
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Run single experiment test
        experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_single_experiment_test(experiment_name))
    else:
        # Run comprehensive test suite
        asyncio.run(run_comprehensive_physics_tests())
