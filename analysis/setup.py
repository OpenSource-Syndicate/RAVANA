"""
Setup script for RAVANA analysis environment
This script creates the necessary structure for experimental work
"""

import os
import json
from datetime import datetime


def setup_analysis_environment():
    """Setup the analysis directory with necessary subdirectories and files"""

    # Create analysis directory structure
    analysis_dirs = [
        "experiments",
        "data",
        "results",
        "models",
        "notebooks",
        "logs"
    ]

    for dir_name in analysis_dirs:
        path = os.path.join("analysis", dir_name)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    # Create a basic configuration file for analysis
    config = {
        "created_at": datetime.now().isoformat(),
        "description": "Analysis environment for RAVANA experimental work",
        "directories": {
            "experiments": "Experimental code and scripts",
            "data": "Datasets and input files",
            "results": "Output and results from experiments",
            "models": "Trained models and checkpoints",
            "notebooks": "Jupyter notebooks for analysis",
            "logs": "Log files from experimental runs"
        },
        "settings": {
            "backup_enabled": True,
            "logging_level": "INFO",
            "experiment_timeout": 3600,
            "max_experiment_concurrency": 4
        }
    }

    config_path = os.path.join("analysis", "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created analysis configuration: {config_path}")

    # Create a README for the analysis directory
    readme_lines = [
        "# RAVANA Analysis Directory",
        "",
        "This directory contains experimental work and analysis for the RAVANA AGI system.",
        "",
        "## Directory Structure",
        "",
        "- `experiments/` - Experimental code and scripts",
        "- `data/` - Datasets and input files  ",
        "- `results/` - Output and results from experiments",
        "- `models/` - Trained models and checkpoints",
        "- `notebooks/` - Jupyter notebooks for analysis",
        "- `logs/` - Log files from experimental runs",
        "",
        "## Getting Started",
        "",
        "To run an experiment:",
        "",
        "1. Create your experiment script in the `experiments/` directory",
        "2. Place any required data in the `data/` directory",
        "3. Run your experiment",
        "4. Results will be saved to the `results/` directory",
        "",
        "## Experiment Template",
        "",
        "Create a new experiment by copying the template below:",
        "",
        "```python",
        "# analysis/experiments/my_experiment.py",
        "",
        "import sys",
        "import os",
        "sys.path.append(os.path.join(os.path.dirname(__file__), '..'))",
        "",
        "from core.system import AGISystem",
        "from database.engine import create_db_and_tables, engine",
        "from core.standard_config import StandardConfig",
        "",
        "def run_experiment():",
        '    """Run your experiment here"""',
        "    print(\"Running experiment...\")",
        "    ",
        "    # Your experimental code here",
        "    ",
        "    print(\"Experiment completed\")",
        "",
        "if __name__ == \"__main__\":",
        "    run_experiment()",
        "```",
    ]

    readme_content = "\n".join(readme_lines)

    readme_path = os.path.join("analysis", "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"Created analysis README: {readme_path}")

    # Copy the pre-created template
    import shutil
    if os.path.exists("analysis/experiments/template_fixed.py"):
        shutil.copy("analysis/experiments/template_fixed.py",
                    "analysis/experiments/template.py")
        print("Created experiment template: analysis/experiments/template.py")

    print("Analysis environment setup complete!")


if __name__ == "__main__":
    setup_analysis_environment()
