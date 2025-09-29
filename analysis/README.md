# RAVANA Analysis Directory

This directory contains experimental work and analysis for the RAVANA AGI system.

## Directory Structure

- `experiments/` - Experimental code and scripts
- `data/` - Datasets and input files  
- `results/` - Output and results from experiments
- `models/` - Trained models and checkpoints
- `notebooks/` - Jupyter notebooks for analysis
- `logs/` - Log files from experimental runs

## Getting Started

To run an experiment:

1. Create your experiment script in the `experiments/` directory
2. Place any required data in the `data/` directory
3. Run your experiment
4. Results will be saved to the `results/` directory

## Experiment Template

Create a new experiment by copying the template below:

```python
# analysis/experiments/my_experiment.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.system import AGISystem
from database.engine import create_db_and_tables, engine
from core.standard_config import StandardConfig

def run_experiment():
    """Run your experiment here"""
    print("Running experiment...")
    
    # Your experimental code here
    
    print("Experiment completed")

if __name__ == "__main__":
    run_experiment()
```