"""Tests for decision engine functionality."""

import pytest
from modules.decision_engine.decision_maker import goal_driven_decision_maker_loop


class TestDecisionEngine:
    """Test decision engine functionality."""

    def test_goal_driven_decision_basic(self):
        """Test basic goal-driven decision making."""
        situation = {
            'type': 'normal',
            'prompt': 'Test situation for decision making',
            'context': {}
        }
        memory = ["Previous experience 1", "Previous experience 2"]
        shared_state = {}
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            memory=memory,
            shared_state=shared_state
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert 'action' in decision or 'reason' in decision

    def test_decision_with_hypotheses(self):
        """Test decision making with hypotheses."""
        situation = {'prompt': 'Test with hypotheses'}
        hypotheses = [
            "Hypothesis 1: System performs better with X",
            "Hypothesis 2: Approach Y is more efficient"
        ]
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            hypotheses=hypotheses
        )
        
        assert decision is not None

    def test_experiment_initiation(self):
        """Test experiment initiation through decision making."""
        situation = {
            'type': 'exploration',
            'prompt': 'Explore new optimization strategies'
        }
        shared_state = {'active_experiment': None}
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            shared_state=shared_state
        )
        
        assert decision is not None

    def test_experiment_analysis(self):
        """Test experiment outcome analysis."""
        situation = {
            'type': 'experiment_analysis',
            'context': {
                'hypothesis': 'Test hypothesis',
                'situation_prompt': 'Test prompt',
                'outcome': 'positive'
            }
        }
        
        decision = goal_driven_decision_maker_loop(situation=situation)
        
        assert decision is not None
