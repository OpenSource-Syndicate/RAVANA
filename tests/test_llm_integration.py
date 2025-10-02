"""Tests for LLM integration and decision making."""

import pytest
import json
from core.llm import (
    call_llm,
    extract_decision,
    decision_maker_loop,
    safe_call_llm,
    _fix_truncated_json
)


class TestLLMIntegration:
    """Test LLM integration functionality."""

    def test_call_llm_basic(self):
        """Test basic LLM call."""
        prompt = "What is 2+2?"
        response = call_llm(prompt)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_extract_decision_valid_json(self):
        """Test decision extraction from valid JSON."""
        raw_response = '''```json
        {
            "analysis": "Test analysis",
            "plan": ["step1", "step2"],
            "action": "log_message",
            "params": {"message": "test"}
        }
        ```'''
        
        decision = extract_decision(raw_response)
        
        assert decision is not None
        assert 'action' in decision
        assert 'params' in decision
        assert decision['action'] == 'log_message'

    def test_extract_decision_truncated_json(self):
        """Test decision extraction from truncated JSON."""
        raw_response = '''{"analysis": "Test", "plan": ["step1"'''
        
        decision = extract_decision(raw_response)
        
        # Should handle truncation gracefully
        assert decision is not None
        assert 'raw_response' in decision

    def test_fix_truncated_json(self):
        """Test JSON fixing functionality."""
        truncated = '{"test": "value", "nested": {"key"'
        fixed = _fix_truncated_json(truncated)
        
        # Should attempt to close brackets
        assert fixed.count('{') == fixed.count('}')

    def test_decision_maker_loop(self):
        """Test decision maker loop."""
        situation = {
            'prompt': 'Test situation',
            'context': {}
        }
        memory = ["Previous interaction 1", "Previous interaction 2"]
        mood = {"Curious": 0.7, "Confident": 0.5}
        actions = [
            {
                'name': 'log_message',
                'description': 'Log a message',
                'parameters': [{'name': 'message', 'type': 'string'}]
            }
        ]
        
        decision = decision_maker_loop(
            situation=situation,
            memory=memory,
            mood=mood,
            actions=actions
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert 'action' in decision or 'raw_response' in decision

    def test_safe_call_llm_with_retry(self):
        """Test safe LLM call with retry logic."""
        prompt = "Test prompt"
        response = safe_call_llm(prompt, retries=2)
        
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_async_safe_call_llm(self):
        """Test async LLM call."""
        from core.llm import async_safe_call_llm
        
        prompt = "Test async prompt"
        response = await async_safe_call_llm(prompt)
        
        assert response is not None
        assert isinstance(response, str)

    def test_decision_with_persona(self):
        """Test decision making with persona."""
        situation = {'prompt': 'Test with persona'}
        persona = {
            'name': 'Ravana',
            'traits': ['analytical', 'curious'],
            'creativity': 0.7,
            'communication_style': {
                'tone': 'professional',
                'encouragement': 'positive'
            }
        }
        
        decision = decision_maker_loop(
            situation=situation,
            persona=persona
        )
        
        assert decision is not None
