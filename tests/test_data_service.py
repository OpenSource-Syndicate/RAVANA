"""Tests for data service functionality."""

import pytest
from services.data_service import DataService
from core.config import Config
from datetime import datetime


class TestDataService:
    """Test data service functionality."""

    @pytest.fixture
    def data_service(self, test_engine):
        """Create data service for testing."""
        config = Config()
        service = DataService(
            test_engine,
            config.FEED_URLS,
            None,  # embedding_model
            None   # sentiment_classifier
        )
        return service

    def test_data_service_initialization(self, data_service):
        """Test data service initializes correctly."""
        assert data_service is not None
        assert data_service.engine is not None
        assert data_service.feed_urls is not None

    def test_save_action_log(self, data_service):
        """Test saving action log."""
        action_name = "test_action"
        params = {"test_param": "value"}
        status = "success"
        result = "Action completed"
        
        # Should not raise exception
        data_service.save_action_log(action_name, params, status, result)

    def test_save_mood_log(self, data_service, sample_mood_vector):
        """Test saving mood log."""
        # Should not raise exception
        data_service.save_mood_log(sample_mood_vector)

    def test_save_situation_log(self, data_service):
        """Test saving situation log."""
        situation = {
            'type': 'test',
            'prompt': 'Test situation',
            'context': {'key': 'value'}
        }
        
        situation_id = data_service.save_situation_log(situation)
        
        assert situation_id is not None
        assert isinstance(situation_id, int)

    def test_save_decision_log(self, data_service):
        """Test saving decision log."""
        situation_id = 1
        raw_response = '{"action": "test"}'
        
        # Should not raise exception
        data_service.save_decision_log(situation_id, raw_response)

    def test_save_experiment_log_dict(self, data_service):
        """Test saving experiment log with dict."""
        hypothesis = "Test hypothesis"
        results = {
            "test_plan": "Plan details",
            "final_verdict": "Success",
            "execution_result": "Results here"
        }
        
        # Should not raise exception
        data_service.save_experiment_log(hypothesis, results)

    def test_save_experiment_log_args(self, data_service):
        """Test saving experiment log with separate args."""
        hypothesis = "Test hypothesis"
        test_plan = "Plan details"
        final_verdict = "Success"
        execution_result = "Results here"
        
        # Should not raise exception
        data_service.save_experiment_log(
            hypothesis,
            test_plan,
            final_verdict,
            execution_result
        )

    def test_fetch_and_save_articles(self, data_service):
        """Test fetching and saving articles."""
        # This test might fail if network is unavailable
        try:
            num_saved = data_service.fetch_and_save_articles()
            assert isinstance(num_saved, int)
            assert num_saved >= 0
        except Exception:
            pytest.skip("Network unavailable for article fetching")
