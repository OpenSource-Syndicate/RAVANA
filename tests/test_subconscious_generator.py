import asyncio
import pytest

from modules.situation_generator.situation_generator import SituationGenerator


@pytest.mark.asyncio
async def test_subconscious_structure_and_nudge():
    g = SituationGenerator()

    class SharedState:
        pass

    ss = SharedState()
    ss.episodic_memories = [
        'Deployed a model that crashed under load.',
        'Team meeting: roadmap delayed, heated words.',
        'Fixed a socket leak last week.'
    ]
    ss.emotional_state = {'dominant_emotion': 'anxious', 'valence': -0.6}
    ss.curiosity_feed = ['novel materials for soft robotics']

    situation = await g.generate_situation(ss, behavior_modifiers={'subconscious_mode': True, 'intensity': 0.8})

    # Basic structure checks
    assert isinstance(situation, dict)
    assert situation.get('type') == 'subconscious'
    assert 'prompt' in situation
    assert 'context' in situation

    # Because valence is low in this test, expect a nudge or a message in polished prompt
    nudge = situation.get('nudge')
    polished = situation.get('prompt', '')

    assert (nudge is not None) or ('pause' in polished.lower() or 're-evaluat' in polished.lower() or 'sanity' in polished.lower())
