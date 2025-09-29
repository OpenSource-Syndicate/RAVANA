import asyncio
from modules.situation_generator.situation_generator import SituationGenerator


async def test():
    g = SituationGenerator()

    class SS:
        pass
    ss = SS()
    ss.episodic_memories = ['Fixed the servo, it stalled and smelled of ozone.',
                            'Meeting about roadmap: stalled priorities, strong opinions.']
    ss.emotional_state = {'dominant_emotion': 'anxious', 'valence': -0.45}
    ss.curiosity_feed = ['quantum resonance in biological tissue']
    s = await g.generate_situation(ss, behavior_modifiers={'subconscious_mode': True, 'intensity': 0.9})
    import json
    print(json.dumps(s, indent=2))

if __name__ == '__main__':
    asyncio.run(test())
