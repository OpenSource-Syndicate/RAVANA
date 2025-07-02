from llm import generate_hypothetical_scenarios, decision_maker_loop

def test_generate_hypothetical_scenarios():
    print("=== Testing generate_hypothetical_scenarios ===")
    trends = ["AI regulation", "quantum computing", "climate tech"]
    interest_areas = ["technology", "finance"]
    gap_topics = ["space exploration"]
    scenarios = generate_hypothetical_scenarios(trends=trends, interest_areas=interest_areas, gap_topics=gap_topics)
    print("Generated Scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    print()

def test_decision_maker_loop():
    print("=== Testing decision_maker_loop ===")
    situation = "A new AI regulation is being discussed in the parliament."
    memory = [
        "Previous AI regulations had significant impact on startups.",
        "Public opinion is divided on strict AI laws."
    ]
    rag_context = "Recent news articles suggest a shift towards more lenient policies."
    result = decision_maker_loop(situation, memory=memory, rag_context=rag_context)
    print("Decision Maker Output:")
    print(result)
    print()

if __name__ == "__main__":
    test_generate_hypothetical_scenarios()
    test_decision_maker_loop()