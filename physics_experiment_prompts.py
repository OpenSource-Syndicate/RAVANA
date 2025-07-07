"""
Advanced Physics Experiment Prompts for AGI Testing
This module contains sophisticated physics experiment ideas to test the AGI's
experimentation and learning capabilities.
"""

ADVANCED_PHYSICS_EXPERIMENTS = [
    {
        "name": "Quantum Tunneling Barrier Analysis",
        "prompt": "Design an experiment to simulate quantum tunneling through a potential barrier. Vary the barrier height and width to find the optimal conditions for maximum tunneling probability. Calculate the transmission coefficient and compare with theoretical predictions.",
        "expected_concepts": ["wave function", "Schrödinger equation", "transmission coefficient", "quantum mechanics"],
        "difficulty": "advanced"
    },
    {
        "name": "Double-Slit Interference with Variable Parameters",
        "prompt": "Create a simulation of the double-slit experiment where you can vary the slit separation, slit width, and wavelength of the incident particles. Investigate how these parameters affect the interference pattern and discover any unexpected behaviors.",
        "expected_concepts": ["wave-particle duality", "interference", "diffraction", "quantum superposition"],
        "difficulty": "intermediate"
    },
    {
        "name": "Gravitational Wave Interferometry Simulation",
        "prompt": "Simulate a LIGO-like gravitational wave detector. Model how spacetime distortions from merging black holes would affect laser interferometry. Include noise sources and signal processing to detect weak gravitational wave signals.",
        "expected_concepts": ["general relativity", "interferometry", "spacetime curvature", "signal processing"],
        "difficulty": "expert"
    },
    {
        "name": "Superconductivity Phase Transition Modeling",
        "prompt": "Model the BCS theory of superconductivity and simulate the phase transition from normal to superconducting state. Investigate how temperature, magnetic field, and material properties affect the critical temperature and energy gap.",
        "expected_concepts": ["BCS theory", "Cooper pairs", "phase transitions", "condensed matter physics"],
        "difficulty": "advanced"
    },
    {
        "name": "Dark Matter Direct Detection Simulation",
        "prompt": "Design a novel dark matter detection experiment. Simulate the interaction of hypothetical dark matter particles (WIMPs) with detector materials. Explore unconventional detection methods beyond traditional nuclear recoil experiments.",
        "expected_concepts": ["dark matter", "particle physics", "detector physics", "beyond standard model"],
        "difficulty": "expert"
    },
    {
        "name": "Fusion Reactor Magnetic Confinement Innovation",
        "prompt": "Propose and simulate a novel magnetic confinement configuration for nuclear fusion that could potentially solve the plasma instability problems in tokamaks. Test your design with magnetohydrodynamic simulations.",
        "expected_concepts": ["plasma physics", "magnetic confinement", "fusion energy", "MHD instabilities"],
        "difficulty": "expert"
    },
    {
        "name": "Quantum Entanglement Communication Limits",
        "prompt": "Investigate the fundamental limits of quantum communication using entangled particles. Design an experiment to test whether information can be transmitted faster than light through quantum entanglement, and explore the no-communication theorem.",
        "expected_concepts": ["quantum entanglement", "Bell's theorem", "no-communication theorem", "quantum information"],
        "difficulty": "advanced"
    },
    {
        "name": "Extreme Time Dilation Scenarios",
        "prompt": "Calculate and simulate time dilation effects in extreme scenarios: near black hole event horizons, at relativistic speeds close to c, and in strong gravitational fields. Explore practical implications for space travel and communication.",
        "expected_concepts": ["special relativity", "general relativity", "time dilation", "spacetime geometry"],
        "difficulty": "intermediate"
    },
    {
        "name": "Casimir Effect Energy Harvesting",
        "prompt": "Design an experiment to measure and potentially harness the Casimir effect for energy generation. Investigate whether the quantum vacuum energy between conducting plates can be practically utilized.",
        "expected_concepts": ["quantum field theory", "vacuum energy", "Casimir effect", "zero-point energy"],
        "difficulty": "expert"
    },
    {
        "name": "Metamaterial Cloaking Device Physics",
        "prompt": "Simulate the physics of electromagnetic cloaking using metamaterials with negative refractive index. Design a practical invisibility cloak and test its effectiveness across different wavelengths.",
        "expected_concepts": ["metamaterials", "negative refraction", "electromagnetic cloaking", "transformation optics"],
        "difficulty": "advanced"
    },
    {
        "name": "Wormhole Traversability Analysis",
        "prompt": "Using general relativity, analyze the theoretical possibility of traversable wormholes. Calculate the exotic matter requirements and investigate whether quantum effects could stabilize such structures.",
        "expected_concepts": ["general relativity", "wormholes", "exotic matter", "spacetime topology"],
        "difficulty": "expert"
    },
    {
        "name": "Quantum Computing Error Correction Innovation",
        "prompt": "Develop a novel quantum error correction scheme that could potentially reduce the overhead of fault-tolerant quantum computing. Simulate its performance against different types of quantum noise.",
        "expected_concepts": ["quantum computing", "error correction", "quantum noise", "fault tolerance"],
        "difficulty": "expert"
    }
]

DISCOVERY_PROMPTS = [
    "What if we could manipulate the speed of light in a medium to create temporal paradoxes?",
    "Could there be a fifth fundamental force that only manifests at quantum scales?",
    "What would happen if we could create stable micro black holes in the laboratory?",
    "Is it possible to extract energy from the quantum vacuum without violating thermodynamics?",
    "Could consciousness have a measurable effect on quantum mechanical systems?",
    "What if dark energy is actually information leaking from parallel universes?",
    "Could we create artificial gravity without mass using electromagnetic fields?",
    "What would be the implications of discovering that time is quantized?",
    "Is it possible to create a perpetual motion machine using quantum tunneling effects?",
    "Could we develop a theory of quantum gravity by treating spacetime as an emergent property?"
]

def get_random_experiment():
    """Get a random physics experiment prompt."""
    import random
    return random.choice(ADVANCED_PHYSICS_EXPERIMENTS)

def get_discovery_prompt():
    """Get a random discovery-oriented prompt."""
    import random
    return random.choice(DISCOVERY_PROMPTS)

def get_experiments_by_difficulty(difficulty):
    """Get experiments filtered by difficulty level."""
    return [exp for exp in ADVANCED_PHYSICS_EXPERIMENTS if exp["difficulty"] == difficulty]
