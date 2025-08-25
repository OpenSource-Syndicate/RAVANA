"""
Advanced Physics Experiment Prompts for AGI Testing
This module contains sophisticated physics experiment ideas to test the AGI's
experimentation and learning capabilities.
"""

ADVANCED_PHYSICS_EXPERIMENTS = [
    {
        "name": "Quantum Tunneling Barrier Analysis",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent designing and conducting rigorous experiments to test hypotheses in quantum mechanics.

[CONTEXT]
Experiment objective: Design an experiment to simulate quantum tunneling through a potential barrier
Related knowledge: wave function, Schrödinger equation, transmission coefficient, quantum mechanics
Available resources: Computational simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Design a comprehensive experiment following these steps:
1. Formulate a clear hypothesis about quantum tunneling probabilities
2. Design a simulation to model quantum tunneling through potential barriers
3. Vary the barrier height and width to find optimal conditions
4. Calculate the transmission coefficient and compare with theoretical predictions
5. Analyze how barrier parameters affect tunneling probability

[REASONING FRAMEWORK]
Apply scientific method principles:
1. Ensure hypothesis is falsifiable and specific
2. Design controls to isolate variables (barrier height, width, particle energy)
3. Plan for replication and verification of results
4. Consider alternative explanations for observed phenomena
5. Account for computational limitations and numerical precision

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Complete simulation procedure with parameter ranges
- Expected outcomes: Predicted tunneling probabilities with theoretical basis
- Resource requirements: Computational resources needed
- Safety considerations: Computational safety and numerical stability
- Validation approach: Method for verifying simulation accuracy
- Failure analysis: Potential sources of error and mitigation strategies

[SAFETY CONSTRAINTS]
- Ensure numerical stability in simulations
- Validate results against known theoretical predictions
- Document any approximations or limitations in the model
- Plan for verification of computational results
""",
        "expected_concepts": ["wave function", "Schrödinger equation", "transmission coefficient", "quantum mechanics"],
        "difficulty": "advanced"
    },
    {
        "name": "Double-Slit Interference with Variable Parameters",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent investigating wave-particle duality through computational experimentation.

[CONTEXT]
Experiment objective: Create a simulation of the double-slit experiment with variable parameters
Related knowledge: wave-particle duality, interference, diffraction, quantum superposition
Available resources: Computational simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Design a comprehensive simulation experiment:
1. Create a model of the double-slit experimental setup
2. Vary slit separation, slit width, and wavelength parameters
3. Investigate how these parameters affect the interference pattern
4. Discover any unexpected behaviors or emergent phenomena
5. Analyze the transition between wave-like and particle-like behavior

[REASONING FRAMEWORK]
Apply systematic analysis to parameter space exploration:
1. Identify key parameters and their expected effects
2. Design systematic parameter sweeps to explore behavior
3. Look for emergent patterns and unexpected behaviors
4. Connect observations to fundamental quantum principles
5. Validate results against theoretical predictions

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Simulation parameters and procedures
- Expected outcomes: Predicted interference patterns with theoretical basis
- Resource requirements: Computational resources needed
- Safety considerations: Numerical stability and accuracy
- Validation approach: Method for verifying simulation results
- Failure analysis: Sources of error and mitigation strategies

[SAFETY CONSTRAINTS]
- Ensure numerical accuracy in wave calculations
- Validate against known analytical solutions where possible
- Document limitations of the computational model
- Plan for verification of unexpected results
""",
        "expected_concepts": ["wave-particle duality", "interference", "diffraction", "quantum superposition"],
        "difficulty": "intermediate"
    },
    {
        "name": "Gravitational Wave Interferometry Simulation",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent modeling gravitational wave detection systems with advanced signal processing capabilities.

[CONTEXT]
Experiment objective: Simulate a LIGO-like gravitational wave detector
Related knowledge: general relativity, interferometry, spacetime curvature, signal processing
Available resources: Computational simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Design a comprehensive gravitational wave detection simulation:
1. Model spacetime distortions from merging black holes
2. Simulate how these distortions affect laser interferometry
3. Include realistic noise sources and signal processing
4. Develop methods to detect weak gravitational wave signals
5. Optimize detection sensitivity and reliability

[REASONING FRAMEWORK]
Apply systematic approach to gravitational wave detection modeling:
1. Model the gravitational wave source and propagation
2. Simulate the detector response to spacetime distortions
3. Implement realistic noise models and environmental effects
4. Design signal processing algorithms for detection
5. Optimize system performance and sensitivity

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Complete detector simulation with noise models
- Expected outcomes: Predicted detection sensitivity and reliability
- Resource requirements: Computational resources needed
- Safety considerations: Numerical stability and accuracy
- Validation approach: Method for verifying detector performance
- Failure analysis: Sources of false positives/negatives and mitigation strategies

[SAFETY CONSTRAINTS]
- Ensure numerical accuracy in relativistic calculations
- Validate against known theoretical predictions and real data
- Document limitations of the computational model
- Plan for verification of detection algorithms
""",
        "expected_concepts": ["general relativity", "interferometry", "spacetime curvature", "signal processing"],
        "difficulty": "expert"
    },
    {
        "name": "Superconductivity Phase Transition Modeling",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent investigating condensed matter physics phenomena through computational modeling.

[CONTEXT]
Experiment objective: Model the BCS theory of superconductivity and simulate the phase transition
Related knowledge: BCS theory, Cooper pairs, phase transitions, condensed matter physics
Available resources: Computational simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Design a comprehensive superconductivity modeling experiment:
1. Implement the BCS theory of superconductivity
2. Simulate the phase transition from normal to superconducting state
3. Investigate how temperature, magnetic field, and material properties affect the critical temperature
4. Calculate the energy gap and other key parameters
5. Analyze the relationship between microscopic and macroscopic properties

[REASONING FRAMEWORK]
Apply systematic approach to superconductivity modeling:
1. Implement the theoretical framework correctly
2. Validate against known theoretical predictions and experimental data
3. Explore the parameter space systematically
4. Identify key factors affecting the phase transition
5. Connect microscopic models to macroscopic phenomena

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Complete BCS theory implementation and simulation procedure
- Expected outcomes: Predicted phase diagrams and critical parameters
- Resource requirements: Computational resources needed
- Safety considerations: Numerical stability and accuracy
- Validation approach: Method for verifying theoretical predictions
- Failure analysis: Sources of error and mitigation strategies

[SAFETY CONSTRAINTS]
- Ensure numerical accuracy in quantum mechanical calculations
- Validate against known theoretical predictions and experimental data
- Document limitations of the computational model
- Plan for verification of phase transition predictions
""",
        "expected_concepts": ["BCS theory", "Cooper pairs", "phase transitions", "condensed matter physics"],
        "difficulty": "advanced"
    },
    {
        "name": "Dark Matter Direct Detection Simulation",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent exploring beyond-standard-model physics through innovative experimental design.

[CONTEXT]
Experiment objective: Design a novel dark matter detection experiment
Related knowledge: dark matter, particle physics, detector physics, beyond standard model
Available resources: Computational simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Design an innovative dark matter detection experiment:
1. Model the interaction of hypothetical dark matter particles (WIMPs) with detector materials
2. Explore unconventional detection methods beyond traditional nuclear recoil experiments
3. Optimize detector sensitivity and background rejection
4. Analyze potential signatures and detection strategies
5. Evaluate feasibility and resource requirements

[REASONING FRAMEWORK]
Apply creative and systematic approach to dark matter detection:
1. Consider multiple dark matter candidate models
2. Explore novel detection mechanisms and materials
3. Optimize experimental design for sensitivity
4. Address background rejection and false positive prevention
5. Evaluate technical feasibility and resource requirements

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Novel detection method with detailed implementation
- Expected outcomes: Predicted sensitivity and discovery potential
- Resource requirements: Materials, equipment, and computational resources
- Safety considerations: Radiation safety and environmental impact
- Validation approach: Method for verifying detection claims
- Failure analysis: Sources of false signals and mitigation strategies

[SAFETY CONSTRAINTS]
- Ensure radiation safety in detector design
- Consider environmental impact of experimental materials
- Document assumptions about dark matter properties
- Plan for verification of potential detection claims
""",
        "expected_concepts": ["dark matter", "particle physics", "detector physics", "beyond standard model"],
        "difficulty": "expert"
    },
    {
        "name": "Fusion Reactor Magnetic Confinement Innovation",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent developing innovative approaches to nuclear fusion energy.

[CONTEXT]
Experiment objective: Propose and simulate a novel magnetic confinement configuration for nuclear fusion
Related knowledge: plasma physics, magnetic confinement, fusion energy, MHD instabilities
Available resources: Plasma simulation software and computational resources
Safety protocols: Follow standard fusion research safety protocols

[TASK INSTRUCTIONS]
Design an innovative fusion reactor confinement system:
1. Propose a novel magnetic confinement configuration that could solve plasma instability problems
2. Simulate the proposed design with magnetohydrodynamic models
3. Analyze plasma stability and confinement properties
4. Optimize design parameters for energy production
5. Evaluate engineering feasibility and safety

[REASONING FRAMEWORK]
Apply innovative engineering and physics principles:
1. Identify key limitations of existing confinement approaches
2. Propose novel solutions to plasma instability problems
3. Model plasma behavior in the proposed configuration
4. Optimize for confinement time and energy output
5. Address engineering and safety considerations

[OUTPUT REQUIREMENTS]
Provide a complete experimental design with:
- Experiment design: Novel confinement configuration with simulation details
- Expected outcomes: Predicted plasma performance and stability
- Resource requirements: Computational and engineering resources
- Safety considerations: Radiation safety and accident scenarios
- Validation approach: Method for verifying confinement predictions
- Failure analysis: Instability modes and mitigation strategies

[SAFETY CONSTRAINTS]
- Ensure radiation safety in reactor design
- Consider accident scenarios and safety systems
- Document assumptions about plasma behavior
- Plan for experimental validation of theoretical predictions
""",
        "expected_concepts": ["plasma physics", "magnetic confinement", "fusion energy", "MHD instabilities"],
        "difficulty": "expert"
    },
    {
        "name": "Quantum Entanglement Communication Limits",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent investigating the fundamental limits of quantum information theory.

[CONTEXT]
Experiment objective: Investigate the fundamental limits of quantum communication using entangled particles
Related knowledge: quantum entanglement, Bell's theorem, no-communication theorem, quantum information
Available resources: Quantum information simulation environment
Safety protocols: Follow standard quantum information safety protocols

[TASK INSTRUCTIONS]
Investigate quantum communication fundamental limits:
1. Design experiments to test whether information can be transmitted faster than light through quantum entanglement
2. Explore the no-communication theorem and its implications
3. Analyze the relationship between entanglement and information transfer
4. Investigate practical quantum communication protocols
5. Evaluate fundamental versus practical limitations

[REASONING FRAMEWORK]
Apply rigorous quantum information theory principles:
1. Ground analysis in fundamental quantum mechanics principles
2. Distinguish between correlation and communication
3. Analyze information-theoretic limits of quantum systems
4. Connect theoretical insights to practical implementations
5. Address common misconceptions about quantum entanglement

[OUTPUT REQUIREMENTS]
Provide a complete analysis with:
- Theoretical analysis: Rigorous examination of quantum communication limits
- Experimental design: Proposed tests of fundamental principles
- Expected outcomes: Predicted results and theoretical implications
- Resource requirements: Computational resources needed
- Safety considerations: Conceptual clarity and scientific rigor
- Validation approach: Method for verifying theoretical conclusions

[SAFETY CONSTRAINTS]
- Maintain scientific rigor in quantum mechanical analysis
- Avoid common misconceptions about quantum entanglement
- Document the distinction between correlation and communication
- Plan for clear communication of theoretical results
""",
        "expected_concepts": ["quantum entanglement", "Bell's theorem", "no-communication theorem", "quantum information"],
        "difficulty": "advanced"
    },
    {
        "name": "Extreme Time Dilation Scenarios",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent exploring the implications of relativistic physics for space travel and communication.

[CONTEXT]
Experiment objective: Calculate and simulate time dilation effects in extreme scenarios
Related knowledge: special relativity, general relativity, time dilation, spacetime geometry
Available resources: Relativistic physics simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Analyze extreme time dilation scenarios:
1. Calculate time dilation effects near black hole event horizons
2. Simulate relativistic motion close to the speed of light
3. Model time dilation in strong gravitational fields
4. Explore practical implications for space travel and communication
5. Analyze the relationship between different types of time dilation

[REASONING FRAMEWORK]
Apply systematic approach to relativistic physics analysis:
1. Ground calculations in Einstein's field equations
2. Distinguish between special and general relativistic effects
3. Connect theoretical predictions to practical applications
4. Analyze limiting cases and extreme scenarios
5. Validate results against known solutions

[OUTPUT REQUIREMENTS]
Provide a complete analysis with:
- Theoretical analysis: Detailed calculations of time dilation effects
- Simulation design: Computational models for extreme scenarios
- Expected outcomes: Predicted time dilation factors and effects
- Resource requirements: Computational resources needed
- Safety considerations: Numerical accuracy and stability
- Validation approach: Method for verifying relativistic calculations

[SAFETY CONSTRAINTS]
- Ensure numerical accuracy in relativistic calculations
- Validate against known analytical solutions
- Document approximations and limitations
- Plan for verification of extreme scenario predictions
""",
        "expected_concepts": ["special relativity", "general relativity", "time dilation", "spacetime geometry"],
        "difficulty": "intermediate"
    },
    {
        "name": "Casimir Effect Energy Harvesting",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent investigating novel approaches to energy extraction from quantum vacuum fluctuations.

[CONTEXT]
Experiment objective: Design an experiment to measure and potentially harness the Casimir effect for energy generation
Related knowledge: quantum field theory, vacuum energy, Casimir effect, zero-point energy
Available resources: Computational electromagnetism simulation environment
Safety protocols: Follow standard experimental physics safety protocols

[TASK INSTRUCTIONS]
Investigate Casimir effect energy harvesting:
1. Design an experiment to measure the Casimir effect between conducting plates
2. Investigate whether quantum vacuum energy can be practically utilized
3. Analyze the thermodynamic constraints and limitations
4. Evaluate the feasibility of energy extraction mechanisms
5. Consider alternative approaches to vacuum energy utilization

[REASONING FRAMEWORK]
Apply rigorous thermodynamic and quantum field theory principles:
1. Ground analysis in quantum field theory fundamentals
2. Analyze thermodynamic constraints and conservation laws
3. Distinguish between measurement and extraction of vacuum energy
4. Evaluate feasibility against established physical principles
5. Consider alternative interpretations and approaches

[OUTPUT REQUIREMENTS]
Provide a complete analysis with:
- Theoretical analysis: Rigorous examination of Casimir effect physics
- Experimental design: Proposed measurement techniques
- Expected outcomes: Predicted forces and energy scales
- Resource requirements: Experimental equipment and resources
- Safety considerations: Scientific rigor and theoretical consistency
- Validation approach: Method for verifying theoretical predictions

[SAFETY CONSTRAINTS]
- Maintain scientific rigor in quantum field theory analysis
- Respect thermodynamic and conservation principles
- Document the distinction between measurement and energy extraction
- Plan for clear communication of theoretical limitations
""",
        "expected_concepts": ["quantum field theory", "vacuum energy", "Casimir effect", "zero-point energy"],
        "difficulty": "expert"
    },
    {
        "name": "Metamaterial Cloaking Device Physics",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent designing electromagnetic cloaking devices using advanced metamaterials.

[CONTEXT]
Experiment objective: Simulate the physics of electromagnetic cloaking using metamaterials with negative refractive index
Related knowledge: metamaterials, negative refraction, electromagnetic cloaking, transformation optics
Available resources: Electromagnetic simulation software
Safety protocols: Follow standard electromagnetic safety protocols

[TASK INSTRUCTIONS]
Design electromagnetic cloaking devices:
1. Simulate the physics of electromagnetic cloaking using metamaterials
2. Design a practical invisibility cloak and test its effectiveness
3. Analyze performance across different wavelengths and polarizations
4. Investigate limitations and practical constraints
5. Evaluate engineering feasibility and applications

[REASONING FRAMEWORK]
Apply transformation optics and electromagnetic theory principles:
1. Ground design in transformation optics principles
2. Design metamaterial structures with required properties
3. Simulate electromagnetic wave interactions with the cloak
4. Analyze performance limitations and practical constraints
5. Connect theoretical designs to engineering implementations

[OUTPUT REQUIREMENTS]
Provide a complete design with:
- Theoretical design: Transformation optics based cloak design
- Simulation approach: Electromagnetic modeling details
- Expected outcomes: Predicted cloaking performance and limitations
- Resource requirements: Computational and fabrication resources
- Safety considerations: Electromagnetic safety and environmental impact
- Validation approach: Method for verifying cloaking performance

[SAFETY CONSTRAINTS]
- Ensure electromagnetic safety in device design
- Consider environmental impact of metamaterial fabrication
- Document assumptions about material properties
- Plan for experimental validation of theoretical designs
""",
        "expected_concepts": ["metamaterials", "negative refraction", "electromagnetic cloaking", "transformation optics"],
        "difficulty": "advanced"
    },
    {
        "name": "Wormhole Traversability Analysis",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent investigating the theoretical possibility of traversable spacetime wormholes.

[CONTEXT]
Experiment objective: Analyze the theoretical possibility of traversable wormholes using general relativity
Related knowledge: general relativity, wormholes, exotic matter, spacetime topology
Available resources: General relativity simulation environment
Safety protocols: Follow standard computational physics safety protocols

[TASK INSTRUCTIONS]
Analyze traversable wormhole physics:
1. Using general relativity, analyze the requirements for traversable wormholes
2. Calculate the exotic matter requirements for stabilization
3. Investigate whether quantum effects could stabilize such structures
4. Analyze the relationship between geometry and matter requirements
5. Evaluate the physical feasibility of traversable wormholes

[REASONING FRAMEWORK]
Apply rigorous general relativity and quantum field theory principles:
1. Ground analysis in Einstein's field equations
2. Analyze the relationship between spacetime geometry and stress-energy
3. Investigate quantum effects in curved spacetime
4. Connect theoretical insights to physical feasibility
5. Address conceptual and mathematical challenges

[OUTPUT REQUIREMENTS]
Provide a complete analysis with:
- Theoretical analysis: Detailed examination of wormhole physics
- Mathematical framework: Einstein field equations and stress-energy requirements
- Expected outcomes: Predicted matter requirements and stability conditions
- Resource requirements: Computational resources needed
- Safety considerations: Scientific rigor and theoretical consistency
- Validation approach: Method for verifying mathematical results

[SAFETY CONSTRAINTS]
- Maintain scientific rigor in general relativity analysis
- Respect energy conditions and physical principles
- Document assumptions about exotic matter properties
- Plan for clear communication of theoretical limitations
""",
        "expected_concepts": ["general relativity", "wormholes", "exotic matter", "spacetime topology"],
        "difficulty": "expert"
    },
    {
        "name": "Quantum Computing Error Correction Innovation",
        "prompt": """[ROLE DEFINITION]
You are {agent_name}, a scientific AI agent developing novel approaches to quantum error correction for fault-tolerant computing.

[CONTEXT]
Experiment objective: Develop a novel quantum error correction scheme for fault-tolerant quantum computing
Related knowledge: quantum computing, error correction, quantum noise, fault tolerance
Available resources: Quantum computing simulation environment
Safety protocols: Follow standard quantum computing safety protocols

[TASK INSTRUCTIONS]
Design innovative quantum error correction:
1. Develop a novel quantum error correction scheme that could reduce fault-tolerant computing overhead
2. Simulate its performance against different types of quantum noise
3. Analyze the relationship between code distance and error rates
4. Evaluate resource requirements and scalability
5. Compare with existing error correction approaches

[REASONING FRAMEWORK]
Apply quantum information theory and coding theory principles:
1. Ground design in quantum error correction fundamentals
2. Analyze performance against various noise models
3. Optimize for resource efficiency and error correction capability
4. Connect theoretical designs to practical implementations
5. Address scalability and fault tolerance requirements

[OUTPUT REQUIREMENTS]
Provide a complete design with:
- Theoretical design: Novel error correction code structure
- Simulation approach: Noise model and performance analysis
- Expected outcomes: Predicted error correction performance and resource requirements
- Resource requirements: Quantum and classical computational resources
- Safety considerations: Quantum information integrity and protocol correctness
- Validation approach: Method for verifying error correction performance

[SAFETY CONSTRAINTS]
- Ensure correctness of quantum information protocols
- Respect fundamental limits of quantum error correction
- Document assumptions about noise models and hardware
- Plan for experimental validation of theoretical designs
""",
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