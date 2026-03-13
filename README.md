Multi-Agent Predator-Prey
  
Core concept: Predators learn to coordinate hunts, prey learn evasion strategies. Both improve over time through co-evolution / self-play.

Key Design Decisions

  1. Environment
  - 2D arena 
  - Continuous 
  - (planned) Obstacles/terrain

  2. Agent Design
  - 10s. (planned) 100s of agents
  - Observation: local vision cone. (planned) Communication between teammate
  - Acceleration/steering.
  - Shared policy. (planned) subgroups 

  3. Training
  - JAX-based for massive parallelism (thousands of arenas simultaneously)
  - PPO with parameter sharing per team is the standard starting point
  - Co-training: both teams improve against each other

  4. What Makes It Wow
  - Emergent formation hunting (pincer moves, herding)
  - Prey developing schooling/flocking behavior defensively
  - Clear generational improvement visible in replays
  - Live dashboard showing population dynamics, reward curves

  Proposed Starting Point

  - 2D continuous arena, top-down
  - 3-5 predators vs 10-20 prey
  - Local observations (vision radius), continuous actions (velocity)
  - Shared policy per team, PPO
  - JAX/Brax or custom JAX env for parallelism

