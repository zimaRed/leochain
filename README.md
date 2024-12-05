leochain is a simple framework for building multi-agent systems using GPT.

it consists of a few parts:
- a supervisor, which manages the conversation and the agents
- agents, which are responsible for specific tasks and can use specific tools
- tools, which are used by agents to perform specific tasks
- an ensemble that runs the agents and the supervisor in a loop

you can use this framework by instantiating an ensemble, then adding supervisor/agents/tools, and then running the ensemble in a loop.

this system is fully composable-- ensembles can be stacked and nested to create complex systems of agent teams.
(a tool can actually call an ensemble to run a set of agents, and then return the result)

limitations: only works with openai models, agents can only call one tool at a time, agents can't write/read files... many more limitations
