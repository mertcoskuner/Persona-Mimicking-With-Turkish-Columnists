# Agents Module

This module contains the implementation of LLM agents that simulate Turkish columnists' personas.

## Structure

- `base_agent.py`: Base class for all agents
- `columnist_agent.py`: Implementation of columnist-specific agents
- `agent_factory.py`: Factory for creating different types of agents

## Usage

```python
from agents.agent_factory import create_agent

# Create a new agent
agent = create_agent(
    name="Ahmet",
    persona="Conservative columnist",
    model="meta-llama"
)

# Interact with the agent
response = agent.generate_response("What do you think about current economic policies?")
```

## Adding New Agents

1. Create a new agent class that inherits from `BaseAgent`
2. Implement required methods:
   - `generate_response()`
   - `update_context()`
   - `get_persona()`
3. Register the new agent in `agent_factory.py`

## Testing

Run agent-specific tests:
```bash
python -m pytest tests/agents/
``` 