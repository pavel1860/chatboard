# chatboard Documentation

## 1. Introduction

**chatboard** is a modular, extensible framework for building, managing, and interacting with prompt-based AI systems. It is designed to help developers create, test, and deploy complex prompt workflows, manage conversational agents, and integrate with various language models and vector stores.

Key features include:
- A flexible Block system for composing and rendering prompt components.
- prompt decorator for building extendable, structured prompts with dependencies injection
- Agent architecture for orchestrating actions and managing conversations.
- Support for multiple model backends (OpenAI, Anthropic, custom models, etc.).
- Integration with vector databases (Qdrant, Neo4j, Postgres) for advanced retrieval and context management.
- Extensible API layer for building custom endpoints and integrations.
- Comprehensive testing and evaluation tools for prompt and agent workflows.

chatboard aims to accelerate the development of robust, production-ready AI applications by providing reusable abstractions, clear architectural patterns, and developer-friendly tooling.

---

## 2. Getting Started

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) (recommended) or `pip`
- (Optional) Docker, if you want to run services in containers

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/chatboard.git
   cd chatboard
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```
   Or, using pip:
   ```bash
   pip install -r services/socket_manager/requirements.txt
   ```

3. **(Optional) Set up environment variables:**
   - Copy `.env.example` to `.env` and adjust as needed.

### Basic Usage Example

To run the main application:
```bash
poetry run python -m chatboard.app.main
```
Or, if using pip:
```bash
python -m chatboard.app.main
```

To run tests:
```bash
poetry run pytest
```
or
```bash
pytest
```

---

## 3. Architecture Overview

chatboard is organized as a modular Python application, designed for extensibility and clarity. The architecture is centered around several core components:

- **Block System:** Provides composable building blocks for prompt construction, rendering, and serialization.
- **Agent System:** Manages conversational agents, orchestrates actions, and routes messages.
- **Model Layer:** Handles data models, vector storage, and integration with databases (e.g., Qdrant, Neo4j, Postgres).
- **API Layer:** Exposes RESTful endpoints for interacting with models, sessions, users, and artifacts.
- **LLM Integration:** Supports multiple language model backends (OpenAI, Anthropic, Azure, etc.) via a unified interface.
- **Testing & Evaluation:** Includes tools and frameworks for testing blocks, agents, and workflows.

### Component Relationships

- The **Agent System** uses the **Block System** to construct and process prompts.
- The **Model Layer** provides persistent storage and retrieval for artifacts, sessions, and vector data.
- The **API Layer** enables external access to core functionality and data.
- **LLM Integration** is abstracted so that agents and blocks can work with any supported model backend.

---

## 4. Core Concepts

### Block

Blocks are the fundamental building units in chatboard. They represent discrete components of a prompt or workflow, such as text, variables, or logic. Blocks can be composed, rendered, and serialized, enabling flexible construction of complex prompts and workflows.

- **Purpose:** Encapsulate logic, content, or formatting for a part of a prompt.
- **Usage:** Combine multiple blocks to build dynamic prompts or conversational flows.
- **Key files:** `chatboard/block/block.py`, `chatboard/block/block_renderer.py`

**Example:** Creating and rendering a nested block structure.

```python
from chatboard.block.block import Block


with Block(tags=["system"]) as b:
    b /= "you are a helpful assistant"
    with b("Task", style="md", tags=["task"]):
        b /= "this is task you need to complete"

assert b.render() == "you are a helpful assistant\n# Task\nthis is task you need to complete"
```

### Advanced Example: The block Decorator

The `block` decorator allows you to encapsulate block-building logic in reusable functions, and enables context management for nested blocks. This is especially useful for building extendable complex, structured prompts.

```python
from chatboard.block.block import block, Block

@block()
def create_table_block(blk: Block, name: str, *fields):
    with blk(f"CREATE TABLE IF NOT EXISTS {name}", style="func-col") as blk:
        for field in fields:
            blk /= f'"{field.name}"', field.sql_type
            if field.is_primary_key:
                blk += "PRIMARY KEY"
            else:
                blk += "NULL" if field.is_optional else "NOT NULL"
```

if the block is extendable, for instance if you want a reusable assistant profile, but with different rules 
each time

```python
# Usage:
@block()
def assistant_profile(blk: Block, query: str):
    with Block(tags=["system"]) as b:
        b /= "you are a helpful assistant"
        with b("Task", style="md", tags=["task"]):
            b /= "this is task you need to complete"

        with b("Rules", style="list", tags["rules"]):
            yield b 



with assistant_profile("tell me a story") as b:
    b /= "you should speak like a pirate"
    b /= "you are on a ship looking for a treasure"


with assistant_profile("tell me a story") as b:
    b /= "you should speak like shakespeare"
    b /= "you should right 15th centry poems"

```

This pattern ensures that nested blocks and content are appended to the correct parent, making your block logic modular and maintainable.

### Prompt Decorator

The `prompt` decorator allows you to build extendable, structured prompts with dependencies injection.

```python
from chatboard.block.block import block, Block
from chatboard.llms import LLM, OpenAiLLM
from chatboard.prompt import prompt, agent, Depends
from pydantic import BaseModel, Field




@prompt()
async def pirate_prompt(
    request: Block, 
    llm: LLM = Depends(LLM)
    ):
    with Block(role="system") as sys:
        sys([
            "you are a pirate by the name of jack black",            
        ])
        with sys("Task", style="md"):
            sys([
                "you need to talk to the player as a pirate"
            ])
        
    with Block(role="user", style="md") as u:
        u /= request
    
    response = await llm(sys, u)
    return response




request = Block("Hello, how are you?")

response = await pirate_prompt(request)
response.print()
```


### Agent

Agents are responsible for managing conversations, orchestrating actions, and interacting with language models. They use blocks to construct prompts and handle the flow of information between users, models, and the system.

- **Purpose:** Manage the lifecycle of a conversation or task, including prompt generation and response handling.
- **Usage:** Implement custom agents to define specific behaviors or workflows.
- **Key files:** `chatboard/agent/action_agent.py`, `chatboard/agent/function_agent.py`


### Model

The model layer provides abstractions for data storage, retrieval, and manipulation. It supports integration with various backends, such as vector databases (Qdrant, Neo4j, Postgres), and manages artifacts, sessions, and other persistent data.

- **Purpose:** Store and retrieve data required for prompt workflows and agent operations.
- **Usage:** Define and interact with models for artifacts, sessions, users, etc.
- **Key files:** `chatboard/model/`, `chatboard/model/neo4j/`, `chatboard/model/postgres/`, `chatboard/model/qdrant/`

**Example:** Defining and using a model with relations.

```python
from chatboard.model import Model, ModelField, KeyField, RelationField, Relation

class User(Model):
    id: int = KeyField(primary_key=True)
    name: str = ModelField()
    age: int = ModelField()
    posts: Relation["Post"] = RelationField(foreign_key="user_id")

class Post(Model):
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    content: str = ModelField()
    user_id: int = ModelField(foreign_key=True)
```

---

## 5. Component Deep Dives
For each major component, provide:
- **Purpose**
- **How it works**
- **Key classes/functions**
- **Example usage**

### 5.1 Block System
- Description and usage
- Key files/classes

### 5.2 Agent System
- Description and usage
- Key files/classes

### 5.3 Model Layer
- Description and usage
- Key files/classes

### 5.4 API Layer
- Description and usage
- Key files/classes

### 5.5 Other Components
- Add more as needed (e.g., Vectorizers, Auth, etc.)

---

## 6. Testing
- How to run tests
- Testing philosophy and structure

---

## 7. Extending and Contributing
- How to add new features or components
- Coding standards and guidelines
- Contribution process

---

## 8. FAQ / Troubleshooting
- Common issues and solutions

---

## 9. References & Further Reading
- Links to related documentation, papers, or resources
