this repo is a framework for building agents.
it give basic building blocks for developers to build reliable agents.


# Core principles

1. versioning and event sourcing data - all agent actions should be loged saved and be revertible.
2. streaming architecture - streaming is first class citizen in the framework.
3. blocks - smallest composable and reusable unit of a text prompt, separating content from style.




# Components

## Model ORM
@chatboard/model - ORM system for the agent framework.
it should support Postgres, Qdrant and Neo4j with a single API. 

the base components of the ORM are 3 types of Models:
1. Model - regular model just like in every ORM.
2. VersionedModel - model that is versioned by Turns, Branches and Artifact for context switching.
3. ArtifactModel - event sourced model that saves every change so it could be revertable.

## Blocks
@chatboard/block 
blocks help build reusable components for prompt building.
they separate the style from the content



## FBP flow components
@chatboard/prompt
give ability to create application logic in an imerative way but support streaming natively with Flow Based Programming Pronciples






# Tasks
those are the tasks that the User may ask you to do and you should know how to do them.
## User Playground Script
when you are asked to create a playground script for a feature or a component you should 
create a jupyter notebookd file (.ipynb) inside the root folder @research directory and write the evaluations and test there.
## commands
to run the notebook use the following bash command:

`source .venv/bin/activate && set -a && source .env && set +a && PYTHONPATH=$(pwd) jupyter nbconvert --to notebook --execute [path to notebook] --stdout`

* make sure to activate virtual environment
* load .env file
* add the root directory to the python path
* run the notebook


### Rules
1. there is no need to wrap the tests in functions, just write the logic plainly in the notebook cells.
2. jupyter notebook .ipynb file is just a json file, you can write directly into it.
3. the structuring of the playground should match the tasks that you have to build the feature.
4. you should split the playground with markdown cells where you write in Markdown format the description of the tests
5. in this file you can see the output of the cell execution

look at @research/example_playground.ipynb for an example how this should be formated


## Unit / Integration tests building
when asked to create unit or integration tests you should create them at @__tests__ folder at the root.
the user may ask you to convert the playground script into test file.
use pytest package to create those tests.
for async tests don't forget to add the `@pytest.mark.asyncio` decorator for the test function.

# environment 
at the root folder there is a .venv folder that contains python 12 environment.