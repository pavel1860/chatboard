from setuptools import setup, find_packages, find_namespace_packages

setup(
    name="promptview",
    version="0.1.0",
    author='Pavel Schudel',
    author_email='pavel1860@gmail.com',
    url='https://github.com/pavel1860/promptview',
    description="A framework for generating prompts for LLMs",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pydantic>=2.8.2, <3",
        "jsonref==1.1.0",
        "tiktoken==0.5.2",
        "pinecone-text==0.9.0",
        # "scipy==1.11.4",
        # "boto3==1.24.47",
        "openai==1.56.2",
        "neo4j==5.28.1",
        "anthropic==0.39.0",
        "langdetect==1.0.9",
        "qdrant-client==1.10.1",
        "docstring_parser==0.16",
        "iso-639==0.4.5",
        "aiohttp==3.10.0",
        "langsmith==0.1.95",
        "asyncpg==0.30.0",
        "tabulate==0.9.0",
        "lxml==6.0.0",
        "PyYAML==6.0.2",
        "PyJWT==2.10.1",
        "google-auth[requests]==2.40.1",
        "google==3.0.0",
        "authlib==1.5.2",
        "pycryptodome==3.23.0"
    ],    
    classifiers=[
        # Classifiers help users find your project by categorizing it.
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    keywords="prompt, llm, generation",
)

