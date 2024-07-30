from setuptools import setup, find_namespace_packages

setup(
    name="chatboard",
    version="0.1",
    author='Pavel Schudel',
    author_email='pavel1860@gmail.com',
    url='https://github.com/pavel1860/chatboard',
    description="A modular chatboard package",
    # packages=find_namespace_packages(include=['chatboard*', 'chatboard.*', 'text']),
    packages=find_namespace_packages(include=['chatboard*', 'chatboard.*', 'text', 'media', 'scrape']),
    install_requires=[
        # list common dependencies here
    ],
    extras_require={
        'text': [
            "numpy==1.26.4",
            "langchain==0.1.9",
            "langchain-openai==0.0.5",
            # "pydantic==1.10.4",
            "pydantic>=1.10.4, <3",
            "tiktoken==0.5.2",
            "pinecone-client==3.0.1",
            "pinecone-text==0.9.0",
            "scipy==1.11.4",
            "boto3==1.24.47",
            "openai==1.37.1",
            "langdetect==1.0.9",
            "GitPython==3.1.31",
            "qdrant-client==1.8.2",
            "Jinja2==3.1.3",
            "docstring_parser==0.16"
        ],
        'media': [
            "starlette==0.24.0",
            # "torch==2.2.0",
            "xformers==0.0.24",
            "transformers==4.38.1",
            "diffusers==0.26.3",
            "numpy==1.26.4",
            "boto3==1.24.47",
            "gunicorn==20.1.0",
            "python-dotenv==0.20.0",
            "opencv-python==4.7.0.68",
            "pydantic==1.10.4",
            "mediapipe==0.9.1.0",
            "soundfile==0.12.1",
        ],
        'scrape': []
    },
    classifiers=[
        # Classifiers help users find your project by categorizing it.
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

