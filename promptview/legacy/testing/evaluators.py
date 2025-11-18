from ..prompt import prompt, Depends
from ..block import BlockChunk
from ..llms import OpenAiLLM, LLM
from pydantic import BaseModel, Field
from ..llms.llm import OutputModel



class EvalResult(OutputModel):
    reasoning: str = Field(..., description="step by step reasoning about the response, differences between expected and actual output and is it confirms with the requirements")
    score: int = Field(default=-1, description="score between 0 and 10")




@prompt()
async def prompt_score_evaluator(    
    request: BlockChunk | str, 
    response: BlockChunk | str,
    expected: list[BlockChunk] | list[str] | BlockChunk | str,
    description: BlockChunk | str,
    config: dict,
    llm: LLM = Depends(LLM)
    ):
    
    if response is None:
        raise ValueError("Response is not set")
    if request is None:
        raise ValueError("Message is not set")
    if expected is None:
        raise ValueError("Expected is not set")
    
    with BlockChunk(role="system") as sys:
        sys([
            "You are an expert evaluator tasked with assessing the response of an AI assistant to a user's query.",
            "Your purpose is to evaluate responses and act as 'LLM as a judge'.",
        ])
        with sys("Task"):
            sys([
                "the user will send an output of for you to evaluate. you should evaluate it."
                "you should evaluate the response of an agent based on the expected output.",
                "you should return a score between 0 and 10."
            ])
        with sys("Rules", style=["list"]):
            sys([
                "remember that you are an evaluator, that is the only purpose of your output.",
            ])

    with BlockChunk(role="user") as usr:
        with usr("Test Case"):
            usr(description)
            with usr("Input"):
                usr /= request
            with usr("Response"):
                usr /= response
            with usr("Expected"):
                if isinstance(expected, list):
                    usr /= "can be one of the following:"
                    with usr(style=["list"]):
                        for exp in expected:
                            usr /= exp
                else:
                    usr /= "the output should be:"
                    usr /= expected
                
    res = await llm(sys, usr).complete(EvalResult)    
    return res
