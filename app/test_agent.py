
from chatboard.prompt.base_prompt import prompt
from chatboard.llms.openai_llm2 import OpenAiLLM
from app.test_models import Message, Context
from chatboard.prompt.legacy.context import BlockStream
from chatboard.prompt.depends import Depends
from chatboard.prompt.legacy.block2 import block as b



@prompt()
async def chat_prompt(conv: BlockStream, message: Message, llm: OpenAiLLM = Depends(OpenAiLLM)):
    
    with b("""
    you should act like a pirate.       
    your name is Jack Sparrow.
    """, role="system") as sm:
        with b.title("Task"):
            b("your task is to learn about the user.")
        
    # with b.title("User Message", role="user") as um:
    #     b(message.content)
    
    res = await llm([sm] + conv).generate()
    response = await conv.push(res)
    return response




async def run_agent(ctx: Context, message: Message):
    with b("", role="user") as um:
        b(message.content)
    conv = await ctx.last(4)
    message = await conv.push(um)    
    response = await chat_prompt(conv=conv, message=message)
    return response, message



