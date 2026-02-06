import asyncio
from contextlib import asynccontextmanager
import json
from typing import Annotated
from fastapi import Depends, FastAPI, Form, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from chatboard.app import Chatboard
from app.test_models import TestContext, Manager, Message
from app.test_agent import chat_prompt, run_agent






bot = Chatboard(
        message_model=Message, 
        user_model=Manager,
    )



@bot.entrypoint("/api/chat")
async def chat(ctx: TestContext, message: Message):
    response, message = await run_agent(ctx=ctx, message=message)
    return response
    
    

    
app = bot.get_app()
    
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
