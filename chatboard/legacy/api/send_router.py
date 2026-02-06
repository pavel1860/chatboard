from fastapi import APIRouter, HTTPException, Depends
from typing import Type, Callable, Optional
import logging

logger = logging.getLogger(__name__)

def create_message_router(
    
) -> APIRouter:
    router = APIRouter()
    
    @router.post("/{identifier}/chat")
    async def handle_message(
        identifier: str,
        payload: message_model,
        *args,
        **kwargs
    ):
        if not payload.content:
            raise HTTPException(status_code=400, detail="Message content cannot be empty")
            
        # Handle message deletion if from_message_id is provided
        if payload.from_message_id and router_config.delete_messages:
            await router_config.delete_messages(
                identifier=identifier,
                from_message_id=payload.from_message_id,
                session_id=payload.session_id
            )
        
        # Get or create context
        context = await router_config.get_context(
            identifier=identifier,
            message=payload.content,
            session_id=payload.session_id,
            metadata=payload.metadata
        )
        
        if not context:
            raise HTTPException(status_code=404, detail="Could not create or retrieve context")
            
        # Run agent and collect messages
        messages = []
        try:
            async for bot_message in router_config.run_agent(
                context=context,
                message=payload.content,
                **kwargs
            ):
                if bot_message is not None:
                    logger.info(f"Bot message: {bot_message}")
                    messages.append(bot_message)
                else:
                    logger.info("Agent finished")
                    
            # Get final messages
            responses = await router_config.get_messages(
                context=context,
                limit=len(messages) + 1
            )
            
            return responses
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router