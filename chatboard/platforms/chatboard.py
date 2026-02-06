




import json
import os
from aiohttp_retry import Any
from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel
from faststream.rabbit import RabbitExchange, RabbitQueue, RabbitBroker
from faststream.rabbit.fastapi import Logger, RabbitRouter



class SocketManager:
    
    
    def __init__(self) -> None:
        self._active_connections = {}    
    
    
    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active_connections[user_id] = websocket
        
        
    async def disconnect(self, user_id: str) -> None:
        socket = self._active_connections.pop(user_id)
        await socket.close()
        
    async def send_message(self, user_id: str, message: dict) -> None:
        socket = self._active_connections.get(user_id)
        if socket:
            await socket.send_text(json.dumps(message))
            
            
socket_manager = SocketManager()      



router = APIRouter(
    prefix="/subscribe",
    tags=["subscribe"],
    # dependencies=[Depends(get_current_user), ]
)


def get_user_id(websocket: WebSocket):
    user_id = websocket.query_params.get("user_id")
    if not user_id:
        raise ValueError("User ID not provided")
    return user_id


@router.websocket("/socket")
async def chat_endpoint(websocket: WebSocket, user_id: str = Depends(get_user_id)):

    await socket_manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # await socket_manager.send_message(user_id, data)
    except Exception as e:
        print(e)
    finally:
        await socket_manager.disconnect(user_id)


# message_queue = RabbitQueue("message_queue")
msg_router = RabbitRouter(url=os.getenv('RABBITMQ_URL'), max_consumers=5)



class Message(BaseModel):
    user_id: str    
    content: Any  

@msg_router.subscriber("message")
async def message_handler(message: Message):
    print("Received message", message)
    await socket_manager.send_message(message.user_id, message.content)


          
            
class ChatboardClient:
    
    def __init__(self) -> None:
        self._broker_started = False
        
    async def start_broker(self) -> None:
        if not self._broker_started:
            await msg_router.broker.connect()
            self._broker_started = True
    
    async def send_text_message(self, user_id: str, message: BaseModel) -> None:
        # if not self._broker_started:
        #     await self.start_broker()
        
        async with RabbitBroker(url=os.getenv('RABBITMQ_URL')) as br:
            msg = Message(user_id=user_id, content=message)
            return await br.publish(msg, "message")
        # await socket_manager.send_message(user_id, message.model_dump_json())