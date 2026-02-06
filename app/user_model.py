

from chatboard.conversation.alchemy_models import BaseUserModel, Base
from chatboard.conversation.models import User, engine
from sqlalchemy import UUID, create_engine, Column, Integer, String, JSON, ForeignKey, DateTime, Text, Boolean, inspect
import os




class UserProfileModel(BaseUserModel):
    __mapper_args__ = {
        # "polymorphic_identity": "app_a_user",
        "polymorphic_identity": "user",
    }
    
    # Add your custom columns here:
    name = Column(String)
    language = Column(String, nullable=True)
    phone_number = Column(String)
    manager_phone_number = Column(String)
    send_notifications = Column(Boolean)
    auto_respond = Column(Boolean)
    unread_messages = Column(Integer, default=0)
    is_test = Column(Boolean, default=False)
    last_notification_sent_time = Column(DateTime(timezone=True), nullable=True)
    last_message_received_time = Column(DateTime(timezone=True), nullable=True)
    
    
    
    
class Client(User):
    name: str
    phone_number: str
    email: str
    
    
    
    
    
    
    
async def instantiate_all_models():    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)