from pydantic import BaseModel, Field
import datetime as dt

from promptview.conversation.models import Branch, Message, MessageBackend, Session, Turn
from promptview.conversation.protocols import BaseProto







        
    

class MessageLogError(Exception):
    pass

class UserManager:
    def __init__(self, backend):
        from promptview.conversation.models import UserBackend, MessageBackend
        self._backend = backend
        self._message_backend = MessageBackend()
    
    async def create_user(self, **user_data):
        """Create a new user with the given data"""
        return await self._backend.create_user(**user_data)
    
    async def get_user(self, user_id: int):
        """Get a user by ID"""
        return await self._backend.get_user(user_id=user_id)
    
    async def list_users(self, limit: int = 10, offset: int = 0):
        """List users with pagination"""
        return await self._backend.list_users(limit=limit, offset=offset)
    
    async def add_session(self, user_id: int):
        """Create a new session for a user"""
        session = await self._backend.add_session(user_id)
        branch = await self._message_backend.add_branch(Branch(session_id=session.id, branch_order=0, message_counter=0))
        return session
    
    async def list_user_sessions(self, user_id: int, limit: int = 10, offset: int = 0):
        """List sessions for a specific user"""
        return await self._backend.list_user_sessions(user_id=user_id, limit=limit, offset=offset)
    
    async def get_user_messages(self, user_id: int, limit: int = 10, offset: int = 0):
        """Get all messages from all sessions for a user"""
        return await self._backend.get_user_messages(user_id=user_id, limit=limit, offset=offset)

class SessionManager:
    
    def __init__(self):
        self._backend = MessageBackend()
        
        
    async def get_session(self, session_id: int | None = None):
        if session_id is None:  
            raise MessageLogError("No session_id provided")
        return await self._backend.get_session(id=session_id)
        
    async def get_last_user_session(self, user_id: int):
        return await self._backend.last_session(user_id=user_id)
    
    async def list_sessions(self, user_id: int | None = None, limit: int = 10, offset: int = 0):
        return await self._backend.list_sessions(user_id=user_id, limit=limit, offset=offset)
    
    async def create_session(self, user_id: int):
        session = await self._backend.add_session(Session(user_id=user_id))
        branch = await self._backend.add_branch(Branch(session_id=session.id, branch_order=0, message_counter=0))
        return session
    
        
    
    
    
    
class Head:
    
    def __init__(self):
        self._branch: Branch | None = None
        self._turn: Turn | None = None
        
    @property
    def branch(self) -> Branch:
        if self._branch is None:
            raise MessageLogError("Branch is not initialized")
        return self._branch
    
    @branch.setter
    def branch(self, value: Branch):
        self._branch = value
    
    @property
    def turn(self) -> Turn:
        if self._turn is None:
            raise MessageLogError("Turn is not initialized")
        return self._turn
    
    @turn.setter
    def turn(self, value: Turn):
        self._turn = value
    
    
    @property
    def is_initialized(self) -> bool:
        return self._branch is not None and self._turn is not None
    
    @property
    def session_id(self) -> int:
        if self.branch is None:
            raise MessageLogError("Branch is not initialized")
        if self.branch.session_id is None:
            raise MessageLogError("Session is not initialized")
        return self.branch.session_id
    
    @property
    def branch_id(self) -> int:
        if self.branch is None:
            raise MessageLogError("Branch is not initialized")
        return self.branch.id
    
    @property
    def turn_id(self) -> int:
        if self.turn is None:
            raise MessageLogError("Turn is not initialized")
        return self.turn.id
    
    def reset(self):
        self._branch = None
        self._turn = None
    
    

class MessageLog:
    
    
    def __init__(self, backend: MessageBackend):
        self.head = Head()
        self._backend = backend
    
    @staticmethod
    async def from_user_last_session(user_id: int):
        backend = MessageBackend()
        session = await backend.last_session(user_id=user_id)
        if session is None:
            raise MessageLogError("No session found")
        message_log = MessageLog(backend)
        branch = await backend.get_branch(session_id=session.id, branch_order=0)
        if branch is None:
            raise MessageLogError("No branch found")
        message_log.head.branch = branch
        turn = await backend.get_last_turn(branch_id=branch.id)
        if turn is not None:
            message_log.head.turn = turn
        return message_log
    
    @staticmethod
    async def from_session(session_id: int):
        backend = MessageBackend()
        message_log = MessageLog(backend)
        branch = await backend.get_branch(session_id=session_id, branch_order=0)
        if branch is None:
            raise MessageLogError("No branch found")
        message_log.head.branch = branch
        turn = await backend.get_last_turn(branch_id=branch.id)
        if turn is not None:
            message_log.head.turn = turn
        return message_log
    
    
    async def add_turn(self):
        if self.head.branch is None:
            raise MessageLogError("Branch is not initialized")
        turn = await self._backend.add_turn(Turn(branch_id=self.head.branch.id))
        self.head.turn = turn
        return turn
    
    async def update_turn_local_state(self, prompt_name: str, local_state: dict):
        if self.head.turn is None:
            raise MessageLogError("Turn is not initialized")
        self.head.turn.local_state[prompt_name] = local_state
        await self._backend.update_turn(self.head.turn.id, local_state=self.head.turn.local_state)
    
    async def append(self, message: Message):
        if self.head.branch is None:
            raise MessageLogError("Branch is not initialized")
        if self.head.turn is None:
            raise MessageLogError("Turn is not initialized")
        message.branch_id = self.head.branch.id
        message.turn_id = self.head.turn.id
        message.branch_order = self.head.branch.message_counter
        message = await self._backend.add_message(message)
        self.head.branch.message_counter += 1
        return message
    
    async def list_branches(self):
        pass
    
    async def checkout(self, branch: Branch | None = None):
        if branch is None:
            branch = await self.storage_backend.get_branch(is_main=True)
        self.head.branch = branch
        
    async def get_messages(self, limit: int = 10, offset: int = 0, session_id: int | None = None):
        count = 0
        if self.head.branch is None:
            raise MessageLogError("Branch is not initialized")
        all_msgs = []
        msgs = await self._backend.list_messages(branch_id=self.head.branch.id, limit=limit, offset=offset, session_id=session_id)
        all_msgs.extend(msgs)
        count += len(msgs)
        branch = self.head.branch        
        while count < limit:            
            if not branch.forked_from_branch_id:
                break            
            forked_from_message_order = branch.forked_from_message_order
            branch = await self._backend.get_branch_by_id(branch.forked_from_branch_id)
            if branch is None:
                raise MessageLogError("Branch not found")            
            msgs = await self._backend.list_messages(branch_id=branch.id, limit=limit-count, offset=0, max_order=forked_from_message_order, session_id=session_id)
            count += len(msgs)
            all_msgs.extend(msgs)
        return all_msgs

    # async def branch_from(self, message: Message, checkout: bool = True):
    #     if self.head.branch is None:
    #         raise MessageLogError("Branch is not initialized")
    #     branch = await self._backend.branch_from(session_id=self.head.branch.session_id, message=message)
    #     if checkout:
    #         self.head.branch = branch
    #         self.head.turn = None
    #     return branch
    
    async def branch_from(self, message: Message, checkout: bool = True):
        if self.head.branch is None:
            raise MessageLogError("Branch is not initialized")
        
        # branch = Branch(session_id=self.head.branch.session_id, forked_from_message_id=message.id)
        branch = Branch(
            session_id=self.head.branch.session_id, 
            forked_from_branch_id=message.branch_id, 
            forked_from_message_order=message.branch_order
        )
        branch = await self._backend.add_branch(branch)
        if checkout:
            self.head.reset()
            self.head.branch = branch
            # self.head.turn = None
        return branch
    
    async def switch_to(self, branch_id: int):
        branch = await self._backend.get_branch_by_id(branch_id)
        turn = await self._backend.get_last_turn(branch_id)
        if branch is None:
            raise MessageLogError("Branch not found")
        self.head.branch = branch
        if turn is not None:
            self.head.turn = turn
        return branch
    
    async def commit(self, message: Message):
        pass

    
    async def update(self, message: Message):
        pass
    
    
    

    