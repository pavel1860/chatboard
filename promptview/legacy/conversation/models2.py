import uuid
from sqlalchemy import UUID, create_engine, Column, Integer, String, JSON, ForeignKey, DateTime, Text, Boolean, inspect
from sqlalchemy.orm import declarative_base, relationship, Session, validates
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
import os
from contextlib import contextmanager





def get_json_type(db_url):
    if db_url.startswith("postgresql"):
        return JSONB
    return JSON


HISTORY_DB_URL = os.getenv("HISTORY_DB_URL", "postgresql://snack:Aa123456@localhost:5432/snackbot")
        
json_type = get_json_type(HISTORY_DB_URL)

Base = declarative_base()



class MessageSession(Base):
    __tablename__ = "message_sessions"
    id = Column(Integer, primary_key=True)    
    branches = relationship("Branch", 
                          back_populates="session",
                          cascade="all, delete-orphan")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # main_branch_id = Column(Integer, ForeignKey("branches.id", name="fk_main_branch_id", ondelete="SET NULL"), nullable=True)
    # main_branch = relationship("Branch", foreign_keys=[main_branch_id])



class Branch(Base):
    __tablename__ = 'branches'
    
    id = Column(Integer, primary_key=True)
    # name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_test = Column(Boolean, default=False)  # Indicates if this is a test branch
    
    # The turn from which this branch was created
    forked_from_turn_id = Column(Integer, 
                                ForeignKey('turns.id',
                                         name='fk_branch_forked_from_turn_id',
                                         ondelete="SET NULL"), 
                                nullable=True)
    forked_from_turn = relationship("Turn", 
                                  back_populates="forked_branches",
                                  foreign_keys=[forked_from_turn_id])
    
    # All turns in this branch
    turns = relationship("Turn", 
                        back_populates="branch",
                        foreign_keys="Turn.branch_id",
                        cascade="all, delete-orphan")
    
    # All messages in this branch (denormalized for query performance)
    messages = relationship("Message", 
                          back_populates="branch",
                          foreign_keys="Message.branch_id",
                          cascade="all, delete-orphan")
    
    session_id = Column(Integer, 
                       ForeignKey("message_sessions.id",
                                name='fk_branch_session_id',
                                ondelete="CASCADE"), 
                       nullable=False)
    session = relationship("MessageSession", back_populates="branches")
    

class Turn(Base):
    __tablename__ = 'turns'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    branch_id = Column(Integer, 
                      ForeignKey('branches.id', 
                                name='fk_turn_branch_id',
                                ondelete="CASCADE"), 
                      nullable=False)
    
    local_state = Column(JSONB, default={})
    
    # Parent Turn (the Turn this one was based on)
    parent_turn_id = Column(Integer, 
                          ForeignKey('turns.id',
                                   name='fk_turn_parent_id',
                                   ondelete="SET NULL"), 
                          nullable=True)
    parent_turn = relationship("Turn", remote_side=[id],
                             backref="child_turns")
    
    # The first message of this turn
    start_message_id = Column(Integer, 
                            ForeignKey('messages.id',
                                     name='fk_turn_start_message_id',
                                     ondelete="SET NULL"), 
                            nullable=True)
    
    # All messages in this turn
    messages = relationship("Message", 
                          back_populates="turn",
                          foreign_keys="Message.turn_id",
                          cascade="all, delete-orphan")
    
    
    
    # The branch this turn belongs to
    branch = relationship("Branch", back_populates="turns", foreign_keys=[branch_id])
    
    # Branches that were forked from this turn
    forked_branches = relationship("Branch", 
                                 back_populates="forked_from_turn",
                                 foreign_keys="Branch.forked_from_turn_id")
    
    def get_turn_history(self):
        """Get the history of turns leading to this one."""
        history = []
        current = self
        while current is not None:
            history.append(current)
            current = current.parent_turn
        return history

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    role = Column(String, nullable=False, default="user")
    prompt = Column(String, nullable=False, default="main")
    run_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    action_calls = Column(json_type, nullable=True)
    platform_id = Column(String, nullable=True)
    
    # The turn this message belongs to
    turn_id = Column(Integer, 
                    ForeignKey('turns.id',
                             name='fk_message_turn_id',
                             ondelete="CASCADE"), 
                    nullable=False)
    turn = relationship("Turn", back_populates="messages", foreign_keys=[turn_id])
    
    # The branch this Message belongs to (denormalized for query performance)
    branch_id = Column(Integer, 
                      ForeignKey('branches.id',
                               name='fk_message_branch_id',
                               ondelete="CASCADE"), 
                      nullable=False)
    branch = relationship("Branch", back_populates="messages", foreign_keys=[branch_id])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If turn_id is provided but branch_id isn't, set branch_id from turn
        session = inspect(self).session if hasattr(self, '_sa_instance_state') else None
        if session and kwargs.get('turn_id') and not kwargs.get('branch_id'):
            turn = session.query(Turn).get(kwargs['turn_id'])
            if turn:
                self.branch_id = turn.branch_id
    
    @validates('branch_id', 'turn_id')
    def validate_branch_consistency(self, key, value):
        session = inspect(self).session if hasattr(self, '_sa_instance_state') else None
        if session:
            if key == 'turn_id' and value and self.branch_id:
                turn = session.query(Turn).get(value)
                if turn and turn.branch_id != self.branch_id:
                    raise ValueError(f"Message branch_id {self.branch_id} does not match turn's branch_id {turn.branch_id}")
            elif key == 'branch_id' and value and self.turn_id:
                turn = session.query(Turn).get(self.turn_id)
                if turn and turn.branch_id != value:
                    raise ValueError(f"Message branch_id {value} does not match turn's branch_id {turn.branch_id}")
        return value

    def render_html(self):
        def role_chip(role):
            if role == "user":
                return '<div class="role-chip user">User</div>'
            elif role == "assistant": 
                return '<div class="role-chip assistant">Assistant</div>'
            return '<div class="role-chip tool">Tool</div>'
        
        def action_calls_html(action_calls):
            def action_call_html(action_call):
                return f"""<div class="action-call">
                    <div class="action-call-id">{action_call["id"].split("_")[-1][:8]}</div>
                    <div class="action-call-name">{action_call["name"]}</div>                    
                </div>"""
            if action_calls is None:
                return ''
            return '<div class="action-calls">' + ', '.join([action_call_html(action_call) for action_call in action_calls]) + '</div>'
                        
        return f"""
<style>
.message-container {{
    max-width: 800px;
    margin: 2px;
    padding: 4px;
    border-radius: 4px;
    background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}

.message-header {{
    display: flex;
    align-items: center;
    margin-bottom: 1px;
}}

.message-id {{
    font-size: 10px;
    color: #666;
    margin-right: 4px;
}}

.role-chip {{
    padding: 2px 6px;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 500;
    margin-right: 4px;
}}

.role-chip.user {{
    background: #E3F2FD;
    color: #1976D2;
}}

.role-chip.assistant {{
    background: #ffebee;
    color: #c62828;
}}

.role-chip.tool {{
    background: #FFF3E0;
    color: #F57C00;
}}

.message-time {{
    font-size: 10px;
    color: #999;
}}

.message-content {{
    font-size: 12px;
    line-height: 0.6;
    color: #333;
    white-space: pre-wrap;
}}

.action-call {{
    display: flex;
    align-items: center;
    background: #f5f5f5;
    border-radius: 4px;
    padding: 8px;
    margin: 4px 0;
    border-left: 3px solid #F57C00;
}}

.action-call-id {{
    font-size: 10px;
    color: #666;
    margin-right: 8px;
    padding: 2px 6px;
    background: #FFF3E0;
    border-radius: 4px;
}}

.action-call-name {{
    font-size: 11px;
    font-weight: 500;
    color: #F57C00;
    margin-right: 8px;
    text-transform: uppercase;
}}

</style>

<div class="message-container">
    <div class="message-header">
        <span class="message-id">#{self.id}</span>
        {role_chip(self.role)}
        <span class="message-time">{self.created_at.strftime('%m-%d %H:%M:%S')}</span>
    </div>
    <div class="message-content">
        {self.content}
    </div>
    {action_calls_html(self.action_calls)}
</div>
"""

    def show(self):
        from IPython.display import display, HTML
        return display(HTML(self.render_html()))

    def get_message_history(self):
        """Get the history of messages in this turn."""
        return self.turn.messages

# Create a new database or connect to existing one
def init_db(db_path="postgresql://snack:Aa123456@localhost:5432/snackbot"):
    # Remove existing database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create new database and tables
    # engine = create_engine(f'sqlite:///{db_path}')
    engine = create_engine(db_path)
    Base.metadata.drop_all(engine)  # Drop all tables if they exist
    Base.metadata.create_all(engine)  # Create new tables
    return engine





