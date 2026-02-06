import uuid
from sqlalchemy import UUID, create_engine, Column, Integer, String, JSON, ForeignKey, DateTime, Text, Boolean, inspect
from sqlalchemy.orm import declarative_base, relationship, Session, validates, backref
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
import os


def get_json_type(db_url):
    if db_url.startswith("postgresql"):
        return JSONB
    return JSON


HISTORY_DB_URL = os.getenv("HISTORY_DB_URL", "postgresql://snack:Aa123456@localhost:5432/snackbot")
        
json_type = get_json_type(HISTORY_DB_URL)

Base = declarative_base()





class BaseUserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # This column identifies which "subclass" a row belongs to.
    type = Column(String(50))

    # The relationship to the SessionModel (shared for all apps).
    sessions = relationship("SessionModel", back_populates="user")

    __mapper_args__ = {
        # We'll store the subclass type in the 'type' column
        "polymorphic_on": type,
        # The base class identity
        "polymorphic_identity": "base_user",
    }



class SessionModel(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("BaseUserModel", back_populates="sessions")
    
    branches = relationship("BranchModel", back_populates="session", foreign_keys="BranchModel.session_id", cascade="all, delete-orphan")
    


class BranchModel(Base):
    __tablename__ = "branches"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    session_id = Column(Integer, ForeignKey("sessions.id", name="fk_branch_session_id", ondelete="CASCADE"), nullable=False)
    session = relationship("SessionModel", back_populates="branches", foreign_keys=[session_id])
    
    branch_order = Column(Integer, nullable=False, default=0)
    message_counter = Column(Integer, nullable=False, default=0)
    

    forked_from_message_order = Column(Integer, nullable=True)
    # 
    forked_from_branch_id = Column(Integer, 
                                   ForeignKey("branches.id", 
                                              name="fk_branch_forked_from_branch_id", 
                                              ondelete="SET NULL"), nullable=True)
    forked_from_branch = relationship("BranchModel", 
                                      remote_side=[id], 
                                      backref="forked_branches",
                                      cascade="all")
    # forked_from_branch = relationship("BranchModel", foreign_keys=[forked_from_branch_id], back_populates="forked_branches")    
    # forked_branches = relationship("BranchModel", back_populates="forked_from_branch", foreign_keys="BranchModel.forked_from_branch_id")
        
    # Relationships
    turns = relationship("TurnModel", back_populates="branch", foreign_keys="TurnModel.branch_id", cascade="all, delete-orphan")    
    messages = relationship("MessageModel", back_populates="branch", foreign_keys="MessageModel.branch_id", cascade="all, delete-orphan")
    
    is_test = Column(Boolean, nullable=False, default=False)
    # Relationship to forked message
    # forked_from_message_id = Column(Integer, ForeignKey("messages.id", name="fk_branch_forked_from_message_id", ondelete="SET NULL"), nullable=True)
    # forked_from_message = relationship("MessageModel", foreign_keys=[forked_from_message_id], back_populates="forked_branches")




class TurnModel(Base):
    __tablename__ = "turns"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    # Branch relationship
    branch_id = Column(Integer, ForeignKey("branches.id", name="fk_turn_branch_id", ondelete="CASCADE"), nullable=False)
    branch = relationship("BranchModel", back_populates="turns", foreign_keys=[branch_id])

    # Message relationships
    messages = relationship("MessageModel", back_populates="turn", foreign_keys="MessageModel.turn_id", cascade="all, delete-orphan")
    start_message_id = Column(Integer, ForeignKey("messages.id", name="fk_turn_start_message_id", ondelete="SET NULL"), nullable=True)
    start_message = relationship("MessageModel", foreign_keys=[start_message_id])
    
    # State
    local_state = Column(json_type, default={})






class MessageModel(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    role = Column(String, nullable=False, default="user")
    name = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    blocks = Column(json_type, nullable=True)
    extra = Column(json_type, nullable=True)
    run_id = Column(UUID(as_uuid=True), nullable=True)
    platform_id = Column(String, nullable=True)
    ref_id = Column(String, nullable=True)
    branch_order = Column(Integer)
    
    # Branch relationship
    branch_id = Column(Integer, ForeignKey("branches.id", name="fk_message_branch_id", ondelete="CASCADE"), nullable=False)
    branch = relationship("BranchModel", back_populates="messages", foreign_keys=[branch_id])
    
    # Turn relationship
    turn_id = Column(Integer, ForeignKey("turns.id", name="fk_message_turn_id", ondelete="CASCADE"), nullable=False)
    turn = relationship("TurnModel", back_populates="messages", foreign_keys=[turn_id])
    
    # Forked branches relationship
    # forked_branches = relationship("BranchModel", back_populates="forked_from_message", foreign_keys=[BranchModel.forked_from_message_id])
    
    




    
    
    
    
    
    
    
    

class TestCaseModel(Base):
    __tablename__ = "test_cases"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Test case details
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    evaluation_criteria = Column(json_type, nullable=False)  # JSON array of criteria
    
    # Test inputs
    inputs = Column(json_type, nullable=False)
    
    # Starting point for the test
    start_message_id = Column(Integer, ForeignKey("messages.id", name="fk_test_case_start_message_id", ondelete="SET NULL"), nullable=True)
    start_message = relationship("MessageModel", foreign_keys=[start_message_id])
    
    # Session this test case belongs to
    session_id = Column(Integer, ForeignKey("sessions.id", name="fk_test_case_session_id", ondelete="CASCADE"), nullable=False)
    # session = relationship("SessionModel", back_populates="test_cases", foreign_keys=[session_id])
    
    # Test runs for this test case
    test_runs = relationship("TestRunModel", back_populates="test_case", cascade="all, delete-orphan")


class TestRunModel(Base):
    __tablename__ = "test_runs"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Test case this run belongs to
    test_case_id = Column(Integer, ForeignKey("test_cases.id", name="fk_test_run_test_case_id", ondelete="CASCADE"), nullable=False)
    test_case = relationship("TestCaseModel", back_populates="test_runs")
    
    # Branch created for this test run
    branch_id = Column(Integer, ForeignKey("branches.id", name="fk_test_run_branch_id", ondelete="CASCADE"), nullable=False)
    branch = relationship("BranchModel", backref="test_runs")
    
    # Test results
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    results = Column(json_type, nullable=True)  # Detailed evaluation results
    score = Column(Integer, nullable=True)  # Overall score if applicable
    error_message = Column(Text, nullable=True)  # Error message if test failed
    meta = Column(json_type, nullable=True)  # Additional metadata about the test run
    
    @validates('status')
    def validate_status(self, key, value):
        valid_statuses = {'pending', 'running', 'completed', 'failed'}
        if value not in valid_statuses:
            raise ValueError(f"Invalid status: {value}. Must be one of {valid_statuses}")
        return value




    
    
    
    
    
    
    
    
