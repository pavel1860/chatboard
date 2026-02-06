import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import uuid

from chatboard.conversation.models import Base, MessageSession, Branch, Turn, Message

@pytest.fixture(scope="function")
def db_url():
    return "postgresql://snack:Aa123456@localhost:5432/snackbot_test"

@pytest.fixture(scope="function")
def engine(db_url):
    engine = create_engine(db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture(scope="function")
def session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture(scope="function")
def message_session(session):
    msg_session = MessageSession()
    session.add(msg_session)
    session.commit()
    return msg_session

@pytest.fixture(scope="function")
def branch(session, message_session):
    branch = Branch(session=message_session)
    session.add(branch)
    session.commit()
    return branch

@pytest.fixture(scope="function")
def turn(session, branch):
    turn = Turn(branch=branch)
    session.add(turn)
    session.commit()
    return turn

def test_create_message_session(session):
    """Test creating a message session."""
    msg_session = MessageSession()
    session.add(msg_session)
    session.commit()
    
    # Query to verify
    result = session.query(MessageSession).first()
    assert result is not None
    assert isinstance(result.created_at, datetime)
    assert len(result.branches) == 0

def test_create_branch(session, message_session):
    """Test creating a branch."""
    branch = Branch(session=message_session)
    session.add(branch)
    session.commit()
    
    # Query to verify
    result = session.query(Branch).first()
    assert result is not None
    assert result.session_id == message_session.id
    assert result.is_test is False
    assert result.forked_from_turn_id is None
    assert len(result.turns) == 0
    assert len(result.messages) == 0

def test_create_turn(session, branch):
    """Test creating a turn."""
    turn = Turn(branch=branch)
    session.add(turn)
    session.commit()
    
    # Query to verify
    result = session.query(Turn).first()
    assert result is not None
    assert result.branch_id == branch.id
    assert result.start_message_id is None
    assert len(result.messages) == 0
    assert len(result.forked_branches) == 0

def test_create_message(session, turn):
    """Test creating a message."""
    message = Message(
        content="Test message",
        role="user",
        turn=turn,
        branch_id=turn.branch_id
    )
    session.add(message)
    session.commit()
    
    # Query to verify
    result = session.query(Message).first()
    assert result is not None
    assert result.content == "Test message"
    assert result.role == "user"
    assert result.turn_id == turn.id
    assert result.branch_id == turn.branch_id

def test_message_history(session, turn):
    """Test message history through turn relationship."""
    # Create multiple messages in a turn
    message1 = Message(content="Message 1", role="user", turn=turn, branch_id=turn.branch_id)
    message2 = Message(content="Message 2", role="assistant", turn=turn, branch_id=turn.branch_id)
    message3 = Message(content="Message 3", role="user", turn=turn, branch_id=turn.branch_id)
    session.add_all([message1, message2, message3])
    session.commit()
    
    # Query to verify
    result = session.query(Message).filter_by(id=message3.id).first()
    history = result.get_message_history()
    assert len(history) == 3
    assert [m.content for m in history] == ["Message 1", "Message 2", "Message 3"]

def test_cascade_deletes(session, message_session):
    """Test that deleting a session cascades properly."""
    # Create a complete conversation structure
    branch = Branch(session=message_session)
    session.add(branch)
    session.flush()  # Flush to get the branch.id
    
    turn = Turn(branch=branch)
    session.add(turn)
    session.flush()  # Flush to get the turn.id
    
    message = Message(
        content="Test message",
        role="user",
        turn=turn,
        branch_id=branch.id  # Explicitly set branch_id
    )
    session.add(message)
    session.commit()
    
    # Verify the structure is created
    assert session.query(Branch).count() == 1
    assert session.query(Turn).count() == 1
    assert session.query(Message).count() == 1
    
    # Delete the session
    session.delete(message_session)
    session.commit()
    
    # Verify everything is deleted
    assert session.query(MessageSession).count() == 0
    assert session.query(Branch).count() == 0
    assert session.query(Turn).count() == 0
    assert session.query(Message).count() == 0

def test_test_branch(session, message_session):
    """Test creating and identifying test branches."""
    # Create a main branch
    main_branch = Branch(session=message_session)
    session.add(main_branch)
    session.commit()
    
    # Create a test branch
    test_branch = Branch(session=message_session, is_test=True)
    session.add(test_branch)
    session.commit()
    
    # Query to verify
    test_branches = session.query(Branch).filter_by(is_test=True).all()
    main_branches = session.query(Branch).filter_by(is_test=False).all()
    
    assert len(test_branches) == 1
    assert len(main_branches) == 1
    assert test_branches[0].id == test_branch.id
    assert main_branches[0].id == main_branch.id

def test_message_session_cascade(session, message_session):
    """Test that deleting a session cascades to branches."""
    branch = Branch(session=message_session)
    session.add(branch)
    session.commit()
    
    session.delete(message_session)
    session.commit()
    
    assert session.query(Branch).count() == 0

def test_branch_cascade(session, branch):
    """Test that deleting a branch cascades to turns and messages."""
    turn = Turn(branch=branch)
    message = Message(content="Test", role="user", turn=turn, branch=branch)
    session.add_all([turn, message])
    session.commit()
    
    session.delete(branch)
    session.commit()
    
    assert session.query(Turn).count() == 0
    assert session.query(Message).count() == 0

def test_turn_cascade(session, turn):
    """Test that deleting a turn cascades to messages."""
    message = Message(content="Test", role="user", turn=turn, branch=turn.branch)
    session.add(message)
    session.commit()
    
    session.delete(turn)
    session.commit()
    
    assert session.query(Message).count() == 0

def test_turn_parent_child(session, branch):
    """Test turn parent-child relationship."""
    # Create parent turn
    parent_turn = Turn(branch=branch)
    session.add(parent_turn)
    session.commit()
    
    # Create child turn
    child_turn = Turn(branch=branch, parent_turn=parent_turn)
    session.add(child_turn)
    session.commit()
    
    # Test relationships
    assert child_turn.parent_turn == parent_turn
    assert parent_turn.child_turns == [child_turn]
    
    # Test history
    history = child_turn.get_turn_history()
    assert len(history) == 2
    assert history[0] == child_turn
    assert history[1] == parent_turn

def test_branch_fork(session, branch):
    """Test branch forking from turn."""
    # Create turn to fork from
    original_turn = Turn(branch=branch)
    session.add(original_turn)
    session.commit()
    
    # Create new branch forked from turn
    new_branch = Branch(session=branch.session, forked_from_turn=original_turn)
    session.add(new_branch)
    session.commit()
    
    assert new_branch.forked_from_turn == original_turn
    assert original_turn.forked_branches == [new_branch]

def test_message_turn_relationship(session, turn):
    """Test message-turn relationship."""
    message1 = Message(content="Test 1", role="user", turn=turn, branch=turn.branch)
    message2 = Message(content="Test 2", role="assistant", turn=turn, branch=turn.branch)
    session.add_all([message1, message2])
    session.commit()
    
    # Test turn.messages relationship
    assert len(turn.messages) == 2
    assert turn.messages[0].content == "Test 1"
    assert turn.messages[1].content == "Test 2" 