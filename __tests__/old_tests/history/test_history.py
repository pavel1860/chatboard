import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chatboard.conversation.models import Base, Message, Turn, Branch
from chatboard.conversation.history import History

@pytest.fixture(scope="function")
def db_url():
    return "postgresql://snack:Aa123456@localhost:5432/snackbot_test"

@pytest.fixture(scope="function")
def history(db_url):
    """Create a test history instance."""
    history = History(db_url=db_url)
    # Drop all tables and recreate them
    Base.metadata.drop_all(bind=history._engine)
    Base.metadata.create_all(bind=history._engine)
    history.init_new_session()
    return history

def test_init(history):
    """Test initialization creates session and branch."""
    assert history._current_session is not None
    assert history._current_branch is not None
    assert len(history.get_branches()) == 1

def test_add_messages_and_commit(history):
    """Test adding messages and committing turns."""
    # Add messages to a turn
    msg1 = history.add_message("Message 1", role="user")
    msg2 = history.add_message("Message 2", role="assistant")
    
    # Messages should be uncommitted
    assert len(history._uncommitted_messages) == 2
    assert history._current_turn is not None
    
    # Commit the turn
    history.commit()
    assert len(history._uncommitted_messages) == 0
    assert history._current_turn is None
    
    # Messages should be retrievable
    messages = history.get_last_messages(10)
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Message 2"
    

def test_get_last_messages(history):
    turn1 = history.add_turn()
    history.add_message("Turn 1 Msg 1", role="user")
    history.add_message("Turn 1 Msg 2", role="assistant")
    history.commit()
    turn2 = history.add_turn()
    history.add_message("Turn 2 Msg 1", role="user")
    history.add_message("Turn 2 Msg 2", role="assistant")
    history.commit()

    history.rewind_to(turn1)
    turn1 = history.add_turn()
    history.add_message("Branch 2 Turn 1 Msg 1", role="user")
    history.add_message("Branch 2 Turn 1 Msg 2", role="assistant")
    history.commit()
    
    history.switch_to(history.get_branches()[0])
    branch1_msgs = history.get_last_messages()
    assert len(branch1_msgs) == 4
    assert branch1_msgs[0].content == "Turn 1 Msg 1"
    assert branch1_msgs[1].content == "Turn 1 Msg 2"
    assert branch1_msgs[2].content == "Turn 2 Msg 1"
    assert branch1_msgs[3].content == "Turn 2 Msg 2"
    history.switch_to(history.get_branches()[1])
    branch2_msgs = history.get_last_messages()
    assert len(branch2_msgs) == 4
    assert branch2_msgs[0].content == "Turn 1 Msg 1"
    assert branch2_msgs[1].content == "Turn 1 Msg 2"
    assert branch2_msgs[2].content == "Branch 2 Turn 1 Msg 1"
    assert branch2_msgs[3].content == "Branch 2 Turn 1 Msg 2"


# def test_turn_based_branching(history):
#     """Test turn-based branch creation and switching."""
#     # Create initial messages in first turn
#     history.add_message("Turn 1 Msg 1", role="user")
#     history.add_message("Turn 1 Msg 2", role="assistant")
#     history.commit()
    
#     # Create second turn
#     history.add_message("Turn 2 Msg", role="user")
#     history.commit()
    
#     # Get turns and create branch from first turn
#     turns = history.get_turns()
#     assert len(turns) == 2
    
#     new_branch = history.create_branch(from_turn=turns[0])
#     history.switch_to(new_branch)
    
#     # Add message in new branch
#     history.add_message("New Branch Msg", role="user")
#     history.commit()
    
#     # Verify turn relationships
#     new_turn = history.get_turns(new_branch)[0]
#     assert new_turn.parent_turn == turns[0]
    
#     # Check messages in each branch
#     original_messages = history.get_branch_messages(history._current_session.branches[0])
#     assert len(original_messages) == 3  # Two from first turn, one from second
    
#     new_messages = history.get_branch_messages(new_branch)
#     assert len(new_messages) == 1  # Just the new message

# def test_recursive_history(history):
#     """Test recursive history traversal across branches."""
#     # Create a chain of turns and branches
#     history.add_message("Main Turn 1", role="user")
#     history.commit()
    
#     history.add_message("Main Turn 2", role="user")
#     history.commit()
    
#     # Branch from first turn
#     turns = history.get_turns()
#     branch1 = history.create_branch(from_turn=turns[0])
#     history.switch_to(branch1)
#     history.add_message("Branch 1 Turn", role="user")
#     history.commit()
    
#     # Branch from the branch
#     branch1_turns = history.get_turns(branch1)
#     branch2 = history.create_branch(from_turn=branch1_turns[0])
#     history.switch_to(branch2)
#     history.add_message("Branch 2 Turn", role="user")
#     history.commit()
    
#     # Test recursive queries
#     messages = history.get_recent_messages_across_branches()
#     assert len(messages) == 4
#     contents = {msg.content for msg in messages}
#     assert contents == {"Main Turn 1", "Main Turn 2", "Branch 1 Turn", "Branch 2 Turn"}
    
#     turns = history.get_recent_turns_across_branches()
#     assert len(turns) == 4
    
#     fork_history = history.get_branch_fork_history()
#     assert len(fork_history) == 3  # Main branch + 2 forked branches

def test_test_branches(history):
    """Test creating and managing test branches."""
    # Create a test branch
    test_branch = history.create_test_branch()
    
    # Add some messages
    history.switch_to(test_branch)
    history.add_message("Test message", role="user")
    history.commit()
    
    # Get test branches
    test_branches = history.get_test_branches()
    assert len(test_branches) == 1
    assert test_branches[0].is_test is True

def test_turn_management(history):
    """Test turn-based operations."""
    # Create multiple turns
    history.add_message("Turn 1 Msg 1", role="user")
    history.add_message("Turn 1 Msg 2", role="assistant")
    history.commit()
    
    history.add_message("Turn 2 Msg 1", role="user")
    history.commit()
    
    # Get all turns
    turns = history.get_turns()
    assert len(turns) == 2
    
    # Get messages from specific turn
    turn1_messages = history.get_turn_messages(turns[0])
    assert len(turn1_messages) == 2
    assert turn1_messages[0].content == "Turn 1 Msg 1"
    
    # Rewind to first turn and create new branch
    history.rewind_to(turns[0])
    assert history._current_branch != turns[0].branch
    
    # Add message in new branch
    history.add_message("New branch msg", role="user")
    history.commit()
    
    # Verify turn parent relationship
    new_turn = history.get_turns(history._current_branch)[0]
    assert new_turn.parent_turn == turns[0]

def test_cleanup(history):
    """Test cleanup closes database connection."""
    history.cleanup()
    # Attempting operations after cleanup should raise
    with pytest.raises(Exception):
        history.add_message("Should fail", role="user")

def test_message_history(history):
    """Test message history through turns."""
    # Create first turn with messages
    history.add_message("Turn 1 Msg 1", role="user")
    history.add_message("Turn 1 Msg 2", role="assistant")
    history.commit()
    
    # Create second turn
    history.add_message("Turn 2 Msg 1", role="user")
    history.commit()
    
    # Get messages from first turn
    turns = history.get_turns()
    turn1_messages = history.get_turn_messages(turns[0])
    assert len(turn1_messages) == 2
    assert [m.content for m in turn1_messages] == ["Turn 1 Msg 1", "Turn 1 Msg 2"]
    
    # Get messages from second turn
    turn2_messages = history.get_turn_messages(turns[1])
    assert len(turn2_messages) == 1
    assert turn2_messages[0].content == "Turn 2 Msg 1" 