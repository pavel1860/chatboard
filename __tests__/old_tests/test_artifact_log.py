import pytest
from sqlalchemy import create_engine, text, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from chatboard.artifact_log.artifact_log import (
    Base,
    VersionedMixin,
    Branch,
    Turn,
    Head,
    VersionedArtifact,
    HistoryStore,
)

# Test models
class UserDetails(Base, VersionedMixin):
    __tablename__ = "user_details"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    
    def __repr__(self):
        return f"<UserDetails(id={self.id}, name={self.name})>"

class MealPlans(Base, VersionedMixin):
    __tablename__ = "meal_plans"
    id = Column(Integer, primary_key=True)
    plan = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    
    def __repr__(self):
        return f"<MealPlans(id={self.id}, user_id={self.user_id})>"

class Messages(Base, VersionedMixin):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    sender = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    user_id = Column(Integer, nullable=False)
    
    def __repr__(self):
        return f"<Messages(id={self.id}, sender={self.sender})>"

@pytest.fixture(scope="function")
def db_engine():
    """Create a test database engine."""
    # Connect to PostgreSQL server to create/drop test database
    postgres_url = "postgresql://snack:Aa123456@localhost:5432/postgres"
    test_db_name = "snackbot_test"
    
    # Connect to default postgres database to create/drop test database
    conn = psycopg2.connect(postgres_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Drop test database if it exists and create it fresh
    cursor.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
    cursor.execute(f"CREATE DATABASE {test_db_name}")
    
    cursor.close()
    conn.close()
    
    # Now connect to the test database
    DATABASE_URL = f"postgresql://snack:Aa123456@localhost:5432/{test_db_name}"
    engine = create_engine(DATABASE_URL)
    
    # Create ltree extension if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree;"))
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup after tests
    Base.metadata.drop_all(engine)
    
    # Drop the test database
    conn = psycopg2.connect(postgres_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
    cursor.close()
    conn.close()

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a new database session for a test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    
    yield session
    
    session.close()

@pytest.fixture(scope="function")
def history_store(db_session):
    """Create a HistoryStore instance."""
    return HistoryStore(db_session)

def test_create_main_branch(history_store):
    """Test creating a main branch."""
    branch = history_store.create_branch("main")
    assert branch.name == "main"
    assert branch.parent is None
    assert branch.fork_turn_id is None
    assert str(branch.path) == str(branch.id)

def test_create_child_branch(history_store):
    """Test creating a child branch."""
    main_branch = history_store.create_branch("main")
    turn = history_store.commit_turn(main_branch, "Initial commit")
    
    child_branch = history_store.create_branch(
        "feature",
        parent_branch=main_branch,
        fork_turn=turn
    )
    
    assert child_branch.name == "feature"
    assert child_branch.parent == main_branch
    assert child_branch.fork_turn_id == turn.id
    assert str(child_branch.path) == f"{main_branch.path}.{child_branch.id}"

def test_commit_turn(history_store):
    """Test committing turns to a branch."""
    branch = history_store.create_branch("main")
    
    # First turn
    turn1 = history_store.commit_turn(branch, "First commit")
    assert turn1.branch_id == branch.id
    assert turn1.turn_index == 1
    assert turn1.message == "First commit"
    
    # Second turn
    turn2 = history_store.commit_turn(branch, "Second commit")
    assert turn2.branch_id == branch.id
    assert turn2.turn_index == 2
    assert turn2.message == "Second commit"

def test_version_record(history_store, db_session):
    """Test versioning a record."""
    # Create branch and turn
    branch = history_store.create_branch("main")
    turn = history_store.commit_turn(branch, "Initial commit")
    
    # Create a user record
    user = UserDetails(name="Alice", age=30)
    db_session.add(user)
    db_session.commit()
    
    # Version the record
    artifact = history_store.version_record(
        branch,
        turn,
        "user_details",
        user.id,
        user.to_dict()
    )
    
    assert artifact.branch_id == branch.id
    assert artifact.turn_id == turn.id
    assert artifact.table_name == "user_details"
    assert artifact.record_id == str(user.id)
    assert artifact.data["name"] == "Alice"
    assert artifact.data["age"] == 30

def test_get_versioned_state(history_store, db_session):
    """Test retrieving versioned state."""
    # Setup
    branch = history_store.create_branch("main")
    turn1 = history_store.commit_turn(branch, "First commit")
    
    # Create and version first user
    user1 = UserDetails(name="Alice", age=30)
    db_session.add(user1)
    db_session.commit()
    history_store.version_record(branch, turn1, "user_details", user1.id, user1.to_dict())
    
    # Create second turn and update user
    turn2 = history_store.commit_turn(branch, "Second commit")
    user1.age = 31
    db_session.commit()
    history_store.version_record(branch, turn2, "user_details", user1.id, user1.to_dict())
    
    # Test state at turn1
    state1 = history_store.get_versioned_state(branch, turn1, "user_details")
    assert state1[str(user1.id)]["age"] == 30
    
    # Test state at turn2
    state2 = history_store.get_versioned_state(branch, turn2, "user_details")
    assert state2[str(user1.id)]["age"] == 31

def test_get_context(history_store, db_session):
    """Test retrieving overall context."""
    # Setup
    branch = history_store.create_branch("main")
    turn = history_store.commit_turn(branch, "Initial commit")
    
    # Create and version a user
    user = UserDetails(name="Alice", age=30)
    db_session.add(user)
    db_session.commit()
    history_store.version_record(branch, turn, "user_details", user.id, user.to_dict())
    
    # Get context
    context = history_store.get_context(branch)
    assert "user_details" in context
    assert context["user_details"][str(user.id)]["name"] == "Alice"
    assert context["user_details"][str(user.id)]["age"] == 30

def test_update_head(history_store):
    """Test updating HEAD pointer."""
    # Setup
    branch = history_store.create_branch("main")
    turn = history_store.commit_turn(branch, "Initial commit")
    
    # Update HEAD
    head = history_store.update_head(branch, turn)
    assert head.branch_id == branch.id
    assert head.turn_id == turn.id
    
    # Update HEAD again
    turn2 = history_store.commit_turn(branch, "Second commit")
    head = history_store.update_head(branch, turn2)
    assert head.branch_id == branch.id
    assert head.turn_id == turn2.id

def test_version_meal_plan(history_store, db_session):
    """Test versioning a meal plan record."""
    # Create branch and turn
    branch = history_store.create_branch("main")
    turn = history_store.commit_turn(branch, "Initial commit")
    
    # Create a user and meal plan
    user = UserDetails(name="Alice", age=30)
    db_session.add(user)
    db_session.commit()
    
    meal_plan = MealPlans(plan="Vegetarian", user_id=user.id)
    db_session.add(meal_plan)
    db_session.commit()
    
    # Version the meal plan
    artifact = history_store.version_record(
        branch,
        turn,
        "meal_plans",
        meal_plan.id,
        meal_plan.to_dict()
    )
    
    assert artifact.branch_id == branch.id
    assert artifact.turn_id == turn.id
    assert artifact.table_name == "meal_plans"
    assert artifact.record_id == str(meal_plan.id)
    assert artifact.data["plan"] == "Vegetarian"
    assert artifact.data["user_id"] == user.id

def test_version_meal_plan_updates(history_store, db_session):
    """Test versioning updates to a meal plan."""
    # Setup
    branch = history_store.create_branch("main")
    turn1 = history_store.commit_turn(branch, "First commit")
    
    # Create initial records
    user = UserDetails(name="Alice", age=30)
    db_session.add(user)
    db_session.commit()
    
    meal_plan = MealPlans(plan="Vegetarian", user_id=user.id)
    db_session.add(meal_plan)
    db_session.commit()
    
    # Version initial state
    history_store.version_record(branch, turn1, "meal_plans", meal_plan.id, meal_plan.to_dict())
    
    # Update meal plan in second turn
    turn2 = history_store.commit_turn(branch, "Update meal plan")
    meal_plan.plan = "Vegan"
    db_session.commit()
    history_store.version_record(branch, turn2, "meal_plans", meal_plan.id, meal_plan.to_dict())
    
    # Test states at different turns
    state1 = history_store.get_versioned_state(branch, turn1, "meal_plans")
    assert state1[str(meal_plan.id)]["plan"] == "Vegetarian"
    
    state2 = history_store.get_versioned_state(branch, turn2, "meal_plans")
    assert state2[str(meal_plan.id)]["plan"] == "Vegan"

def test_version_messages(history_store, db_session):
    """Test versioning message records."""
    # Create branch and turn
    branch = history_store.create_branch("main")
    turn = history_store.commit_turn(branch, "Initial commit")
    
    # Create a user and message
    user = UserDetails(name="Alice", age=30)
    db_session.add(user)
    db_session.commit()
    
    message = Messages(
        sender="Alice",
        text="I need help with meal planning",
        user_id=user.id
    )
    db_session.add(message)
    db_session.commit()
    
    # Version the message
    artifact = history_store.version_record(
        branch,
        turn,
        "messages",
        message.id,
        message.to_dict()
    )
    
    assert artifact.branch_id == branch.id
    assert artifact.turn_id == turn.id
    assert artifact.table_name == "messages"
    assert artifact.record_id == str(message.id)
    assert artifact.data["sender"] == "Alice"
    assert artifact.data["text"] == "I need help with meal planning"
    assert artifact.data["user_id"] == user.id

def test_branched_conversation(history_store, db_session):
    """Test versioning a conversation that branches into different paths."""
    # Create main branch and initial turn
    main_branch = history_store.create_branch("main")
    turn1 = history_store.commit_turn(main_branch, "Initial message")
    
    # Create user and initial message
    user = UserDetails(name="Alice", age=30)
    db_session.add(user)
    db_session.commit()
    
    msg1 = Messages(
        sender="Alice",
        text="What should I eat today?",
        user_id=user.id
    )
    db_session.add(msg1)
    db_session.commit()
    history_store.version_record(main_branch, turn1, "messages", msg1.id, msg1.to_dict())
    
    # Add response in main branch
    turn2 = history_store.commit_turn(main_branch, "Main branch response")
    msg2 = Messages(
        sender="Assistant",
        text="I suggest a vegetarian meal.",
        user_id=user.id
    )
    db_session.add(msg2)
    db_session.commit()
    history_store.version_record(main_branch, turn2, "messages", msg2.id, msg2.to_dict())
    
    # Create alternative branch from turn1
    alt_branch = history_store.create_branch("alternative", parent_branch=main_branch, fork_turn=turn1)
    alt_turn = history_store.commit_turn(alt_branch, "Alternative response")
    
    alt_msg = Messages(
        sender="Assistant",
        text="How about trying a new recipe?",
        user_id=user.id
    )
    db_session.add(alt_msg)
    db_session.commit()
    history_store.version_record(alt_branch, alt_turn, "messages", alt_msg.id, alt_msg.to_dict())
    
    # Test main branch conversation
    main_state = history_store.get_versioned_state(main_branch, turn2, "messages")
    assert len(main_state) == 2
    assert any(msg["text"] == "What should I eat today?" for msg in main_state.values())
    assert any(msg["text"] == "I suggest a vegetarian meal." for msg in main_state.values())
    
    # Test alternative branch conversation
    alt_state = history_store.get_versioned_state(alt_branch, alt_turn, "messages")
    assert len(alt_state) == 2
    assert any(msg["text"] == "What should I eat today?" for msg in alt_state.values())
    assert any(msg["text"] == "How about trying a new recipe?" for msg in alt_state.values()) 