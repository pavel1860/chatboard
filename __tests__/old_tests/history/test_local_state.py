import pytest
from pydantic import BaseModel
from chatboard.conversation.history import History
from chatboard.prompt.local_state import LocalStore, LocalState, TurnHooks

class TestModel(BaseModel):
    value: int
    text: str

@pytest.fixture
def history():
    history = History()
    history.init_new_session()
    return history

def test_local_store_basic(history):
    """Test basic LocalStore functionality with primitive types"""
    store = LocalStore(history, prompt_name="test")
    
    # Test setting and getting values
    store.set("counter", 42)
    assert store.get("counter") == 42
    
    store.set("text", "hello")
    assert store.get("text") == "hello"
    
    # Test updating values
    store.set("counter", 43)
    assert store.get("counter") == 43
    
    # Test non-existent key
    assert store.get("non_existent") is None

def test_local_store_with_pydantic(history):
    """Test LocalStore with Pydantic models"""
    store = LocalStore(history, prompt_name="test")
    
    model = TestModel(value=1, text="test")
    store.set("model", model.model_dump())
    
    stored_data = store.get("model")
    restored_model = TestModel(**stored_data)
    
    assert restored_model.value == 1
    assert restored_model.text == "test"

def test_local_state_primitive(history):
    """Test LocalState with primitive types"""
    store = LocalStore(history, prompt_name="test")
    state = LocalState(store, "counter", 0)
    
    assert state.get() == 0
    
    state.set(1)
    assert state.get() == 1
    
    # Test persistence through store
    assert store.get("counter") == 1

def test_local_state_pydantic(history):
    """Test LocalState with Pydantic models"""
    store = LocalStore(history, prompt_name="test")
    initial_model = TestModel(value=1, text="test")
    state = LocalState(store, "model", initial_model)
    
    # Test initial state
    current = state.get()
    assert isinstance(current, TestModel)
    assert current.value == 1
    assert current.text == "test"
    
    # Test updating state
    new_model = TestModel(value=2, text="updated")
    state.set(new_model)
    
    current = state.get()
    assert current.value == 2
    assert current.text == "updated"

def test_turn_hooks(history):
    """Test TurnHooks functionality"""
    hooks = TurnHooks(history, prompt_name="test")
    
    # Test primitive state
    counter_state = hooks.use_state("counter", 0)
    assert counter_state.get() == 0
    
    counter_state.set(1)
    assert counter_state.get() == 1
    
    # Test model state
    model_state = hooks.use_state("model", TestModel(value=1, text="test"))
    current_model = model_state.get()
    assert current_model.value == 1
    assert current_model.text == "test"

def test_state_persistence_across_turns(history):
    """Test that state persists across turns"""
    # First turn
    hooks1 = TurnHooks(history, prompt_name="test")
    counter_state = hooks1.use_state("counter", 0)
    counter_state.set(42)
    
    # Create new turn
    history.add_turn()
    
    # Second turn - should have access to previous state
    hooks2 = TurnHooks(history, prompt_name="test")
    counter_state2 = hooks2.use_state("counter", 0)
    assert counter_state2.get() == 42

def test_multiple_prompt_states(history):
    """Test handling multiple prompt states in the same turn"""
    # First prompt
    hooks1 = TurnHooks(history, prompt_name="prompt1")
    state1 = hooks1.use_state("counter", 0)
    state1.set(1)
    
    # Second prompt
    hooks2 = TurnHooks(history, prompt_name="prompt2")
    state2 = hooks2.use_state("counter", 0)
    state2.set(2)
    
    # Verify states are independent
    assert state1.get() == 1
    assert state2.get() == 2

def test_error_handling(history):
    """Test error handling in local state management"""
    store = LocalStore(history, prompt_name=None)
    
    # Should raise ValueError when trying to set state without prompt name
    with pytest.raises(ValueError):
        store.set("key", "value")
    
    # Should handle None values
    store = LocalStore(history, prompt_name="test")
    store.set("null_value", None)
    assert store.get("null_value") is None 