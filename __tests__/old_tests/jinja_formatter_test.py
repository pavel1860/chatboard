import json
import pytest
from chatboard.utils.string_utils import SafeJinjaFormatter




@pytest.fixture
def formatter():
    return SafeJinjaFormatter()

def test_basic_formatting(formatter):
    result = formatter.format('Hello, {{ name }}!', name='World')
    assert result == 'Hello, World!'

def test_json_with_placeholder(formatter):
    result = formatter.format('{"balance": "{{ amount }}"}', amount='1000')
    assert result == '{"balance": "1000"}'  # JSON with placeholder should remain unchanged

def test_mixed_formatting(formatter):
    result = formatter.format(
        'Hello, {{ name }}! Your balance is {"balance": "{{ amount }}"} and status is {{ status }}.',
        name='World',
        amount='1000',
        status='active'
    )
    assert result == 'Hello, World! Your balance is {"balance": "1000"} and status is active.'
  
  
def test_json_formatting(formatter):
    j = {"balance": "1000", "status": "active", "name": "World"}
    result = formatter.format(
        json.dumps(j),
        name="Alice",
        balance= "2000",
        status="inactive"
    )
    assert result == json.dumps(j)
  

def test_advanced_json_formatting(formatter):
    j = {"balance": "1000", "status": "active", "name": "World"}
    result = formatter.format(
        "Hello this is the status of {{ name }} " + json.dumps(j),
        name="Alice",
        balance= "2000",
        status="inactive"
    )
    assert result == f"Hello this is the status of Alice " + json.dumps(j)
    
    

def test_escaped_braces(formatter):
    result = formatter.format('Hello, \\{\\{ name \\}\\}!', name='World')
    assert result == 'Hello, \\{\\{ name \\}\\}!'

def test_no_placeholder(formatter):
    result = formatter.format('Hello, World! This is a static string.')
    assert result == 'Hello, World! This is a static string.'

def test_placeholder_not_in_kwargs(formatter):
    result = formatter.format('Hello, {{ name }}!', other='Value')
    assert result == 'Hello, !'  # Jinja leaves it empty if the key is not found

def test_complex_structure(formatter):
    result = formatter.format(
        'Data: {"users": [{"name": "{{ username }}", "balance": "{{ balance }}"}]}',
        username='john_doe',
        balance=1000
    )
    assert result == 'Data: {"users": [{"name": "john_doe", "balance": "1000"}]}'

def test_recursive_formatting(formatter):
    result = formatter.format(
        'Hello, {{ name_123 }}!',
        name_123='Alice'
    )
    assert result == 'Hello, Alice!'

def test_ignore_non_jinja_placeholders(formatter):
    result = formatter.format('Hello, {name}! This is a non-Jinja placeholder.')
    assert result == 'Hello, {name}! This is a non-Jinja placeholder.'

if __name__ == "__main__":
    pytest.main()
