import pytest
import textwrap
from chatboard.prompt.legacy.block3 import Block, TitleBlock, ListBlock, block

class TestBlockBasics:
    """Test basic functionality of the Block class."""
    
    def test_block_creation(self):
        """Test creating a basic block with content."""
        b = Block("Hello, world!")
        assert "Hello, world!" in b.render()
        
    def test_block_append(self):
        """Test appending content to a block."""
        b = Block()
        b.append("First line")
        b.append("Second line")
        rendered = b.render()
        assert "First line" in rendered
        assert "Second line" in rendered
        
    def test_block_extend(self):
        """Test extending a block with multiple items."""
        b = Block()
        b.extend(["First line", "Second line", "Third line"])
        rendered = b.render()
        assert "First line" in rendered
        assert "Second line" in rendered
        assert "Third line" in rendered
        
    def test_block_iadd_operator(self):
        """Test the += operator for adding content."""
        b = Block()
        b += "First line"
        b += "Second line"
        rendered = b.render()
        assert "First line" in rendered
        assert "Second line" in rendered
        
    def test_block_add_operator(self):
        """Test the + operator for concatenation."""
        b1 = Block("First block")
        b2 = Block("Second block")
        b3 = b1 + b2
        rendered = b3.render()
        assert "First block" in rendered
        assert "Second block" in rendered
        
    def test_block_str_representation(self):
        """Test string representation of a block."""
        b = Block("Hello, world!")
        assert str(b) == "Hello, world!"
        
    def test_block_with_role(self):
        """Test creating a block with a specific role."""
        b = Block("System message", role="system")
        assert b.role == "system"
        
    def test_block_with_tag(self):
        """Test creating a block with a tag."""
        b = Block("Tagged content", tag="important")
        assert b.tag == "important"
        
    def test_block_get_by_tag(self):
        """Test retrieving a block by tag."""
        parent = Block()
        child = Block("Tagged content", tag="important")
        parent.append(child)
        retrieved = parent.get("important")
        assert retrieved is child
        
    def test_block_find_all(self):
        """Test finding all blocks matching criteria."""
        parent = Block()
        child1 = Block("User message 1", role="user")
        child2 = Block("User message 2", role="user")
        child3 = Block("System message", role="system")
        parent.extend([child1, child2, child3])
        
        user_blocks = parent.find_all(role="user")
        assert len(user_blocks) == 2
        assert child1 in user_blocks
        assert child2 in user_blocks
        assert child3 not in user_blocks
        
    def test_block_as_title(self):
        """Test using a regular block as a title block."""
        b = Block("Section Title", title_type="md")
        b.append("Content under the title")
        rendered = b.render()
        assert "## Section Title" in rendered
        assert "Content under the title" in rendered
        
    def test_block_as_list(self):
        """Test using a regular block as a list block."""
        b = Block(bullet_type="number")
        b.extend(["First item", "Second item", "Third item"])
        rendered = b.render()
        assert "1. First item" in rendered
        assert "2. Second item" in rendered
        assert "3. Third item" in rendered


class TestTitleBlock:
    """Test TitleBlock functionality."""
    
    def test_title_block_md(self):
        """Test markdown title block rendering."""
        tb = TitleBlock("Section Title", type="md")
        tb.append("Content under the title")
        rendered = tb.render()
        assert "## Section Title" in rendered
        assert "Content under the title" in rendered
        
    def test_title_block_xml(self):
        """Test XML title block rendering."""
        tb = TitleBlock("Section", type="xml")
        tb.append("Content inside the section")
        rendered = tb.render()
        assert "<Section>" in rendered
        assert "Content inside the section" in rendered
        assert "</Section>" in rendered
        
    def test_title_block_html(self):
        """Test HTML title block rendering."""
        tb = TitleBlock("Section Title", type="html")
        tb.append("Content under the title")
        rendered = tb.render()
        assert "<h2>Section Title</h2>" in rendered
        assert "Content under the title" in rendered
        
    def test_nested_title_blocks(self):
        """Test nesting title blocks."""
        parent = TitleBlock("Parent Title", type="md")
        child = parent.title("Child Title", type="md")
        child.append("Child content")
        rendered = parent.render()
        assert "## Parent Title" in rendered
        assert "### Child Title" in rendered
        assert "Child content" in rendered
        
    def test_title_block_equivalence(self):
        """Test that TitleBlock is equivalent to Block with title_type."""
        tb1 = TitleBlock("Title", type="md")
        tb2 = Block("Title", title_type="md")
        
        assert tb1.render() == tb2.render()


class TestListBlock:
    """Test ListBlock functionality."""
    
    def test_list_block_number(self):
        """Test numbered list rendering."""
        lb = ListBlock(bullet="number")
        lb.extend(["First item", "Second item", "Third item"])
        rendered = lb.render()
        assert "1. First item" in rendered
        assert "2. Second item" in rendered
        assert "3. Third item" in rendered
        
    def test_list_block_alpha(self):
        """Test alphabetical list rendering."""
        lb = ListBlock(bullet="alpha")
        lb.extend(["First item", "Second item", "Third item"])
        rendered = lb.render()
        assert "a. First item" in rendered
        assert "b. Second item" in rendered
        assert "c. Third item" in rendered
        
    def test_list_block_bullet(self):
        """Test bullet list rendering."""
        lb = ListBlock(bullet="*")
        lb.extend(["First item", "Second item", "Third item"])
        rendered = lb.render()
        assert "* First item" in rendered
        assert "* Second item" in rendered
        assert "* Third item" in rendered
        
    def test_nested_list_items(self):
        """Test nesting blocks within list items."""
        lb = ListBlock(bullet="number")
        item1 = Block("Main point")
        item1.append("Supporting detail")
        lb.append(item1)
        lb.append("Simple item")
        
        rendered = lb.render()
        assert "1. Main point" in rendered
        assert "Supporting detail" in rendered
        assert "2. Simple item" in rendered
        
    def test_list_block_equivalence(self):
        """Test that ListBlock is equivalent to Block with bullet_type."""
        lb1 = ListBlock(bullet="number")
        lb2 = Block(bullet_type="number")
        
        lb1.append("Item")
        lb2.append("Item")
        
        assert lb1.render() == lb2.render()


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with Block() as b:
            b.append("Content inside context")
            b.append("More content")
        
        rendered = b.render()
        assert "Content inside context" in rendered
        assert "More content" in rendered
        
    def test_nested_context_managers(self):
        """Test nesting context managers."""
        with Block() as parent:
            parent.append("Parent content")
            with Block() as child:
                child.append("Child content")
            parent.append("More parent content")
        
        rendered = parent.render()
        assert "Parent content" in rendered
        assert "Child content" in rendered
        assert "More parent content" in rendered
        
    def test_auto_append_in_context(self):
        """Test auto-appending blocks in context."""
        with Block() as parent:
            Block("Auto-appended child")
            Block("Another auto-appended child")
        
        rendered = parent.render()
        assert "Auto-appended child" in rendered
        assert "Another auto-appended child" in rendered


class TestAdvancedUseCases:
    """Test advanced use cases for blocks."""
    
    def test_llm_prompt_generation(self):
        """Test generating a complete LLM prompt."""
        with Block(role="system") as system_message:
            with system_message.title("Instructions", type="md") as instructions:
                instructions.append("You are a helpful AI assistant.")
                instructions.append("Answer questions accurately and concisely.")
                
            with system_message.title("Guidelines", type="md") as guidelines:
                with guidelines.list(bullet="number") as rules:
                    rules.append("Be respectful and polite.")
                    rules.append("Avoid making things up.")
                    rules.append("Cite sources when possible.")
                    
            with system_message.title("Output Format", type="md") as output_format:
                output_format.append("Provide your answer in markdown format.")
                
        user_message = Block("What is the capital of France?", role="user")
        
        # Simulate a complete prompt
        prompt = system_message.render() + "\n\n" + user_message.render()
        
        # Assertions
        assert "# Instructions" in prompt
        assert "You are a helpful AI assistant." in prompt
        assert "# Guidelines" in prompt
        assert "1. Be respectful and polite." in prompt
        assert "2. Avoid making things up." in prompt
        assert "3. Cite sources when possible." in prompt
        assert "# Output Format" in prompt
        assert "Provide your answer in markdown format." in prompt
        assert "What is the capital of France?" in prompt
    
    def test_sql_query_generation(self):
        """Test generating a SQL query with blocks."""
        with Block() as sql_query:
            sql_query.append("-- Generated SQL Query")
            sql_query.append("SELECT")
            
            with sql_query.list(bullet="-") as columns:
                columns.append("users.id")
                columns.append("users.name")
                columns.append("users.email")
                columns.append("orders.order_date")
                columns.append("orders.total_amount")
            
            sql_query.append("FROM users")
            sql_query.append("JOIN orders ON users.id = orders.user_id")
            sql_query.append("WHERE orders.status = 'completed'")
            sql_query.append("AND orders.order_date >= '2023-01-01'")
            sql_query.append("ORDER BY orders.order_date DESC")
            sql_query.append("LIMIT 100;")
        
        rendered = sql_query.render()
        
        # Assertions
        assert "-- Generated SQL Query" in rendered
        assert "SELECT" in rendered
        assert "- users.id" in rendered
        assert "- users.name" in rendered
        assert "FROM users" in rendered
        assert "JOIN orders ON users.id = orders.user_id" in rendered
        assert "WHERE orders.status = 'completed'" in rendered
        assert "ORDER BY orders.order_date DESC" in rendered
        assert "LIMIT 100;" in rendered
    
    def test_xml_generation(self):
        """Test generating XML with blocks."""
        with Block() as xml_doc:
            xml_doc.append('<?xml version="1.0" encoding="UTF-8"?>')
            
            with xml_doc.title("root", type="xml") as root:
                with root.title("person", type="xml") as person:
                    with person.title("name", type="xml") as name:
                        name.append("John Doe")
                    
                    with person.title("contact", type="xml") as contact:
                        with contact.title("email", type="xml") as email:
                            email.append("john.doe@example.com")
                        with contact.title("phone", type="xml") as phone:
                            phone.append("+1-555-123-4567")
                    
                    with person.title("address", type="xml") as address:
                        with address.title("street", type="xml") as street:
                            street.append("123 Main St")
                        with address.title("city", type="xml") as city:
                            city.append("Anytown")
                        with address.title("state", type="xml") as state:
                            state.append("CA")
                        with address.title("zip", type="xml") as zip_code:
                            zip_code.append("12345")
        
        rendered = xml_doc.render()
        
        # Assertions
        assert '<?xml version="1.0" encoding="UTF-8"?>' in rendered
        assert "<root>" in rendered
        assert "<person>" in rendered
        assert "<name>" in rendered
        assert "John Doe" in rendered
        assert "<contact>" in rendered
        assert "<email>" in rendered
        assert "john.doe@example.com" in rendered
        assert "<phone>" in rendered
        assert "+1-555-123-4567" in rendered
        assert "<address>" in rendered
        assert "<street>" in rendered
        assert "123 Main St" in rendered
        assert "</zip>" in rendered
        assert "</address>" in rendered
        assert "</person>" in rendered
        assert "</root>" in rendered
    
    def test_complex_document_generation(self):
        """Test generating a complex document with mixed formatting."""
        with Block() as document:
            with document.title("Technical Documentation", type="md") as doc:
                doc.append("This document provides technical specifications for the system.")
                
                with doc.title("Overview", type="md") as overview:
                    overview.append("The system consists of multiple components working together.")
                    
                    with overview.list(bullet="number") as components:
                        components.append("Frontend UI")
                        components.append("Backend API")
                        components.append("Database")
                        components.append("Authentication Service")
                
                with doc.title("API Reference", type="md") as api:
                    api.append("Below is the API reference for the main endpoints.")
                    
                    with api.title("Endpoints", type="md") as endpoints:
                        with endpoints.title("GET /users", type="md") as get_users:
                            get_users.append("Retrieves a list of users.")
                            
                            with get_users.title("Parameters", type="md") as params:
                                with params.list(bullet="-") as param_list:
                                    param_list.append("page: Page number (default: 1)")
                                    param_list.append("limit: Items per page (default: 20)")
                            
                            with get_users.title("Response", type="md") as response:
                                response.append("```json")
                                response.append('{')
                                response.append('  "users": [')
                                response.append('    {')
                                response.append('      "id": 1,')
                                response.append('      "name": "John Doe",')
                                response.append('      "email": "john@example.com"')
                                response.append('    }')
                                response.append('  ],')
                                response.append('  "total": 100,')
                                response.append('  "page": 1,')
                                response.append('  "limit": 20')
                                response.append('}')
                                response.append('```')
        
        rendered = document.render()
        
        # Assertions
        assert "# Technical Documentation" in rendered
        assert "## Overview" in rendered
        assert "1. Frontend UI" in rendered
        assert "2. Backend API" in rendered
        assert "## API Reference" in rendered
        assert "### GET /users" in rendered
        assert "#### Parameters" in rendered
        assert "- page: Page number (default: 1)" in rendered
        assert "#### Response" in rendered
        assert "```json" in rendered
        assert '  "users": [' in rendered
        assert '      "name": "John Doe",' in rendered
        assert '```' in rendered


class TestBlockFunction:
    """Test the block function."""
    
    def test_block_function_basic(self):
        """Test basic usage of the block function."""
        b = block("Hello, world!")
        assert "Hello, world!" in b.render()
        
    def test_block_function_with_role(self):
        """Test block function with role."""
        b = block("System message", role="system")
        assert b.role == "system"
        assert "System message" in b.render()


class TestBlockSerialization:
    """Test block serialization and deserialization."""
    
    def test_to_dict(self):
        """Test converting a block to a dictionary."""
        b = Block("Parent content", role="system", tag="parent")
        child = Block("Child content", role="user", tag="child")
        b.append(child)
        
        dict_repr = b.to_dict()
        
        assert dict_repr["role"] == "system"
        assert dict_repr["tag"] == "parent"
        assert dict_repr["content"] == "Parent content"
        assert len(dict_repr["items"]) == 1
        assert isinstance(dict_repr["items"][0], dict)
        assert dict_repr["items"][0]["role"] == "user"
        assert dict_repr["items"][0]["tag"] == "child"
        assert dict_repr["items"][0]["content"] == "Child content"
        
    def test_from_dict(self):
        """Test creating a block from a dictionary."""
        dict_repr = {
            "role": "system",
            "tag": "parent",
            "content": "Parent content",
            "items": [
                {
                    "role": "user",
                    "tag": "child",
                    "content": "Child content",
                    "items": []
                },
                "Simple string item"
            ]
        }
        
        b = Block.from_dict(dict_repr)
        
        assert b.role == "system"
        assert b.tag == "parent"
        assert b.content == "Parent content"
        assert len(b.items) == 2
        assert isinstance(b.items[0], Block)
        assert b.items[0].role == "user"
        assert b.items[0].tag == "child"
        assert b.items[0].content == "Child content"
        assert "Simple string item" in b.render()


class TestWhitespaceHandling:
    """Test whitespace handling in blocks."""
    
    def test_dedent_on_append(self):
        """Test that text is dedented when appended."""
        b = Block()
        b.append("""
            This text has
            extra indentation
            that should be removed.
        """)
        
        rendered = b.render()
        assert "This text has" in rendered
        assert "extra indentation" in rendered
        assert "that should be removed." in rendered
        assert "            This text" not in rendered  # Indentation should be removed
        
    def test_preserve_relative_indentation(self):
        """Test that relative indentation is preserved."""
        b = Block()
        b.append("""
            First level
                Second level
                    Third level
            Back to first
        """)
        
        rendered = b.render()
        assert "First level" in rendered
        assert "    Second level" in rendered
        assert "        Third level" in rendered
        assert "Back to first" in rendered


if __name__ == "__main__":
    pytest.main(["-v", "test_block3.py"]) 