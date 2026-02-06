import pytest
from chatboard.prompt.legacy.block2 import StrBlock, TitleBlock, Block

def test_str_block_basic():
    # Test basic string block with different input formats
    blk1 = StrBlock("this is a test")
    assert blk1.render() == "this is a test"

    blk2 = StrBlock("""
     this is a test
     """)
    assert blk2.render() == "this is a test"

    blk3 = StrBlock("""
    this is a test
    """)
    assert blk3.render() == "this is a test"

def test_title_block_with_append():
    # Test title block with append method
    b = TitleBlock("this is a test", type="md")
    b.append("this is a test")
    b.append("this is a test")
    b.append("this is a test")
    b.append("this is a test")

    target = """
# this is a test
this is a test
this is a test
this is a test
this is a test
""".strip()

    assert b.render() == target

def test_title_block_with_context():
    # Test title block with context manager
    target = """
# this is a test
this is a test
this is a test
this is a test
this is a test
""".strip()

    with TitleBlock("this is a test", type="md") as b:
        StrBlock("this is a test")
        StrBlock("this is a test")
        StrBlock("this is a test")
        StrBlock("this is a test")
    
    assert b.render() == target

def test_nested_title_blocks():
    # Test nested title blocks with different depths
    target_nested = """
# main title
this is a test
## sub title
this is a test
### sub sub title
this is a test
this is a test
""".strip()

    block = Block()
    # Test with direct string addition
    with block.title("main title", "md") as sb1:
        block("this is a test")
        with block.title("sub title", "md") as sb2:
            block("this is a test")
            with block.title("sub sub title", "md") as sb3:    
                block("this is a test")
                block("this is a test")
    
    assert sb1.render() == target_nested

    # Test with += operator
    with block.title("main title", "md") as sb1:
        block("this is a test")
        with block.title("sub title", "md") as sb2:
            block += "this is a test"
            with block.title("sub sub title", "md") as sb3:    
                block += "this is a test"
                block("this is a test")

    assert sb1.render() == target_nested

def test_list_blocks_with_different_bullets():
    # Test list blocks with different bullet types
    block = Block()
    
    # Test with number bullets
    with block.title("main title", "md") as sb1:
        block.li += "this is a test1"
        block.li += "this is a test2"
        block.li += "this is a test3"

    assert sb1.render() == """
# main title
1. this is a test1
2. this is a test2
3. this is a test3
""".strip()

    # Test with alpha bullets
    with block.title("main title", "md", bullet="alpha") as sb1:
        block.li += "this is a test1"
        block.li += "this is a test2"
        block.li += "this is a test3"
    
    assert sb1.render() == """
# main title
a. this is a test1
b. this is a test2
c. this is a test3
""".strip()

    # Test with roman numerals
    with block.title("main title", "md", bullet="roman") as sb1:
        block.li += "this is a test1"
        block.li += "this is a test2"
        block.li += "this is a test3"

    assert sb1.render() == """
# main title
i. this is a test1
ii. this is a test2
iii. this is a test3
""".strip() 