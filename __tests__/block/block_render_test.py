import pytest

from chatboard.block.block import Block



def assert_render(block, target):
	render_out = block.render()
	print(render_out)
	assert render_out == target



def test_col_row_render():
    with Block("say") as blk:
        blk /= "Hello"
        blk /= "World"

    assert_render(blk, (
        "say\n"
        "   Hello\n"
        "   World"
    ))
    
    with Block("say", style="md") as blk:
        blk /= "Hello"
        blk /= "World"
        
    assert_render(blk, (
        "# say\n"
        "Hello\n"
        "World"
    ))
        
    with Block("say", style="md") as blk:
        blk += "Hello"
        blk += "World"

    assert_render(blk, (
        "# say\n"
        "Hello World"
    ))
    # row append should be in the same line
    with Block("say", style="md") as blk:
        blk += "Hello", "World"

    assert_render(blk, (
        "# say\n"
        "Hello World"
    ))
    
    with Block("say", style="md") as blk:
        blk /= "Hello", "World"

    assert_render(blk, (
        "# say\n"
        "Hello World"
    ))


def test_basic_render():
    
    with Block() as r:
        r += "you should speak like a pirate"
    
    assert_render(r, "you should speak like a pirate")
    
    
    with Block("Pirate", style="md") as r:
        r += "you should speak like a pirate"
    
    assert_render(r, "# Pirate\nyou should speak like a pirate")
    
    with Block("Pirate", style=["xml"]) as r:
        r += "you should speak like a pirate"
    
    assert_render(r, "<Pirate>\n  you should speak like a pirate\n</Pirate>")

def test_basic_topic():
    with Block("Tasks", style="md") as blk:
        blk /= "go to the store"
        blk /= "buy the groceries"
        blk /= "call the bank"
        blk /= "call the lawyer"
        
    assert_render(blk, """# Tasks
go to the store
buy the groceries
call the bank
call the lawyer""")


def test_basic_list():
    with Block("Tasks", style="md list") as blk:
        blk /= "go to the store"
        blk /= "buy the groceries"
        blk /= "call the bank"
        blk /= "call the lawyer"
        
        assert_render(blk, """# Tasks
1. go to the store
2. buy the groceries
3. call the bank
4. call the lawyer""")


def test_wrapper_block():
    with Block(tags=["system"]) as b:
        b /= "you are a helpful assistant"
        with b("Task", style="md", tags=["task"]):
            b /= "this is task you need to complete"

    assert_render(b, "you are a helpful assistant\n# Task\nthis is task you need to complete")

    


def test_indentation():
    s = Block("""
        you are a helpful assistant
        Task:
            this is task you need to complete
        Rules:
        - you need to complete the task
        - you need to follow the rules
        Output Format:
        - you need to output the task
        - you need to output the rules
        - you need to output the output format
    """)
    target = """you are a helpful assistant
Task:
    this is task you need to complete
Rules:
- you need to complete the task
- you need to follow the rules
Output Format:
- you need to output the task
- you need to output the rules
- you need to output the output format"""

    assert_render(s, target)
    
    
    
    
    
    
    
def test_get_block():
    with Block(tags=["system"], style=["xml"]) as b:
        b += "you are a helpful assistant"
        with b("Task", tags=["task"]):
            b += "this is task you need to complete"
        
        with b("Rules", tags=["rules"], style=["list"]):
            b += "this is rule 1"
            with b("this is main rule 2"):
                b += "create this 1"
                b += "create this 2"
                b += "create this 3"
            b += "this is rule 3"
            b += "this is rule 4"
        with b("Generation Rules", tags=["rules"]):
            b += "you should speak like a pirate1"
            b += "you should speak like a pirate2"
            with b("this is main rule 2"):
                b += "create this 1"
                with b("this is main rule 2"):
                    b += "create this 1"
                    with b("this is main rule 22"):
                        with b("this is main rule 222", tags=["sub-rule"]):
                            b += "create this 1"
                b += "create this 3"
        with b("Output", tags=["output"]):
            b += "this is output"
            with b("thinking", tags=["thinking"]):
                b += "this is thinking"
            with b("output", tags=["output"]):
                b += "this is output"
            with b("Actions", tags=["actions"]):
                for i in range(4):
                    with b(f"Action {i}"):
                        b += f"this is action {i}"
            

        
    task = b.find("task")
    assert len(task) == 1
    assert task[0].content == "Task"

    rules = b.find("rules")
    assert len(rules) == 2
    assert rules[0].content == "Rules"

    sub_rules = b.find("sub-rule")
    assert len(sub_rules) == 1
    assert sub_rules[0].content == "this is main rule 222"


    multi_rules =b.find(["task", "sub-rule"])

    assert len(multi_rules) == 2
    assert multi_rules[0].content == "Task"
    assert multi_rules[1].content == "this is main rule 222"






