from chatboard.block import Block
from chatboard.block.block9.block_schema import BlockBuilderContext




def test_simple_schema():
    with Block(role="system") as sys:
        sys /= "you are a pirate. you talk like a pirate"
        with sys("Task", tags=["task"]) as task:
            task /= "you must use the following format for the output"
            with task("sub task", tags=["sub task"]) as sub_task:
                sub_task /= "you must use the following format for the output"
                with sub_task.view("output", str) as out:
                    out /= "you must use the following format for the output"
                    with out.view("thoughts", list) as th:
                        th /= "your thoughts go here, you need to think about the user's question in different ways"
                    with out.view("answer", str) as a:
                        a /= "your answer goes here"


    ctx = BlockBuilderContext(out)

    def get_events_types(ctx: BlockBuilderContext):
        events = []
        while ctx.has_events():
            evt = ctx.get_event()
            events.append(evt['_type'])
        return events
        

    assert ctx.has_events() == False
    ctx.append("thoughts", "Hello ")
    assert ctx.has_events() == True
    assert get_events_types(ctx) == ["Block", "Block"]
    ctx.append("thoughts", "World ")
    assert ctx.has_events() == True
    assert get_events_types(ctx) == ["BlockChunk"]
    ctx.append("thoughts", "1\n")
    assert ctx.has_events() == True
    assert get_events_types(ctx) == ["BlockChunk"]
    assert ctx.has_events() == False