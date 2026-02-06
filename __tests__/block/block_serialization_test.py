from chatboard.block import BlockChunk


def recursive_validate(blk1: BlockChunk, blk2: BlockChunk):
    assert blk1.render() == blk2.render()
    assert blk1.tags == blk2.tags
    assert blk1.inline_style.style == blk2.inline_style.style
    for item1, item2 in zip(blk1.items, blk2.items):
        recursive_validate(item1, item2)



def test_simple_block_serialization():

    with BlockChunk("Header", tags=["tag1", "tag2"], style=["list"]) as blk:
        blk /= "rule 1"
        blk /= "rule 2"
        blk /= "rule 3"


    print(blk.render())
    d = blk.model_dump()    
    blk2 = BlockChunk.model_validate(d)
    # blk2.render()
    print(blk2.render())

    recursive_validate(blk, blk2)


def test_complex_block_serialization():
    with BlockChunk("Header", style=["md"]) as blk:
        blk /= "some description"
        with blk("list 1", style=["list"]):
            blk /= "item 1"
            blk /= "item 2"
            blk /= "item 3"
        with blk("Obeservation", style=["xml"]):
            with blk("description", style=["xml"]):       
                blk /= "some description"
                blk /= "some description"
                blk /= "some description"
            with blk("Thought", style=["xml"]):
                blk /= "some thought"
            with blk("Message", style=["xml"]):
                blk /= "some message"
            with blk("Wait for response", style=["xml"]):
                blk /= "some wait for response"
                
    d = blk.model_dump()
    blk2 = BlockChunk.model_validate(d)
    print(blk2.render())

    recursive_validate(blk, blk2)
    
    
def test_very_complex_block_serialization():
    with BlockChunk() as blk:
        with blk(tags=["system"], role="system"):
            blk /= "you are a helpful assistant"
            with blk("Task"):
                blk /= "you need to help the user to find the best asset for them."
            with blk("Rules", style=["list"]):
                blk /= "never talk about fight club"
                blk /= "never talk about fight club!"
                blk /= "never talk about fight club?"
        with blk(tags=["history"], role="user"):
            blk /= "hello"
        with blk(tags=["history"], role="assistant"):
            blk /= "hello"
        with blk(tags=["history"], role="user"):
            blk /= "who are you?"
        with blk(tags=["history"], role="tool"):
            blk /= "I am skynet"
        with blk(tags=["history"], role="user"):
            blk /= "what can you do?"
        with blk(tags=["history"], role="assistant"):
            blk /= "I can search the weather"
        with blk(tags=["user_input"], role="user"):
            blk /= "what is the weather in new york?"
        with blk(tags=["turn_history"], role="assistant"):
            with blk("Observation"):
                blk /= "the user is asking for the weather in new york"
            with blk("Action"):
                blk /= "search_weather"
        
        with blk(tags=["turn_history"], role="tool"):
            blk /= "the weather in new york is sunny"
        with blk(tags=["turn_history"], role="assistant"):
            with blk("Observation"):
                blk /= "the tool has returned the weather in new york"
            with blk(tags=["message"]):
                blk /= "the weather in new york is sunny"
                
                
                
    d = blk.model_dump() 
    blk2 = BlockChunk.model_validate(d)
    recursive_validate(blk, blk2)