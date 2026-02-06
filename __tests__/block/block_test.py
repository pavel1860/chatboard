from chatboard.block import Block, BlockSent, BlockChunk





def test_iadd_block_to_block():
    b1 = Block("hello")
    b2 = Block("world")
    b1 += b2
    assert len(b1.root) == 2
    assert b1.root[0].content == "hello"
    assert b1.root[1].content == "world"
    assert b1.root.render() == "hello world "
    
    
    
def test_add_block_to_block():
    b1 = Block()
    b2 = Block()
    b3 = b1 + b2
    assert len(b3.root) == 2
    assert b3.root[0].content == "hello"
    assert b3.root[1].content == "world"
    assert b3.root.render() == "hello world "
    
    
    
def test_simple_context():
    with Block("title") as b:
        b += "content"
    
    with Block("title") as b1:
        b1 += "hello"
        b1 += "world"

    assert len(b1.root) == 1
    assert b1.root[0].content == "title"
    assert len(b1.children) == 1
    assert b1.children[0][0].content == "hello"
    assert b1.children[0][1].content == "world"


    with Block("title") as b2:
        b2 /= "hello"
        b2 /= "world"


    assert len(b2.root) == 1
    assert b2.root[0].content == "title"
    assert len(b2.children) == 2
    assert b2.children[0][0].content == "hello"
    assert b2.children[1][0].content == "world"
    
    
    with Block("title") as b3:
        b3 += "hello\n"
        b3 += "world"

    assert len(b3.root) == 1
    assert len(b3.children) == 2
    assert b3.children[0][0].content == "hello"
    assert b3.children[1][0].content == "world"

    
    
    