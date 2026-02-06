from chatboard.block import Block, BlockSent, BlockChunk







def test_sent_building():
    c1 = BlockChunk("hello")
    c2 = BlockChunk("world")
    c3 = BlockChunk("!")

    s = BlockSent([c1, c2, c3])

    for i,c in enumerate(s):
        assert c.index == i
        


def test_sent_append():    
    c1 = BlockChunk("hello")
    c2 = BlockChunk("world")
    c3 = BlockChunk("!")
    
    s = BlockSent()
    s.append(c1)
    assert len(s) == 1
    assert s[0].index == 0
    assert s[0].content == "hello"
    s.append(c2)
    assert len(s) == 2
    assert s[0].index == 0
    assert s[0].content == "hello"
    assert s[1].index == 1
    assert s[1].content == "world"
    s.append(c3)
    assert len(s) == 3
    assert s[0].index == 0
    assert s[0].content == "hello"
    assert s[1].index == 1
    assert s[1].content == "world"
    assert s[2].index == 2
    assert s[2].content == "!"
    
    
    
def test_sent_prepend():
    c1 = BlockChunk("hello")
    c2 = BlockChunk("world")
    c3 = BlockChunk("!")
    
    s = BlockSent([c2])
    assert len(s) == 1
    assert s[0].index == 0
    assert s[0].content == "world"
    s.append(c3)
    assert len(s) == 2
    assert s[0].index == 0
    assert s[0].content == "world"
    assert s[1].index == 1
    assert s[1].content == "!"
    s.prepend(c1)
    assert len(s) == 3
    assert s[0].index == 0
    assert s[0].content == "hello"
    assert s[1].index == 1
    assert s[1].content == "world"
    assert s[2].index == 2
    assert s[2].content == "!"
    
    
def test_sent_add():
    c1 = BlockChunk("hello")
    c2 = BlockChunk("world")
    c3 = BlockChunk("!")

    s = BlockSent([c2])

    s += c3
    assert len(s) == 2
    assert s[0].content == "world"
    assert s[1].content == "!"
    assert s.render() == "world ! "
    s1 = c1 + s
    assert len(s1) == 3
    assert s1[0].content == "hello"
    assert s1[1].content == "world"
    assert s1[2].content == "!"
    assert s1.render() == "hello world ! "


def test_sent_add_sent():
    s1 = BlockSent(["world", "!"])
    s2 = BlockSent(["hello"])

    s3 = s2 + s1
    assert len(s3) == 3
    assert s3[0].content == "hello"
    assert s3[1].content == "world"
    assert s3[2].content == "!"
    assert s3.render() == "hello world ! "
    assert s2.render() == "hello "
    assert s1.render() == "world ! "


def test_sent_iadd_sent():
    s1 = BlockSent(["world", "!"])
    assert s1[0].content == "world"
    assert s1[1].content == "!"

    s2 = BlockSent(["hello"])

    s2 += s1
    assert len(s2) == 3
    assert s2[0].content == "hello"
    assert s2[1].content == "world"
    assert s2[2].content == "!"
    
    
    
    