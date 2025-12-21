import pytest
import pytest_asyncio
from promptview.block.block10 import Block, BlockBase, BlockChunk, BlockText, Span, SpanAnchor, XmlParser
from promptview.block.block10.block import BlockSchema, BlockListSchema
from promptview.prompt.fbp_process import Stream
from .helpers import chunk_xml_for_llm_simulation
import textwrap



class TestBlockOperations:
    
    
    
    def test_block_append(self):
        b1 = Block("hello")
        b2 = Block("world")

        b1 = b1 + b2

        assert len(b1) == 0
        assert b1.render() == "hello world"
        
        
    def test_block_append_newlines_as_children(self):
        b1 = Block("hello")
        b2 = Block("world")

        b3 = b1 + b2
        assert len(b3) == 0
        b3 = b1 + b2 + "\n"
        assert len(b3) == 0
        b3 = b1 + b2 + "\n" + "cat"

        assert len(b3) == 1
        b3 = b1 + b2 + "\n" + "cat" + "\n"
        assert len(b3) == 1
        b3 = b1 + b2 + "\n" + "cat" + "\n" + "dog"
        assert len(b3) == 2


class BlockRenderEval:
    target: str = ""
    # def setup_method(self):
    #     self.block = self.block()
        
        
    def block(self):
        raise NotImplementedError("Subclasses must implement this method")
        
    def test_assert_target(self):
        block = self.block()
        rdr = block.render()
        if not self.target:
            raise RuntimeError("Target text not set")
        target_text = textwrap.dedent(self.target).strip()
        assert rdr == target_text








class TestBlockDeepNestedBasic(BlockRenderEval):
    
    target = """
        # Header 1
        sentence 1
        sentence 2
        ## Header 2
        sub sentence 2 1
        sub sentence 2 2
        ### Header 3
        sub sub sentence 3 1
        sub sub sentence 3 2
        """
    
    def block(self):
        with Block("Header 1", style="md") as schema:
            schema /= "sentence 1"
            schema /= "sentence 2"
            with schema("Header 2", style="md") as sub:
                sub /= "sub sentence 2 1"
                sub /= "sub sentence 2 2"
                with sub("Header 3", style="md") as sub2:
                    sub2 /= "sub sub sentence 3 1"
                    sub2 /= "sub sub sentence 3 2"
                    
        return schema
    
    
    
    def test_children_structure(self):
        block = self.block()
        assert len(block.children) == 3
        assert block.content_str == "Header 1"
        assert block.children[0].content_str == "sentence 1"
        assert block.children[1].content_str == "sentence 2"
        assert block.children[2].content_str == "Header 2"
        assert len(block.children[2].children) == 3
        assert block.children[2].children[0].content_str == "sub sentence 2 1"
        assert block.children[2].children[1].content_str == "sub sentence 2 2"
        assert block.children[2].children[2].content_str == "Header 3"
        assert len(block.children[2].children[2].children) == 2
        assert block.children[2].children[2].children[0].content_str == "sub sub sentence 3 1"
        assert block.children[2].children[2].children[1].content_str == "sub sub sentence 3 2"
        



class TestBlockXml(BlockRenderEval):
    
    target = """
        <Hello world>
        this is a test
        </Hello world>
        """
    
    def block(self):
        with Block("Hello world", style="xml")  as schema:            
            schema /= "this is a test"
        return schema
    
    
    def test_block_structure(self):
        block = self.block()
        assert len(block) == 1
        assert block.content_str == "Hello world"
        assert len(block.children) == 1
        assert block.children[0].content_str == "this is a test"
        




class SchemaParsingEval:
    source: str = ""
    target: str = ""
    
    def block(self) -> BlockBase:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_target(self) -> str:
        return textwrap.dedent(self.target).strip()
    
    def get_source(self) -> str:
        return textwrap.dedent(self.source).strip()
            
    async def parse(self) -> Block:
        chunks = chunk_xml_for_llm_simulation(self.get_source())
        schema = self.block()
        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")
        pipe = stream | parser
        async for ip in pipe:
            pass
        return pipe.result
    
    @pytest.mark.asyncio
    async def test_assert_parsing(self):
        block = await self.parse()
        target = self.get_target()
        rdr = block.render()
        assert rdr == target
        
        
        
class TestBlockXmlParsing(SchemaParsingEval):
    source = """
    <item>hello world</item>
    """
    
    target = """
    <item>
    hello world
    </item>
    """
    
    def block(self) -> BlockBase:
        with Block(role="system") as blk:
            blk /= 'you need to answer a question'
            blk /= "use the following format to answer the question"    
            with blk.view("item") as schema:
                schema /= 'the item you want you need to create'
                
            with Block() as um:
                um /= "hello"
        return blk
    
    @pytest.mark.asyncio
    async def test_output_structure(self):
        block = self.block()
        schema = block.extract_schema()
        assert len(schema) == 1
        assert schema.content_str == "item"
        assert len(schema.children) == 1        
        
        
