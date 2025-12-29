import pytest
import pytest_asyncio
from promptview.block.block11 import Block, BlockSchema
from promptview.block.block11.span import BlockChunk, Span
from promptview.block.block11.parsers import XmlParser
from promptview.block.block11.schema import BlockSchema, BlockListSchema
from promptview.prompt.fbp_process import Stream
from pydantic import BaseModel, Field
from ..helpers import chunk_xml_for_llm_simulation
import textwrap





class SchemaParsingEval:
    source: str = ""
    target: str = ""
    filepath: str = ""
    
    def block(self) -> Block:
        raise NotImplementedError("Subclasses must implement this method")
    
    def schema(self) -> BlockSchema:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_target(self) -> str:
        return textwrap.dedent(self.target).strip()
    
    def get_source(self) -> str:
        return textwrap.dedent(self.source).strip()
    
    def get_filepath(self) -> str:
        return self.filepath
            
    async def parse(self) -> Block:
        if filepath:=self.get_filepath():
            stream = Stream.load(filepath)
        elif source:=self.get_source():
            chunks = chunk_xml_for_llm_simulation(source)
            stream = Stream.from_list(chunks, name="stream_from_list")
        else:
            raise ValueError("No source or filepath provided")        
        schema = self.schema()
        parser = XmlParser(schema)        
        pipe = stream | parser
        async for ip in pipe:
            pass
        return pipe.result
    
    @pytest.mark.asyncio
    async def test_assert_parsing(self):
        block = await self.parse()
        target = self.get_target()
        rdr = block.render().replace("\\n", "\n")
        assert rdr == target
















class AddBlock(BaseModel):
    """ use this action if the user wants to write something new to the document. or you think you need to add something new to the document."""
    block_path: str = Field(description="the path to the block you will add", json_schema_extra={"example": "1.3.2.1"} )
    content: str = Field(description="the content of the block you will add. use \\n in between lines.", json_schema_extra={"example": "this is a new block for the document"})
    
class DeleteBlock(BaseModel):
    """ use this action if the user wants to delete a block from the document."""
    block_path: str = Field(description="the path to the block you will delete", json_schema_extra={"example": "1.3.2.1"} )
    
class MoveBlock(BaseModel):
    """ use this action if the user wants to move a block to a new location in the document."""
    block_path: str = Field(description="the path to the block you will move", json_schema_extra={"example": "1.3.2.1"} )
    new_path: str = Field(description="the new path to the block", json_schema_extra={"example": "1.3.2.2"} )
    
class EditBlock(BaseModel):
    """ use this action if the user wants to edit a block in the document."""
    block_path: str = Field(description="the path to the block you will edit", json_schema_extra={"example": "1.3.2.1"} )
    content: str = Field(description="the content of the block you will edit. use \\n in between lines.", json_schema_extra={"example": "this is a new block for the document"})





class TestBlockEditorXmlParsing(SchemaParsingEval):
    filepath = "__tests__/data/writer_agent.jsonl"
    target = """
<thought>
The user wants to write a poem about pirates. The current content of the block is very minimal and not structured like a poem. I will replace the existing content with a more detailed and poetic description of pirates.
</thought>
<answer>
Here is a pirate poem for you.
</answer>
<tool name="edit_block">
  <block_path>
    0.0
  </block_path>
  <content>
    Upon the restless ocean's crest,\n
    A pirate's life, they say, is best.\n
    With sails unfurled and flags of black,\n
    They roam the seas, no turning back.\n\n
    Their eyes as sharp as the cutlass blade,\n
    Their spirits fierce, never to fade.\n
    The salty breeze their lullaby,\n
    As they chase the treasure that's ne'er dry.\n\n
    Under the moon's silvery light,\n
    They sing their shanties in the night.\n
    With rum in hand and tales to spin,\n
    The pirate's heart, a world within.\n\n
    So here's to the rogues who sail the seas,\n
    With daring hearts and lives of ease.\n
    For in their world of salt and spray,\n
    A pirate's life is a grand array.
  </content>
</tool>
"""
    
    def schema(self) -> BlockSchema:
        with Block.schema_view(tags=["output","writer"]) as schema:
            with schema.view("thought", str) as t:
                t /= "think step by step what to do with the information you observed and thinkg if there is a tool that can help you to complete the tasks."
            with schema.view("answer", str) as r:
                r /= "the answer to the user's question goes here. you should use is to give textual answer to the user's question."        
            with schema.view_list("tool", key="name") as actions:            
                actions.register(EditBlock)
                actions.register(AddBlock)
                actions.register(DeleteBlock)
                actions.register(MoveBlock)
        return schema