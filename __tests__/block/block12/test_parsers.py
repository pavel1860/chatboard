"""Tests for XmlParser."""
import pytest
from promptview.block.block12 import Block, XmlParser
from __tests__.utils import chunk_xml_for_llm_simulation, strip_text
class TestXmlParserBasic:
    """Tests for basic XML parsing."""

    @pytest.mark.asyncio
    async def test_parse_simple_xml(self):
        from promptview.prompt.fbp_process import Stream

        xml_str =  strip_text("<item>hello</item>")
        chunks = chunk_xml_for_llm_simulation(xml_str, seed=42)

        with Block(role="system") as blk:
            blk /= 'you need to answer a question'
            with blk.view("item", tags=["item"]) as schema:
                schema /= 'the item content'

        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for _ in pipe:
            pass

        assert pipe.result.render() == xml_str
        
    @pytest.mark.asyncio
    async def test_parse_simple_xml_with_output_format(self):
        from promptview.prompt.fbp_process import Stream

        xml_str = strip_text("""
        <output>
            <item>hello</item>
        </output>
        """)

        chunks = chunk_xml_for_llm_simulation(xml_str, seed=42)

        with Block(role="system") as blk:
            blk /= 'you need to answer a question'
            blk /= "use the following format to answer the question"    
            with blk.view("output", tags=["output"]) as schema:
                with schema.view("item", tags=["item"]) as item:
                    item /= 'the item you want you need to create'
        
        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for ip in pipe:
            pass

        assert pipe.result.render() == xml_str        
        
    @pytest.mark.asyncio
    async def test_parse_with_code_fence(self):
        from promptview.prompt.fbp_process import Stream

        xml_str = """```xml
<item>hello</item>
```"""
        chunks = list(xml_str)

        with Block(role="system") as blk:
            blk /= 'you need to answer a question'
            with blk.view("item") as schema:
                schema /= 'the item content'

        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for _ in pipe:
            pass

        assert pipe.result.render() == xml_str


class TestXmlParserMultipleTags:
    """Tests for parsing multiple XML tags."""

    @pytest.mark.asyncio
    async def test_parse_thought_answer(self):
        from promptview.prompt.fbp_process import Stream

        with Block(tags=["schema"]) as out_fmt:
            with out_fmt("output schema", tags=["schema", "pirate"]) as schema:    
                with schema.view("thought", str, tags=["pirate"]) as t:
                    t /= "think step by step what to do with the information you observed and thinkg if there is a tool that can help you to complete the tasks."
                with schema.view("answer", str, tags=["pirate"]) as r:
                    r /= "the answer to the user's question goes here"


        xml_str = strip_text("""
        <thought>lets say hello world</thought>
        <answer>hello world</answer>
        """)

        chunks = chunk_xml_for_llm_simulation(xml_str, seed=42)

        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for ip in pipe:
            # print(ip)
            pass

        blk = pipe.result

        assert blk[0].head.text == "<thought>"
        assert len(blk[0,0].content) > 0
        assert blk[0].tail.text == "</thought>\n"

        assert blk[1].head.text == "<answer>"
        assert len(blk[1,0].content) > 0
        assert blk[1].tail.text == "</answer>"
        assert blk.render() == xml_str


        xml_str = strip_text("""
        <thought>
            lets say hello world
        </thought>
        <answer>
            hello world
        </answer>
        """)

        chunks = chunk_xml_for_llm_simulation(xml_str, seed=42)

        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for ip in pipe:
            # print(ip)
            pass

 

        blk = pipe.result


        assert blk[0].head.text == "<thought>\n"
        assert len(blk[0,0].content) > 0
        assert blk[0].tail.text == "</thought>\n"

        assert blk[1].head.text == "<answer>\n"
        assert len(blk[1,0].content) > 0
        assert blk[1].tail.text == "</answer>"
        assert blk.render() == xml_str



class TestXmlParserWithTools:
    """Tests for parsing XML with tool attributes."""

    @pytest.mark.asyncio
    async def test_parse_with_tool(self):
        from promptview.prompt.fbp_process import Stream
        from pydantic import BaseModel, Field

        class Attack(BaseModel):
            """Use this tool to attack."""
            weapon: str = Field(description="the weapon")
            target: str = Field(description="the target")

        xml_str = """<thought>lets say hello world</thought>
<answer>hello world</answer>
<tool name="attack">
    <weapon>sword</weapon>
    <target>head</target>
</tool>"""
        chunks = list(xml_str)

        with Block(tags=["schema"]) as out_fmt:
            with out_fmt("output schema", tags=["schema"]) as schema:
                with schema.view("thought", str) as t:
                    t /= "think step by step"
                with schema.view("answer", str) as r:
                    r /= "the answer goes here"
                with schema.view_list("tool", key="name") as tools:
                    tools.register(Attack)

        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for _ in pipe:
            pass

        assert pipe.result.render() == xml_str


class TestXmlParserStreaming:
    """Tests for streaming XML parsing behavior."""

    @pytest.mark.asyncio
    async def test_chunked_parsing(self):
        from promptview.prompt.fbp_process import Stream

        chunks = ['<', 'item', '>', 'hello', ' wo', 'rld', '<', '/', 'item', '>']

        with Block(role="system") as blk:
            with blk.view("item", tags=["item"]) as schema:
                schema /= 'content'

        parser = XmlParser(schema)
        stream = Stream.from_list(chunks, name="stream_from_list")

        pipe = stream | parser
        async for _ in pipe:
            pass

        assert pipe.result is not None
        assert "hello world" in pipe.result.render()
