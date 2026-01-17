import pytest
import pytest_asyncio
from promptview.model import NamespaceManager
from promptview.utils.db_connections import PGConnectionManager
import textwrap
import random

@pytest_asyncio.fixture(scope="function")
async def test_db_pool():
    """Create an isolated connection pool for each test."""
    # Close any existing pool
    if PGConnectionManager._pool is not None:
        await PGConnectionManager.close()
    
    # Create a unique pool for this test
    await PGConnectionManager.initialize(
        url=f"postgresql://ziggi:Aa123456@localhost:5432/promptview_test"
    )
    
    yield
    
    # Clean up this test's pool
    await PGConnectionManager.close()

@pytest_asyncio.fixture()
async def clean_database(test_db_pool):
    # Now uses an isolated pool
    await NamespaceManager.recreate_all_namespaces()
    # NamespaceManager.drop_all_namespaces()
    yield
    # Don't recreate namespaces during teardown to avoid PostgreSQL type cache issues
    # await NamespaceManager.recreate_all_namespaces()
    
    
    
    
    
    
    
    
    
    
    
def chunk_xml_for_llm_simulation(xml_str: str, seed: int | None = None) -> list[str]:
    """
    Break an XML string into irregular chunks simulating LLM streaming output.
    """
    if seed is not None:
        random.seed(seed)
    
    chunks = []
    i = 0
    
    while i < len(xml_str):
        if xml_str[i] == '<':
            # "<" usually comes as its own token
            chunks.append('<')
            i += 1
            
            # Check for closing tag slash
            if i < len(xml_str) and xml_str[i] == '/':
                chunks.append('/')
                i += 1
            
            # Tag name - sometimes split, sometimes whole
            tag_name = ''
            while i < len(xml_str) and xml_str[i] not in ' >\t\n/':
                tag_name += xml_str[i]
                i += 1
            if tag_name:
                chunks.extend(random_split(tag_name))
            
            # Handle attributes
            while i < len(xml_str) and xml_str[i] != '>':
                # Whitespace
                ws = ''
                while i < len(xml_str) and xml_str[i] in ' \t\n':
                    ws += xml_str[i]
                    i += 1
                if ws:
                    chunks.append(ws)
                
                # Attribute name
                attr_name = ''
                while i < len(xml_str) and xml_str[i] not in '= >\t\n':
                    attr_name += xml_str[i]
                    i += 1
                if attr_name:
                    chunks.extend(random_split(attr_name))
                
                # "=" sign
                if i < len(xml_str) and xml_str[i] == '=':
                    chunks.append('=')
                    i += 1
                
                # Attribute value with quotes
                if i < len(xml_str) and xml_str[i] == '"':
                    chunks.append('"')
                    i += 1
                    val = ''
                    while i < len(xml_str) and xml_str[i] != '"':
                        val += xml_str[i]
                        i += 1
                    if val:
                        chunks.extend(random_split(val))
                    if i < len(xml_str) and xml_str[i] == '"':
                        chunks.append('"')
                        i += 1
            
            # ">" usually its own token
            if i < len(xml_str) and xml_str[i] == '>':
                chunks.append('>')
                i += 1
        else:
            # Text content
            text = ''
            while i < len(xml_str) and xml_str[i] != '<':
                text += xml_str[i]
                i += 1
            if text:
                chunks.extend(chunk_text_llm_style(text))
    
    return chunks


def random_split(s: str) -> list[str]:
    """Randomly decide to split a string or keep it whole."""
    if len(s) <= 2 or random.random() > 0.3:
        return [s]
    # Split at a random point
    split_point = random.randint(1, len(s) - 1)
    return [s[:split_point], s[split_point:]]


def chunk_text_llm_style(text: str) -> list[str]:
    """Split text content like LLM tokens - often on word boundaries with leading spaces."""
    if not text:
        return []
    
    chunks = []
    i = 0
    
    while i < len(text):
        chunk = ''
        
        # Leading whitespace often attached to next word
        while i < len(text) and text[i] in ' \t\n':
            chunk += text[i]
            i += 1
        
        # Get some word characters
        word_len = random.randint(1, 6)
        count = 0
        while i < len(text) and text[i] not in ' \t\n<' and count < word_len:
            chunk += text[i]
            i += 1
            count += 1
        
        if chunk:
            chunks.append(chunk)
    
    return chunks



def strip_text(text: str) -> str:
    return textwrap.dedent(text).strip()




def print_event(ev, split: bool = False):
    if split:
        print(f"---------------{ev.type}-----------------")
    if ev.type == "llm_start":
        print(ev.payload)
    elif ev.type == "llm_delta":
        be = ev.payload
        # if be.type == "block_init":
        print(be.type,be.path, be.value)        
    else:
        print(ev.type, ev.payload)