import textwrap
import random
from promptview.block.block12.block import BlockChunk



def assert_events(events):  
    
    def validate_existing(blk, idx):
        try:
            b = blk[idx]
            return True
        except:
            return False
    path_lookup = {}
    root = None
    index = -1
    last_block_event = None
    is_first_chunk_in_block = False
    for ev in events:
        try:
            index += 1
            # print(f"{index:03d}: [{ev.path}] {ev.type}")
            print(f"{index}: [{ev.path}] {ev.type}     {ev.value}")
            # if root is not None:
                # assert len(ev.path) > 0
                
            if ev.type == "block_init":                
                if len(ev.value.body) > 0:
                    raise ValueError(f"Block has body at {ev.int_path}")
                # handle root block
                is_first_chunk_in_block = True
                if root is None:
                    root = ev.get_value()
                    continue               
                last_block_event = ev
                assert ev.path not in path_lookup, f"duplicate path {ev.path}"
                path_lookup[ev.path] = "init"
                if validate_existing(root, ev.int_path):
                    raise ValueError(f"Block already exists at {ev.int_path}")        
                root.insert_child(ev.int_path, ev.get_value())
                
            elif ev.type == "block_delta":
                if not validate_existing(root, ev.int_path):
                    raise ValueError(f"Block does not exist at {ev.int_path}")  
                if last_block_event is not None and ev.path != last_block_event.path:      
                    raise ValueError(f"Chunk has different path from block: {ev.path} != {last_block_event.path}")
                target = root[ev.int_path]
                
                if is_first_chunk_in_block and last_block_event is not None:
                    if last_block_event.get_value().text == ev.get_value()[0].content:
                        raise ValueError(f"First Chunk is the same as the block content: {last_block_event.get_value()} == {ev.get_value()[0]}")
                    
                target.append(ev.get_value())
                is_first_chunk_in_block = False
                            
            elif ev.type == "block_commit":
                if ev.path:
                    assert ev.path in path_lookup, f"commit path '{ev.path}' not found"                
                    path_lookup[ev.path] = "commit"
                is_first_chunk_in_block = False
                last_block_event = None
            elif ev.type == "block":
                if len(ev.value.body) > 0:
                    raise ValueError(f"Block has body at {ev.int_path}")
                if validate_existing(root, ev.int_path):
                    raise ValueError(f"Block already exists at {ev.int_path}")
                is_first_chunk_in_block = True
                last_block_event = ev
                root.insert_child(ev.int_path, ev.get_value())
            else:
                raise ValueError(f"Unknown event type: {ev.type}")
        except Exception as e:
            sep = "─" * 50
            print("################ ERROR Details ################")
            print(sep)
            if root is not None:            
                print("Root Block:")
                print(sep)
                root.print_debug()
                print(sep)
            print(f"Error on Event: {ev.type} {ev.int_path}")
            print(ev.value)
            
            raise e
    for path, status in path_lookup.items():
        assert status == "commit", f"path '{path}' not committed"
    
    
def chunk_xml_for_llm_simulation(xml_str: str, seed: int | None = None, as_chunks: bool = False) -> list[str]:
    """
    Break an XML string into irregular chunks simulating LLM streaming output.
    """
    xml_str = strip_text(xml_str)
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
    if as_chunks:
        return [BlockChunk(content=chunk) for chunk in chunks]
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

        
        
        
# def validate_events(events):
#     path_lookup = {}
#     for e in events:
#         if e.type == "llm_delta":
#             pe = e.payload
#             if pe.type == "block_init":
#                 # if pe.path in path_lookup:
#                     # print(f"duplicate path {pe.path}")
#                 # else:
#                 assert pe.path not in path_lookup, f"duplicate path {pe.path}"
#                 path_lookup[pe.path] = "init"
#             elif pe.type == "block_commit":
#                 assert pe.path in path_lookup, f"commit path {pe.path} not found"
#                 path_lookup[pe.path] = "commit"
#     for path, status in path_lookup.items():
#         assert status == "commit", f"path {path} not committed"