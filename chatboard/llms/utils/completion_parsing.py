import inspect
import re
from enum import Enum
from typing import Optional, Union, get_args, get_origin, get_type_hints

import yaml


def sanitize_text(text: str):
    return text.strip().strip("'").strip('"')

class PromptParsingException(Exception):
    pass


def parse_bool(text):
    text = sanitize_text(text)
    if text.lower() in ["true", "yes", "1"]:
        return True
    elif text.lower() in ["false", "no", "0"]:
        return False
    else:
        raise PromptParsingException(f"Value {text} is not a valid boolean")






def parse_model_list(completion, pydantic_model, delimiter=","):
    rows = [r for r in completion.split("\n") if r != ""]
    segments = []
    for row in rows:
        row_split = row.split(delimiter)
        if len(row_split) != len(pydantic_model.__fields__.keys()):
            continue
            # raise PromptParsingException(f"Row {row} does not have the correct number of fields")
        for i, (field_name, field_info) in enumerate(pydantic_model.__fields__.items()):
            # if field_info.type_ == str:
            #     row_split[i] = sanitize_text(row_split[i])
            # elif field_info.type_ == float:
            #     row_split[i] = float(row_split[i])
            # elif field_info.type_ == int:
            #     row_split[i] = int(row_split[i])        
            if field_info.annotation == str:
                row_split[i] = sanitize_text(row_split[i])
            elif field_info.annotation == float:
                row_split[i] = float(row_split[i])
            elif field_info.annotation == int:
                row_split[i] = int(row_split[i])
            elif field_info.annotation == bool:
                row_split[i] = parse_bool(row_split[i])
            elif inspect.isclass(field_info.annotation) and issubclass(field_info.annotation, Enum):                
                sanitize_content = sanitize_text(row_split[i])
                row_split[i] = field_info.annotation[sanitize_content]                
            elif get_origin(field_info.annotation) == Union:
                if int in get_args(field_info.annotation):
                    try:
                        row_split[i] = int(sanitize_text(row_split[i]))
                        continue
                    except:
                        pass
                if str in get_args(field_info.annotation):
                    row_split[i] = sanitize_text(row_split[i])
                    continue
                
                raise PromptParsingException(f"Field {field_name} is not a valid type")
            else:
                raise PromptParsingException(f"Field {field_name} is not a valid type")

            
        segments.append(pydantic_model(**{k: v for k, v in zip(pydantic_model.__fields__.keys(), row_split)}))
    return segments



def split_rows(text):
    return [r for r in text.split("\n") if r != '']

def sanitize_content(content):
    # content = re.sub(r'\n+', '\n', content)
    # content = re.sub(r'^\n', '', content)
    # content = re.sub(r'\n$', '', content)
    content = content.strip().strip('"').strip("'").strip('\n')
    # content = content.strip('"').strip("'").string('\n')
    return content

def append_output(output, curr_field, content):
    if curr_field:
        # if output[curr_field] is None:
        #     output[curr_field] = ""
        # elif output[curr_field] != "":
        #     output[curr_field] += " "
        # output[curr_field] += sanitize_content(content)
        output[curr_field] = sanitize_content(content)
    return output


def search_field(field, text):
    pattern = f"{field}:"
    # pattern = r"(?i)script: "
    pattern = r"(?i)" + pattern
    results = re.findall(pattern, text)
    if not results:
        return False
    return True

def get_field(field, text):
    pattern = f"{field}:"
    # pattern = r"(?i)script: "
    pattern = r"(?i)" + pattern
    results = re.findall(pattern, text)
    if not results:
        return None
    result = re.sub(pattern, "", text)
    return result

def split_field(field, text):
    pattern = f"{field}:"
    # pattern = r"(?i)script: "
    pattern = r"(?i)" + pattern
    results = re.split(pattern, text)
    if not results:
        return None
    return results


def num_split_field(field, text, maxsplit=1):
    pattern = f"{field}:"
    # Make the pattern case-insensitive
    pattern = r"(?i)" + re.escape(pattern)  # Also escape the pattern to handle special characters
    # Split only on the first instance of the pattern
    results = re.split(pattern, text, maxsplit=maxsplit)
    if len(results) <= 1:
        # If results has 1 or fewer elements, the pattern was not found, or there's nothing after it.
        return None
    return results


def to_dict(pydantic_model):
    d = {}
    for field_name, field_info in pydantic_model.__fields__.items():
        d[field_name] = None
    return d   


def sanitize_item(item_chunk):
    item_raw = yaml.safe_load(item_chunk)
    item = {}
    for k, v in item_raw.items():
        key = k.strip().lower()
        if isinstance(v, str):
            item[key] = v.strip()
        else:
            item[key] = v
    return item

def split_item_list(text, field):
    curr_item_chunk = ""
    chunk_list = []
    for row in [t for t in text.split("\n") if t.strip() != "" and t.strip() != "---"]:
        if field in row.lower():
            if curr_item_chunk != "":
                chunk_list.append(sanitize_item(curr_item_chunk))
            curr_item_chunk = row + "\n"
        else:
            curr_item_chunk += row + "\n"
    if curr_item_chunk != "":
        chunk_list.append(sanitize_item(curr_item_chunk))
    return [c for c in chunk_list if c != {}]


def parse_completion(completion, pydantic_model):
    """Parse completion into a Pydantic model instance.
    the function searchs for the fields in the completion and assigns the content to the corresponding field.
    the completion is split into chunks and the function iterates over the chunks to find the fields.
    the fields can be multi-line and the function will concatenate the content until the next field is found.
    """
    output = ''
    output = to_dict(pydantic_model)
    curr_field = None
    curr_content = ""
    for chunk in split_rows(completion):
        output, curr_field, curr_content = auto_split_row_completion(curr_content, chunk, output, curr_field, pydantic_model)
    else:
        if curr_field and output.get(curr_field) is None:
            output[curr_field] = sanitize_content(curr_content)        
    
    return pydantic_model(**output)


def auto_split_row_completion(curr_content, chunk, output, curr_field, pydantic_model):
    curr_content += chunk
    for field_name, field_info in pydantic_model.__fields__.items():
        if search_field(field_name, curr_content):
            prev_content, curr_content = split_field(field_name, curr_content) #type: ignore
            if curr_field:
                # another field had been found, so we assign the content to the previous field                
                output[curr_field] = sanitize_content(prev_content)
            curr_field = field_name            
            if field_info.annotation == int:                    
                output[field_name] = int(sanitize_content(curr_content))
                curr_field = None
                curr_content = ""
            elif field_info.annotation == float:
                output[field_name] = float(sanitize_content(curr_content))
                curr_field = None
                curr_content = ""
            elif field_info.annotation == bool:
                output[field_name] = parse_bool(sanitize_content(curr_content))
                curr_field = None
                curr_content = ""
    return output, curr_field, curr_content




def auto_split_completion(curr_content, chunk, output, curr_field, pydantic_model):
    curr_content += chunk
    for field_name, field_info in pydantic_model.__fields__.items():
        if search_field(field_name, curr_content):
            prev_content, curr_content = split_field(field_name, curr_content) #type: ignore
            if prev_content and curr_field:
                """another field had been found, so we assign the content to the previous field"""
                output[curr_field] = sanitize_content(prev_content)
            curr_field = field_name
    return output, curr_field, curr_content



def auto_split_list_completion(pydantic_model, curr_content, chunk, output_list=None, curr_field=None):
    is_new_output = False
    if not output_list:
        output = to_dict(pydantic_model)
        output_list.append(output)#type: ignore
    else:
        output = output_list[-1]
    curr_content += chunk
    for field_name, field_info in pydantic_model.__fields__.items():
        if search_field(field_name, curr_content):
            prev_content, curr_content = split_field(field_name, curr_content)#type: ignore
            if prev_content and curr_field:
                if output.get(curr_field) is not None:
                    output = to_dict(pydantic_model)
                    output_list.append(output)#type: ignore
                    is_new_output = True
                output[curr_field] = sanitize_content(prev_content)
            curr_field = field_name
    return output, curr_field, curr_content, is_new_output



def is_list_model(pydantic_model):
    return get_origin(pydantic_model) == list

def unpack_list_model(pydantic_model):
    return get_args(pydantic_model)[0]


def auto_split_completion2(content, output, curr_field, pydantic_model):
    # output = output.copy()
    for field_name, field_info in pydantic_model.__fields__.items():
        try:
            if re.search(f"^\n*{field_name.lower()}:", content, flags=re.IGNORECASE):
                # ? new field started
                # split_content = re.split(f"^\n*{field_name.lower()}:", content, flags=re.IGNORECASE)
                # content = split_content[0]
                content = re.sub(f"^\n*{field_name.lower()}:", "", content, flags=re.IGNORECASE)
                curr_field = field_name
            elif re.search(field_name.lower()+":\n*$", content, flags=re.IGNORECASE):
                # ? new field ending
                split_content = re.split(f"{field_name.lower()}:\n*", content, flags=re.IGNORECASE)
                content = split_content[0]                    
                output = append_output(output, curr_field, content)
                    # output[curr_field] = sanitize_content(content)
                content = ""
                curr_field = field_name
            elif f"\n{field_name.lower()}:" in content.lower():
                # ? new field in middle
                split_content = re.split(f"\n{field_name.lower()}:\n*", content, flags=re.IGNORECASE)
                left_content = split_content[0]
                content += left_content
                output = append_output(output, curr_field, content)
                    # output[curr_field] = sanitize_content(content)
                right_content = split_content[1]
                content = right_content
                curr_field = field_name
        except Exception as e:
            print("error:", e)
            print(content)
            raise e        
    output = append_output(output, curr_field, content)
    return output, curr_field, content
