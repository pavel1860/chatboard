from ..utils.completion_parsing import parse_bool, sanitize_content, split_rows, to_dict
from enum import Enum
from ..utils.completion_parsing import search_field, split_field
from typing import Optional, Union, get_type_hints, get_origin, get_args, Type
from pydantic import BaseModel
import inspect

class SplitAction(str, Enum):
    union = 'union'
    model = 'model'
    field = 'field'


def get_field_graph(output_model, parents):        
    field_graph = []
    for field_name, field_type in output_model.__annotations__.items():     
        # if get_origin(field_type) == Union:
        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            raise ValueError(f"Enum type {field_type} is not supported in the output model")
        elif any(get_args(field_type)):
            union_args = get_args(field_type)
            action = SplitAction.union
            options = {union_arg.__name__ : get_field_graph(union_arg, parents + [field_name]) for union_arg in union_args}
        elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            action = SplitAction.model
            options = get_field_graph(field_type, parents + [field_name])
        else:
            action = SplitAction.field
            options = None
        field_graph.append({"field": field_name, "schema": field_type, "content": '', 'action': action, "options": options, 'parents': parents})
    return field_graph





class OutputParser:

    def __init__(self, output_model, verbose=False):
        self.field_graph = get_field_graph(output_model, [])        
        self.output_model = output_model
        self.verbose = verbose
        self.next_queue = [f for f in self.field_graph]
        self._curr_content = ''
        self.output = {}

    def _reset_state(self):
        self._curr_content = ''
        self.output = {}
        self.next_queue = [f for f in self.field_graph]
    # def push(self, field_name, field_type, action, options):
    #     self.field_graph.append({"field": field_name, "schema": field_type, "content": '', 'action': action, "options": options})

    def _top_stack(self):
        if not self.next_queue:
            return None
        return self.next_queue[0]
    
    def _second_top_stack(self):
        # if len(self.next_queue) < 2:
            # self.next_queue = [f for f in self.field_graph]
        if len(self.next_queue) < 2:
            return None
        return self.next_queue[1]

    @property
    def curr_field(self):
        top_item = self._top_stack()
        if not top_item:
            return None
        return top_item['field']
    
    @property
    def curr_parents(self):
        top_item = self._top_stack()
        if not top_item:
            return []
        return top_item['parents']
    
    @property
    def curr_action(self):
        top_item = self._top_stack()
        if not top_item:
            return None
        return top_item['action']
    
    @property
    def curr_field_type(self):
        top_item = self._top_stack()
        if not top_item:
            return None
        return top_item['schema']
    
    @property
    def curr_options(self):
        top_item = self._top_stack()
        if not top_item:
            return None
        return top_item['options']
    
    @property
    def next_field(self):
        second_item = self._second_top_stack()
        if second_item:
            return second_item['field']
    
    @property
    def next_field_type(self):
        second_item = self._second_top_stack()
        if second_item:
            return second_item['schema']

    def append_content(self, content):
        search_result = search_field(self.curr_field, content)
        if search_result:
            _, content = split_field(self.curr_field, content)
        self._curr_content += content

    def has_next_field(self):
        search_result = search_field(self.next_field, self._curr_content)
        return search_result
    
    def set_curr_output(self, field, parents, content=None):
        output = self.output
        for parent in parents:
            output = output[parent]
        if content is None:
            content = {}
        output[field] = content
        

    def commit_field(self):
        if self.next_field:
            prev_content, curr_content = split_field(self.next_field, self._curr_content)
            self.set_curr_output(self.curr_field, self.curr_parents, sanitize_content(prev_content))
        else:
            self.set_curr_output(self.curr_field, self.curr_parents, sanitize_content(self._curr_content))
            curr_content = ''
        self.next_queue.pop(0)
        if self.curr_action == SplitAction.union:
            if self.curr_options:
                self.set_curr_output(self.curr_field, self.curr_parents)
                sub_graph = self.curr_options[sanitize_content(curr_content)] 
                self.next_queue.pop(0)
                self.next_queue = [f for f in sub_graph] + self.next_queue
                self._curr_content = ''
        elif self.curr_action == SplitAction.model:
            sub_graph = self.curr_options
            self.set_curr_output(self.curr_field, self.curr_parents)
            self.next_queue.pop(0)
            self.next_queue = [f for f in sub_graph] + self.next_queue
            self._curr_content = ''
        else:
            self._curr_content = curr_content


    def parse(self, text):
        for chunk in split_rows(text):    
            self.append_content(chunk)    
            if self.has_next_field():
                self.commit_field()
        self.commit_field()
        output = self.output_model(**self.output)
        self._reset_state()
        return output