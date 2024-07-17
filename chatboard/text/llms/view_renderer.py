from typing import Any, Iterable, Type
from chatboard.text.llms.function_utils import call_function, filter_func_args
from chatboard.text.llms.model_schema_prompt_parser import ModelSchemaPromptParser
from chatboard.text.llms.mvc import View, BaseModel, Action
from langchain_core.utils.function_calling import convert_to_openai_tool
import json





def format_dict_to_multiline(d, indent=4):
    formatted_string = ""
    for key, value in d.items():
        if isinstance(value, dict):
            value = "\n" + format_dict_to_multiline(value, indent + 4)
        elif isinstance(value, list):
            value = "[\n" + "\n".join(" " * (indent + 4) + str(item) + "," for item in value) + "\n" + " " * indent + "]"
        formatted_string += " " * indent + f"{key}: {value}\n"
    return formatted_string



class RenderOutput(BaseModel):
    view_prompt: str
    system_prompt: str
    action_prompt: str | None
    output_model_prompt: str | None
    output_model: Any
    actions: dict[str, Type[BaseModel]] = {}

    def log(self, system=True, actions=True, view=True):
        if system:
            print("##### System Message #####")
            print(self.system_prompt)
        if actions:
            if self.action_prompt:
                print("----- Action Message -----")
                print(self.action_prompt)
            if self.output_model_prompt:
                print("----- Output Model Message -----")
                print(self.output_model_prompt)
        if view:
            print("##### View Message #####")
            print(self.view_prompt)
        
    @property
    def is_output(self):
        return self.output_model is not None

    @property
    def is_actions(self):
        return len(self.actions) > 0



    
class ViewRenderer():        
    """
    ViewRenderer is a class that renders a view to a prompt or a system output.
    Attributes:
        view_indent: int | None = None: Whether to render the view as a multi-line json.
        view_to_prompt: bool = False: Whether to render the view as a prompt.
        system_indent: int | None = None: Whether to render the system output as a multi-line json.
        system_to_prompt: bool = False: Whether to render the system output as a prompt without json format. 
    """

    def __init__(
        self,
        view_indent: int | None = None,
        view_to_prompt: bool = False,
        system_indent: int | None = None,
        system_to_prompt: bool = False,
        
    ):
        self._view_indent = view_indent
        self._view_to_prompt = view_to_prompt
        self._system_indent = system_indent
        self._system_to_prompt = system_to_prompt
        self._prompt_parser = ModelSchemaPromptParser()

    def use_default_view_render(self, view: View | BaseModel) -> str:
        if self._view_to_prompt:            
            render_output = format_dict_to_multiline(view.dict(), indent=self._view_indent or 4)
        else:
            render_output = json.dumps(view.dict(), indent=self._view_indent)
        view_tool = self.convert_to_openai_tool(view.__class__)

        prefix = ''
        title = view_tool['function'].get('name')
        if title and title != view.__class__.__name__:
        # if title:
            prefix = f"{title}:"
        if view_tool['function']['description']:
            prefix += f"{view_tool['function']['description']}"
        if prefix:
            render_output = f"{prefix}\n{render_output}"
        return render_output

    def use_default_render_system(self, view: View | BaseModel) -> str:        
        if self._system_to_prompt:
            return self._prompt_parser.to_prompt(view.__class__)
            # return self.model_to_prompt(openai_tool, hide_output=True)
        else:
            openai_tool = self.convert_to_openai_tool(view.__class__)
            return json.dumps(openai_tool, indent=self._system_indent)
        
    def use_default_render_tool(self, tool: Action | BaseModel) -> str:
        if self._system_to_prompt:
            if hasattr(tool, 'default'):
                return self._prompt_parser.to_prompt(tool.default)
            return self._prompt_parser.to_prompt(tool)
                # return self.model_to_prompt(self.convert_to_openai_tool(tool.default), hide_output=True)
            # return self.model_to_prompt(self.convert_to_openai_tool(tool), hide_output=True)
            
        if hasattr(tool, 'default'):
            openai_tool = self.convert_to_openai_tool(tool.default)
        openai_tool = self.convert_to_openai_tool(tool)
        return json.dumps(openai_tool, indent=self._system_indent)
    
            

    async def render_view_aux(
            self, 
            view: View | BaseModel, 
            # render_system: bool = False, 
            # render_tool: bool = False,
            **kwargs
        ):
        # if render_system:
        #     render_output = view.render_system()
        #     if render_output is None:
        #         render_output = self.use_default_render_system(view)
        # elif render_tool:
        #     render_output = view.render_tool()
        #     if render_output is None:
        #         render_output = self.use_default_render_tool(view)
        # else: # render regular view
        filtered_args = filter_func_args(view.render, kwargs)
        render_output = await call_function(view.render, **filtered_args)
        if render_output is None:
            render_output = self.use_default_view_render(view)
        return render_output
    
    def render_system_aux(self, view):
        render_output = view.render_system()
        if render_output is None:
            render_output = self.use_default_render_system(view)
        return render_output
    
    def render_tool_aux(self, view):
        render_output = None
        if hasattr(view, 'render_tool'):
            render_output = view.render_tool()
        if render_output is None:
            render_output = self.use_default_render_tool(view)
        return render_output
    
    def get_subclasses(self, view_cls):
        stack = [view_cls]
        system_visited = set()
        system_visited.add(view_cls.__name__)
        while stack:
            curr_model = stack.pop()
            for field, field_info in curr_model.__fields__.items():
                print(field)
                if issubclass(field_info.annotation, BaseModel):            
                    stack.append(field_info.annotation)
                    system_visited.add(field_info.annotation.__name__)
        return system_visited

    async def render_view(self, view: View | BaseModel, **kwargs):
        visited = set()
        visited_system = set()
        stack = []
        render_tree = {}
        if isinstance(view, list) or isinstance(view, tuple):
            stack = [v for v in view]
        else:
            stack.append(view)

        view_prompt = ""
        system_prompt = ""
        action_prompt = None
        output_model_prompt = None
        actions = {}
        output_model = None
        for _ in range(10):
            if not stack:
                break
            curr_view = stack.pop(0)
            
            if isinstance(curr_view, BaseModel):
                if curr_view.__class__.__name__ in visited:
                    continue        
                # handle rendering
                view_render = await self.render_view_aux(curr_view, **kwargs)
                # handle system rendering
                # if curr_view.__class__.__name__ not in visited_system:
                #     system_render = self.render_system_aux(curr_view)
                #     system_prompt += system_render + "\n"
                #     visited_system = visited_system | self.get_subclasses(curr_view.__class__)
                system_render = self.render_system_aux(curr_view)
                system_prompt += system_render + "\n"
                visited.add(curr_view.__class__.__name__)
                # handle tool gathering
                if curr_view._output_model:
                    if output_model is not None or actions:
                        raise ValueError("Only one output model is allowed")            
                    output_model = view._output_model
                if curr_view._actions:
                    if output_model:
                        raise ValueError("Only one output model is allowed")
                    actions.update({a.__name__: a for a in curr_view._actions})
            elif isinstance(curr_view, str):
                view_render = curr_view
            else:
                raise ValueError(f"Invalid view type {curr_view}")
                
            
            # appending to the prompt
            if isinstance(view_render, str):
                view_prompt += view_render + "\n"
            elif isinstance(view_render, BaseModel):
                stack = [view_render] + stack
            elif isinstance(view_render, Iterable):
                stack = [v for v in view_render] + stack
            else:
                raise ValueError("Invalid view render type")

        if actions:
            action_prompt = ""
            for _, action_cls in actions.items():
                action_prompt += self.render_tool_aux(action_cls) + "\n"
        if output_model:
            output_model_prompt = self.render_tool_aux(output_model)
        

        return RenderOutput(
            view_prompt=view_prompt,
            system_prompt=system_prompt,
            action_prompt=action_prompt,
            output_model_prompt=output_model_prompt,
            output_model=output_model,
            actions=actions            
        )
        
        

    # def render_view_system(self, view):
    #     return self.model_to_prompt(self.convert_to_openai_tool(view), hide_output=True)

    
    # def render_output_as_prompt(self, view):
    #     if view._output_model is not None:
    #         if hasattr(view._output_model, 'default'):
    #             return self.model_to_prompt(self.convert_to_openai_tool(view._output_model.default), hide_output=True)
    #         return self.model_to_prompt(self.convert_to_openai_tool(view._output_model), hide_output=True)
    #     elif view._actions:
    #         prompt = ""
    #         for action in view._actions:                
    #             prompt += self.model_to_prompt(self.convert_to_openai_tool(action), hide_output=True)
    #             prompt += "\n"
    #         return prompt
    #     else:
    #         raise ValueError(f"no output model or actions found in view: {view}")
        

    def render_output_as_tools(self, view):
        if view._output_model is not None:
            if hasattr(view._output_model, 'default'):
                return self.convert_to_openai_tool(view._output_model.default)
            return self.convert_to_openai_tool(view._output_model)
        
        elif view._actions:
            actions = []
            for action in view._actions:            
                actions.append(self.convert_to_openai_tool(action))
            return actions
        else:
            raise ValueError(f"no output model or actions found in view: {view}")
    


    def parse_properites(self, properties, add_type=True, add_constraints=True, tabs="\t", hide_output=False):
        prompt = ""
        for prop, value in properties.items():
            if hide_output and 'is_output' in value:
                continue
            param_promp = f"\n{tabs}{prop}"
            if 'allOf' in value:                 
                obj = value['allOf'][0]
                prompt += f"\n{tabs}{obj['title']}:"
                prompt += self.parse_properites(obj['properties'], tabs=tabs+"\t")
            elif 'anyOf' in value:            
                prompt += f"\n{tabs}{prop}: "
                if 'description' in value:
                    prompt += value['description']
                action_names = ",".join([obj['title'] for obj in value['anyOf']])
                prompt += f"has to be One of {action_names}"
                for obj in value['anyOf']:                            
                    prompt += f"\n{tabs}\t{obj['title']}:"
                    prompt += self.parse_properites(obj['properties'], add_type=add_type, add_constraints=add_constraints, tabs=tabs+"\t\t")
            elif value.get('type') == 'object':
                prompt += f"\n{tabs}{prop}:"
                prompt += self.parse_properites(value['properties'], tabs=tabs+"\t")
            else:
                if add_type:
                    param_promp += f":({value['type']})"
                if 'description' in value:
                    param_promp += f" {value['description']}"
                if add_constraints and ('minimum' in value or 'maximum' in value):
                    param_promp += f". should be"
                    if 'minimum' in value:
                        param_promp += f" minimum {value['minimum']}"
                    if 'maximum' in value:
                        param_promp += f" maximum {value['maximum']}"
                    param_promp += "."
                prompt += param_promp
        return prompt



    def model_to_prompt(self, tool_dict, add_type=True, add_constraints=True, hide_name=False, hide_output=False):
        tool_function = tool_dict['function']
        
        if not hide_name:
            prompt = f"""{tool_function["name"]}:"""
        else:
            prompt = ""
        if 'description' in tool_function:
            prompt += f" {tool_function['description']}"
        properties = tool_dict['function']["parameters"]['properties']
        prompt += self.parse_properites(properties, add_type, add_constraints, hide_output=hide_output)
        return prompt

    def convert_to_openai_tool(self, view_cls):
        # return view_cls.model_json_schema()
        return convert_to_openai_tool(view_cls)
    



    # async def recursive_render_view(self, parent_view, **kwargs):
    #     filtered_args = filter_func_args(parent_view.render, kwargs)    
    #     render_output = await call_function(parent_view.render, **filtered_args)
    #     if isinstance(render_output, str):
    #         return render_output
    #     elif isinstance(render_output, Iterable):
    #         prompt = ""
    #         for child_view in render_output:
    #             if isinstance(child_view, str):
    #                 prompt += "\n" + child_view 
    #             else:
    #                 child_render_output = await self.render_view(child_view, **kwargs)
    #                 prompt += "\n" + child_render_output + "\n"
    #         return prompt
    #     else:
    #         raise ValueError(f"Invalid view type: {parent_view}")
