import json
from pydantic import BaseModel





class ModelSchemaPromptParser:


    def __init__(self, indent=None, to_prompt=True) -> None:
        self._to_prompt = to_prompt
        self._indent = indent



    def get_schema(self, model_instance):        
        if self._to_prompt:
            return self.model_to_prompt(model_instance)
        if self.indent:
            return json.dumps(model_instance.dict(), indent=self.indent)


    def to_prompt(self, model_instance):
        model_schema = self.get_schema(model_instance)
        if self._to_prompt:
            return self.model_to_prompt(model_schema)
                

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
    


    # def dump_enum(self, prop, value):


    def handle_enum(self, prop: str, value: dict) -> str:
        enum_field = value['enum']
        prompt = f"{prop}: (enum)"
        if 'description' in value:
            prompt += f" {value['description']}"
        prompt += f" value should be one of the following: {','.join(enum_field)}"                
        return prompt


    def parse_properites(self, properties, add_type=True, add_constraints=True, tabs="\t", hide_output=False):
        prompt = ""
        for prop, value in properties.items():
            if hide_output and 'is_output' in value:
                continue
            param_promp = f"\n{tabs}{prop}"
            if 'allOf' in value: 
                for obj in value['allOf']:
                    if obj['type'] == 'object':
                        prompt += f"\n{tabs}{obj['title']}:"
                        prompt += self.parse_properites(obj['properties'], tabs=tabs+"\t")
                    elif obj['type'] == 'string':
                        prompt += f"\n{tabs}{self.handle_enum(prop, obj)}"
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

    def get_schema(self, model_instance: BaseModel):
        return model_instance.model_json_schema()