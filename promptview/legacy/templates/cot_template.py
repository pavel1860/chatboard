
from uuid import uuid4
from pydantic import BaseModel, Field
from promptview.llms.exceptions import LLMToolNotFound
from promptview.llms.messages import AIMessage, ActionCall
from promptview.prompt.legacy.base_prompt import ToolChoiceParam
from promptview.prompt.legacy.chat_prompt import ChatPrompt
from promptview.prompt.legacy.mvc import view
import xml.etree.ElementTree as ET


class CotMessage(AIMessage):
    observation: str = Field(..., title="Observation", description="The current state observation")
    thought: str = Field(..., title="Thought", description="The thought process")
    # actions: List[ActionCall] = Field(..., title="Actions", description="The actions to take")
    
    @classmethod
    def render(cls):
        return f"""
            <root>
                <observation>[{cls.model_fields['observation'].description}]</observation>
                    <thought>[{cls.model_fields['thought'].description}]</thought>
                    <action name="action name">
                        <param name="param1">[Value 1]</param>
                        <param name="param2">[Value 2]</param>
                        (...)
                        <param name="paramN">[Value N]</param>
                    </action>
                    <action name="action name
                (Repeat as needed)
            </root>
        """



@view(title="use the following template for your output")
def chain_of_thought_inference():
    return """<?xml version="1.0"?>
        <root>
            <observation>[Current state observation]</observation>
            <thought>[Step-by-step reasoning]</thought>
            <action name="action name">
                <param name="param1">[Value 1]</param>
                <param name="param2">[Value 2]</param>
                (...)
                <param name="paramN">[Value N]</param>
            </action>
            <action name="second action name (if needed)">
                <param name="param1">[Value 1]</param>
                <param name="param2">[Value 2]</param>
            </action>
            (... Repeat as needed)
        </root>
        
"""


@view(title="Output Format Rules")
def chain_of_thought_rules():
    return [
        "you have to output your anwer with the output xml like format provided.",
        "do not add any other comments or formatting to your answer.",
        "use only the provided output format.",
        "you should elaborate on your reasoning at the thought section.",
        "you should observe what is happening at the observation section."
        "the final action should be the one that is based on the user input.",
        "the parametrs of the final action should be based on the user input."        
    ]

    
@view(title="use the following template for your output")
def chain_of_thought_view():
    return chain_of_thought_inference(), chain_of_thought_rules()



def sanitize_text(text):
    if text.strip() in ["None", "", "none", "null", "nil", "NIL", "NULL", "NoneType", "noneType", "undefined"]:
        return None
    return text


def sanitize_xml(xml_string_raw):
    xml_string = xml_string_raw.strip()
    if xml_string.startswith("xml"):
        xml_string = xml_string[3:]
    return xml_string.strip()
        


class CotPrompt(ChatPrompt):
    # output_format = chain_of_thought_inference
    output_format = chain_of_thought_view
    tool_choice: ToolChoiceParam = "none"
    
    def find_actions(self, actions, root, action_tag="action", param_tag="param"):
        action_calls = []        
        for action in root.findall(action_tag):
            action_cls = actions.get(action.attrib["name"])
            if not action_cls:
                raise LLMToolNotFound(action.attrib["name"])
            params = {param.attrib["name"]: sanitize_text(param.text) for param in action.findall(param_tag)}
            action_inst = action_cls(**params)
            action_calls.append(
                ActionCall(
                    # id=f"tool_call_{uuid4()}", 
                    id=f"toolu_{uuid4().hex[:23]}",
                    name=action.attrib["name"], 
                    action=action_inst
                )
            )
        return action_calls
    
    def get_model_fields(self, model_cls):
        if not issubclass(model_cls, BaseModel):
            raise ValueError("model_cls must be a subclass of pydantic.BaseModel")
        parent_cls = model_cls.__bases__[0]
        if not issubclass(parent_cls, BaseModel):
            raise ValueError("model_cls Parent must be a subclass of pydantic.BaseModel")
        parent_fields = parent_cls.model_fields
        child_fields = model_cls.model_fields
        child_only_fields = {k: v for k, v in child_fields.items() if k not in parent_fields}
        return child_only_fields

    def parse_xml_response(self, response, actions, model_cls):
        try:
            xml_string = sanitize_xml(response.content)
            root = ET.fromstring(xml_string)
            fields = self.get_model_fields(model_cls)
            params = {k: root.find(k).text for k in fields}
            action_calls = {"action_calls": self.find_actions(actions, root)}
            try:
                return model_cls(**(response.model_dump(exclude=["raw"]) | params | action_calls))
            except Exception as e:
                print(response.content)
                raise e
        except Exception as e:
            print(response.content)
            raise e
        
    async def parse_output(self, response, actions):        
        return self.parse_xml_response(response, actions, CotMessage)
    # async def parse_output(self, response, actions):
    #     xml_string = response.content
    #     root = ET.fromstring(xml_string)        
    #     params = response.model_dump() & {
    #         "observation": root.find("observation").text,
    #         "thought": root.find("thought").text,
    #         "action_calls": self.find_actions(actions, root)
    #     }
    #     message = CotMessage(**params)                     
    #     return message



cot_prompt = CotPrompt.decorator_factory()