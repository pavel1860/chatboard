




class LlmError(Exception):
    pass


class LLMToolNotFound(Exception):
    
    def __init__(self, tool_name) -> None:
        self.tool_name = tool_name
        super().__init__(f"Action {tool_name} is not found")


class BadClientLlmRequest(Exception):
    pass