import re
from jinja2 import Environment

def render_tabs(num: int):
    return "\t" * num


def add_tabs(content: str, tabs: int):
    return "\n".join([render_tabs(tabs) + c for c in content.split("\n")])



# def camel_to_snake(name):    
#     """Convert CamelCase to snake_case"""
#     s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
#     return s1.lower()

def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()



def int_to_roman(num: int, upper=True) -> str:
    """
    Convert an integer to a Roman numeral (1 <= num <= 3999).
    """
    # Each tuple is (integer value, Roman representation)
    roman_map = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"),  (90,  "XC"), (50,  "L"), (40,  "XL"),
        (10,  "X"),  (9,   "IX"), (5,   "V"), (4,   "IV"),
        (1,   "I")
    ]
    
    result = []
    for value, symbol in roman_map:
        # Figure out how many times the Roman numeral fits
        count, num = divmod(num, value)
        result.append(symbol * count)
        if num == 0:
            break
    if not upper:
        return "".join(result).lower()
    return "".join(result)



class SafeJinjaFormatter:
    def format(self, template_string, **kwargs):
        # Create a Jinja2 environment
        env = Environment()

        # Render the template with the given kwargs
        # Use the env.from_string method to create a template from the provided string
        template = env.from_string(template_string)
        return template.render(**kwargs)
    
    def __call__(self, template_string, **kwargs):
        return self.format(template_string, **kwargs)