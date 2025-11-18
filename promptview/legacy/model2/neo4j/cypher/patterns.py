


def cypher_props(props: dict):
    if not props:
        return ""
    parts = []
    for k, v in props.items():
        # If v is a CyParam, will render as $paramname
        parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


class PatternChain:
    """Chains node/rel patterns into a single Cypher pattern string."""
    def __init__(self, *elements):
        self.elements = elements
    def __str__(self):
        return "".join(str(e) for e in self.elements)



class NodePattern:
    def __init__(self, var, labels=None, props=None):
        self.var = var
        self.labels = labels or []
        self.props = props or {}
    def __str__(self):
        label_str = ":" + ":".join(self.labels) if self.labels else ""
        prop_str = " " + cypher_props(self.props) if self.props else ""
        return f"({self.var}{label_str}{prop_str})"


class RelPattern:
    def __init__(self, var, rel_type=None, direction="right", props=None):
        self.var = var
        self.rel_type = rel_type
        self.direction = direction
        self.props = props or {}
    def __str__(self):
        typ = f":{self.rel_type}" if self.rel_type else ""
        prop_str = " " + str(self.props) if self.props else ""
        if self.direction == "right":
            return f"-[{self.var}{typ}{prop_str}]->"
        elif self.direction == "left":
            return f"<-[{self.var}{typ}{prop_str}]-"
        else:
            return f"-[{self.var}{typ}{prop_str}]-"




