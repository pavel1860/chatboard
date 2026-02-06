from typing import Dict, List, Any, Tuple, Optional, TypedDict, Union, Literal, TypeVar, cast
from collections import defaultdict
import random
import re
import uuid



BlockType = Literal["xml", "md", "li", "func", "func-col"]
BulletType = Literal["number", "alpha", "roman", "roman_upper", "*", "-"] 
ListType = Literal["list", "list:number", "list:alpha", "list:roman", "list:roman_lower", "list:*", "list:-"]
# BlockType = Literal["list", "code", "table", "text"]



# Define types for style properties
StyleValue = Union[str, int, bool, None]
StyleDict = Dict[str, StyleValue]
InlineStyle = List[ListType | BlockType] | str
T = TypeVar('T')


class UndefinedTagError(Exception):
    pass


class BlockStyle:    
    depth: int
    block_type: BlockType
    is_list: bool
    bullet_type: BulletType | None
    style: InlineStyle
    
    def __init__(self, style: InlineStyle | None = None):
        if isinstance(style, str):
            style = style.split(" ")
        self.style = style or []
        self.block_type = "md"
        self.is_list = False
        self.bullet_type = None
        for tag in self.style:
            if tag.startswith("list"):
                self.is_list = True
                tag_parts = tag.split(":")
                self.bullet_type = tag_parts[1] if len(tag_parts) > 1 else "number" # type: ignore
            elif tag in {"md", "xml", "func", "func-col"}:
                self.block_type = tag
            
                
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default) or default
        
    def update(self, style: InlineStyle) -> None:
        self.style.extend(style)
        for tag in style:
            if tag.startswith("list"):
                self.is_list = True
                tag_parts = tag.split(":")
                self.bullet_type = tag_parts[1] if len(tag_parts) > 1 else "number" # type: ignore
            elif tag in {"md", "xml", "func", "func-col"}:
                self.block_type = tag
            # elif tag == "md":
            #     self.block_type = "md"
            # elif tag == "xml":
            #     self.block_type = "xml"
            # elif tag == "func" or tag == "func-col":
            #     self.block_type = "func"
                
    def __or__(self, other: InlineStyle) -> "BlockStyle":
        self.update(other)
        return self
    
    def __ior__(self, other: InlineStyle) -> "BlockStyle":
        self.update(other)
        return self
    
    def model_dump(self) -> dict:
        return {
            "depth": self.depth,
            "block_type": self.block_type,
            "is_list": self.is_list,
            "bullet_type": self.bullet_type,
            "style": self.style,
        }
    
    @classmethod    
    def model_validate(cls, data: dict) -> "BlockStyle":
        return cls(**data)

class StyleSelector:
    """
    A selector for matching blocks based on tags, similar to CSS selectors.
    Supports tag selectors, class selectors (using tags), and specificity calculation.
    """
    def __init__(self, selector: str):
        self.selector = selector
        self.is_nested = ' ' in selector
        
        if self.is_nested:
            # Handle nested selectors like "section .subsection"
            self.selector_parts = [self._parse_selector(part) for part in selector.split(' ')]
            # Compute specificity for all parts combined
            self.specificity = self._compute_nested_specificity()
        else:
            self.selector_parts = [self._parse_selector(selector)]
            self.specificity = self._compute_specificity(self.selector_parts[0])
    
    def _parse_selector(self, selector: str) -> Dict[str, Any]:
        """
        Parse a selector string into components.
        Examples:
        - "task" -> matches blocks with tag "task"
        - ".important" -> matches blocks with tag "important"
        - "#unique" -> matches blocks with tag "unique" (treated as highest specificity)
        """
        result = {"tag": None, "id": None, "classes": []}
        parts = re.findall(r'([#.]?[a-zA-Z0-9_-]+)', selector)
        
        for part in parts:
            if part.startswith("#"):
                result["id"] = part[1:]
            elif part.startswith("."):
                result["classes"].append(part[1:])
            else:
                result["tag"] = part
        
        return result
    
    def _compute_specificity(self, parsed_selector: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Compute specificity as (id_count, class_count, tag_count)
        Higher values have higher priority when resolving style conflicts
        """
        id_count = 1 if parsed_selector["id"] else 0
        class_count = len(parsed_selector["classes"])
        tag_count = 1 if parsed_selector["tag"] else 0
        return (id_count, class_count, tag_count)
    
    def _compute_nested_specificity(self) -> Tuple[int, int, int]:
        """
        Compute combined specificity for nested selectors
        """
        total_id_count = 0
        total_class_count = 0
        total_tag_count = 0
        
        for part in self.selector_parts:
            id_count, class_count, tag_count = self._compute_specificity(part)
            total_id_count += id_count
            total_class_count += class_count
            total_tag_count += tag_count
            
        return (total_id_count, total_class_count, total_tag_count)
    
    def _matches_single(self, block, parsed_selector: Dict[str, Any]) -> bool:
        """
        Check if a block matches a single parsed selector
        """
        # If no tags on the block, it can't match any selector
        if not hasattr(block, 'tags') or not block.tags:
            return False
        
        # Match ID selector (highest specificity)
        if parsed_selector["id"] and parsed_selector["id"] not in block.tags:
            return False
            
        # Match class selectors
        for cls in parsed_selector["classes"]:
            if cls not in block.tags:
                return False
                
        # Match tag selector
        if parsed_selector["tag"] and parsed_selector["tag"] not in block.tags:
            return False
            
        return True
    
    def matches(self, block) -> bool:
        """
        Check if a block matches this selector based on its tags
        For nested selectors, check if the block matches the last part
        and has ancestors matching the earlier parts
        """
        if not self.is_nested:
            return self._matches_single(block, self.selector_parts[0])
        
        # For nested selectors, we need to check if this block matches the last part
        # and has ancestors matching the earlier parts
        if not self._matches_single(block, self.selector_parts[-1]):
            return False
            
        # Check ancestors
        current = block
        for i in range(len(self.selector_parts) - 2, -1, -1):
            # Try to find a parent that matches
            parent_found = False
            
            # Look for a matching parent in the context stack
            # This is a simplified approach - in a real implementation,
            # you would need to traverse the actual DOM/tree structure
            if hasattr(current, 'parent') and current.parent:
                current = current.parent
                if self._matches_single(current, self.selector_parts[i]):
                    parent_found = True
            
            if not parent_found:
                return False
                
        return True


class StyleRule:
    """
    A style rule that combines a selector with style declarations
    """
    def __init__(self, selector: str, declarations: StyleDict):
        self.selector = StyleSelector(selector)
        self.declarations = declarations
        self.specificity = self.selector.specificity
    
    def matches(self, block) -> bool:
        """
        Check if this rule matches the given block
        """
        return self.selector.matches(block)


class StyleManager:
    """
    Manages style rules and applies them to blocks
    """
    def __init__(self):
        self.rules: List[StyleRule] = []
        self.experiment_id: Optional[str] = None
        self.variant_id: Optional[str] = None
        self.ab_testing_enabled = False
        self.ab_test_variants: Dict[str, List[Tuple[str, StyleDict]]] = {}
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def add_rule(self, selector: str, declarations: StyleDict) -> None:
        """
        Add a style rule to the manager
        """
        rule = StyleRule(selector, declarations)
        self.rules.append(rule)
    
    def clear_rules(self) -> None:
        """
        Clear all style rules
        """
        self.rules = []
    
    def apply_styles(self, block) -> StyleDict:
        """
        Apply matching style rules to a block and return the computed style
        """
        # Dictionary to store property -> (value, specificity, rule_order)
        computed = {}
        order = 0
        
        # Apply global rules if they match the block
        for rule in self.rules:
            if rule.matches(block):
                for prop, value in rule.declarations.items():
                    rule_specificity = rule.specificity
                    
                    if prop in computed:
                        prev_specificity, prev_order, _ = computed[prop]
                        # Override if higher specificity or later with equal specificity
                        if (rule_specificity > prev_specificity or 
                            (rule_specificity == prev_specificity and order > prev_order)):
                            computed[prop] = (rule_specificity, order, value)
                    else:
                        computed[prop] = (rule_specificity, order, value)
                order += 1
        
        # Apply inline styles with highest priority
        if hasattr(block, 'inline_style') and block.inline_style:
            inline_specificity = (float('inf'), float('inf'), float('inf'))
            for prop, value in block.inline_style.items():
                computed[prop] = (inline_specificity, order, value)
                order += 1
        
        # Extract just the values without the metadata
        return {prop: val for prop, (_, _, val) in computed.items()}
    
    def enable_ab_testing(self, experiment_id: Optional[str] = None) -> None:
        """
        Enable A/B testing for styles
        """
        self.ab_testing_enabled = True
        self.experiment_id = experiment_id or str(uuid.uuid4())
    
    def disable_ab_testing(self) -> None:
        """
        Disable A/B testing
        """
        self.ab_testing_enabled = False
    
    def add_variant(self, variant_id: str, rules: List[Tuple[str, StyleDict]]) -> None:
        """
        Add a variant for A/B testing
        """
        self.ab_test_variants[variant_id] = rules
    
    def select_variant(self, variant_id: Optional[str] = None) -> Optional[str]:
        """
        Select a variant for the current rendering
        If variant_id is provided, use that variant, otherwise randomly select one
        Returns the selected variant ID or None if A/B testing is disabled
        """
        if not self.ab_testing_enabled or not self.ab_test_variants:
            return None
            
        if variant_id and variant_id in self.ab_test_variants:
            self.variant_id = variant_id
        else:
            self.variant_id = random.choice(list(self.ab_test_variants.keys()))
            
        # Apply the rules for this variant
        self.clear_rules()
        for selector, declarations in self.ab_test_variants[self.variant_id]:
            self.add_rule(selector, declarations)
            
        return self.variant_id
    
    def record_metric(self, metric_name: str, value: Any) -> None:
        """
        Record a metric for the current variant
        """
        if self.ab_testing_enabled and self.variant_id:
            self.metrics[self.variant_id][metric_name] = value
    
    def get_best_variant(self, metric_name: str, higher_is_better: bool = True) -> Optional[str]:
        """
        Get the best variant based on a specific metric
        Returns the variant ID with the best metric value, or None if no data
        """
        if not self.metrics:
            return None
            
        variants = [(variant_id, data.get(metric_name)) 
                   for variant_id, data in self.metrics.items() 
                   if metric_name in data and data.get(metric_name) is not None]
        
        if not variants:
            return None
            
        if higher_is_better:
            return max(variants, key=lambda x: cast(float, x[1]))[0]
        else:
            return min(variants, key=lambda x: cast(float, x[1]))[0]


# Global style manager instance
style_manager = StyleManager() 