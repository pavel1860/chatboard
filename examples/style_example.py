from chatboard.prompt.legacy.str_block import block
from chatboard.block.style import style_manager

def main():
    # Define some global styles
    style_manager.add_rule("task", {
        "title": "md",
        "heading_level": 2,
        "bold": True
    })
    
    style_manager.add_rule(".important", {
        "bullet": "*",
        "bold": True
    })
    
    style_manager.add_rule("rule", {
        "bullet": "number",
        "indent": 2
    })
    
    style_manager.add_rule("output_format", {
        "bullet": "-",
        "code": True
    })
    
    # Add a more complex selector
    style_manager.add_rule("task .nested", {
        "indent": 4,
        "italic": True
    })
    
    # Create a prompt with blocks
    with block("Task Description", tags=["task"]) as b:
        b += "Generate a Python function that calculates the Fibonacci sequence."
        
        with block("Requirements", tags=["rule"]) as rules:
            rules += "The function should take an integer n as input."
            rules += "It should return the nth Fibonacci number."
            rules += "It should handle edge cases (n=0, n=1)."
            rules += "It should use an efficient algorithm."
            
        with block("Important Notes", tags=["important"]) as notes:
            notes += "The function should be well-documented."
            notes += "Include type hints."
            
        with block("Output Format", tags=["output_format"]) as output:
            output += "Python function"
            output += "Example usage"
            output += "Time complexity analysis"
    
    # Render the prompt
    prompt = b.render()
    print("=== Styled Prompt ===")
    print(prompt)
    print("\n")
    
    # A/B Testing Example
    print("=== A/B Testing Example ===")
    
    # Enable A/B testing
    style_manager.enable_ab_testing("fibonacci_prompt_test")
    
    # Define variant A: Markdown style
    style_manager.add_variant("markdown", [
        ("task", {"title": "md", "heading_level": 2}),
        ("rule", {"bullet": "number", "indent": 2}),
        ("important", {"bullet": "*", "bold": True}),
        ("output_format", {"bullet": "-", "code": True})
    ])
    
    # Define variant B: XML style
    style_manager.add_variant("xml", [
        ("task", {"title": "xml", "tag_name": "task"}),
        ("rule", {"bullet": "alpha", "indent": 4}),
        ("important", {"bullet": "!", "italic": True}),
        ("output_format", {"bullet": ">", "code": True})
    ])
    
    # Test both variants
    for variant in ["markdown", "xml"]:
        style_manager.select_variant(variant)
        
        # Create the same prompt structure
        with block("Task Description", tags=["task"]) as b:
            b += "Generate a Python function that calculates the Fibonacci sequence."
            
            with block("Requirements", tags=["rule"]) as rules:
                rules += "The function should take an integer n as input."
                rules += "It should return the nth Fibonacci number."
                rules += "It should handle edge cases (n=0, n=1)."
                rules += "It should use an efficient algorithm."
                
            with block("Important Notes", tags=["important"]) as notes:
                notes += "The function should be well-documented."
                notes += "Include type hints."
                
            with block("Output Format", tags=["output_format"]) as output:
                output += "Python function"
                output += "Example usage"
                output += "Time complexity analysis"
        
        # Render the prompt with this variant
        prompt = b.render()
        print(f"=== Variant: {variant} ===")
        print(prompt)
        print("\n")
        
        # Simulate recording metrics (in a real scenario, this would be based on LLM output quality)
        style_manager.record_metric("clarity_score", 8.5 if variant == "markdown" else 7.8)
        style_manager.record_metric("response_time", 2.1 if variant == "markdown" else 1.8)
    
    # Get the best variant based on different metrics
    best_clarity = style_manager.get_best_variant("clarity_score", higher_is_better=True)
    best_speed = style_manager.get_best_variant("response_time", higher_is_better=False)
    
    print(f"Best variant for clarity: {best_clarity}")
    print(f"Best variant for speed: {best_speed}")
    
    # Example of inline styles
    print("\n=== Inline Styles Example ===")
    
    with block("Mixed Styling", tags=["task"]) as b:
        b += "This block uses global styles from the 'task' tag."
        
        # Add a block with inline styles that override global styles
        inline_block = "Inline Styled Block"
        b += inline_block
        # Apply style to the last added item
        b.top.items[-1].add_style(
            title="md",
            heading_level=3,
            italic=True,
            bullet=None
        )
        
        # Add a regular block that inherits global styles
        b += "This block inherits styles from the parent."
        
        # Add a nested block with its own tag
        with block("Nested Block", tags=["nested"]) as nested:
            nested += "This block has the 'nested' tag and inherits styles from the parent."
            nested += "It should have additional indentation and be italic."
    
    print(b.render())
    
    # Example of complex style combinations
    print("\n=== Complex Style Combinations ===")
    
    # Clear previous styles
    style_manager.clear_rules()
    
    # Add some complex style rules
    style_manager.add_rule("section", {
        "title": "md",
        "heading_level": 2
    })
    
    style_manager.add_rule("section .subsection", {
        "title": "md",
        "heading_level": 3,
        "indent": 2
    })
    
    style_manager.add_rule("section .subsection .item", {
        "bullet": "-",
        "indent": 4
    })
    
    style_manager.add_rule(".highlight", {
        "bold": True,
        "italic": True
    })
    
    # Create a complex nested structure
    with block("Document Structure", tags=["section"]) as doc:
        doc += "This is a document with nested sections and styling."
        
        with block("First Section", tags=["subsection"]) as section1:
            section1 += "This is the first subsection."
            
            with block("Items", tags=["item"]) as items1:
                items1 += "Item 1"
                items1 += "Item 2"
                
                # Add an item with highlight
                items1 += "Special Item"
                items1.top.items[-1].add_style(bold=True, italic=True)
        
        with block("Second Section", tags=["subsection"]) as section2:
            section2 += "This is the second subsection."
            
            with block("More Items", tags=["item", "highlight"]) as items2:
                items2 += "Item A"
                items2 += "Item B"
                items2 += "Item C"
    
    # Debug parent-child relationships
    print("Debugging parent-child relationships:")
    print(f"Doc tags: {doc._main_inst.tags}")
    
    # Check the first level items
    for i, item in enumerate(doc._main_inst.items):
        if hasattr(item, 'tags'):
            print(f"Item {i} tags: {item.tags}")
            print(f"Item {i} parent: {item.parent}")
            
            # If this is a section, check its items
            if 'subsection' in item.tags and hasattr(item, 'items'):
                print(f"  Subsection items count: {len(item.items)}")
                for j, subitem in enumerate(item.items):
                    if hasattr(subitem, 'tags'):
                        print(f"  Subitem {j} tags: {subitem.tags}")
                        print(f"  Subitem {j} parent: {subitem.parent}")
    
    # Compute styles and print them
    doc._main_inst.compute_styles()
    
    print("\nComputed Styles:")
    print(f"Doc: {doc._main_inst.computed_style}")
    
    # Check styles of first level items
    print("\nDoc items:")
    for i, item in enumerate(doc._main_inst.items):
        print(f"Item {i}: {item}")
        if hasattr(item, 'tags'):
            print(f"  Tags: {item.tags}")
            print(f"  Computed Style: {item.computed_style}")
            
            # If this is a section, check its items
            if 'subsection' in item.tags and hasattr(item, 'items'):
                print(f"  Subsection items:")
                for j, subitem in enumerate(item.items):
                    print(f"    Subitem {j}: {subitem}")
                    if hasattr(subitem, 'tags'):
                        print(f"      Tags: {subitem.tags}")
                        print(f"      Computed Style: {subitem.computed_style}")
    
    print(doc.render())

if __name__ == "__main__":
    main() 