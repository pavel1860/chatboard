
from ..legacy.block1 import BaseBlock, ResponseBlock
from ..legacy.context import BlockStream


def block_to_html(block: BaseBlock):
    def role_chip(role):
        if role == "user":
            return '<div class="role-chip user">User</div>'
        elif role == "assistant": 
            return '<div class="role-chip assistant">Assistant</div>'
        return '<div class="role-chip tool">Tool</div>'
    
    def action_calls_html(action_calls):
        def action_call_html(action_call):
            return f"""<div class="action-call">
                <div class="action-call-id">{action_call["id"].split("_")[-1][:8]}</div>
                <div class="action-call-name">{action_call["name"]}</div>                    
            </div>"""
        if action_calls is None:
            return ''
        return '<div class="action-calls">' + ', '.join([action_call_html(action_call) for action_call in action_calls]) + '</div>'
                    
    return f"""
<style>
.message-container {{
    max-width: 800px;
    margin: 2px;
    padding: 4px;
    border-radius: 4px;
    background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}

.message-header {{
    display: flex;
    align-items: center;
    margin-bottom: 1px;
}}

.message-id {{
    font-size: 10px;
    color: #666;
    margin-right: 4px;
}}

.role-chip {{
    padding: 2px 6px;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 500;
    margin-right: 4px;
}}

.role-chip.user {{
    background: #E3F2FD;
    color: #1976D2;
}}

.role-chip.assistant {{
    background: #ffebee;
    color: #c62828;
}}

.role-chip.tool {{
    background: #FFF3E0;
    color: #F57C00;
}}

.message-time {{
    font-size: 10px;
    color: #999;
}}

.message-content {{
    font-size: 12px;
    line-height: 1.1;
    color: #333;
    white-space: pre-wrap;
}}

.action-call {{
    display: flex;
    align-items: center;
    background: #f5f5f5;
    border-radius: 4px;
    padding: 8px;
    margin: 4px 0;
    border-left: 3px solid #F57C00;
}}

.action-call-id {{
    font-size: 10px;
    color: #666;
    margin-right: 8px;
    padding: 2px 6px;
    background: #FFF3E0;
    border-radius: 4px;
}}

.action-call-name {{
    font-size: 11px;
    font-weight: 500;
    color: #F57C00;
    margin-right: 8px;
    text-transform: uppercase;
}}

</style>

<div class="message-container">
    <div class="message-header">
        <span class="message-id">#{block.id}</span>
        {role_chip(block.role)}        
    </div>
    <div class="message-content">
        {block.render()}
    </div>
    {action_calls_html(block.action_calls) if isinstance(block, ResponseBlock ) else ''}
</div>
"""




def block_stream_to_html(block_stream: BlockStream):
    html_str = f"""<html>
<div style="display: flex; flex-direction: column; width: 400px; margin: 0;">
    {"".join([block_to_html(block) for block in block_stream])}
</div>
</html>"""
    return html_str
    
    
    
    
def display_block_stream(block_stream: BlockStream):
    from IPython.display import display, HTML
    html = HTML(block_stream_to_html(block_stream))
    display(html)