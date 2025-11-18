#! working
import threading
from typing import Callable, Coroutine, Literal, Union, Any
from ...prompt.agent import Agent
from ...prompt.util.block_visualization import block_to_html
from ...prompt.legacy.block1 import TitleBlock
from ...prompt.legacy.context import Context
from abc import abstractmethod

import nest_asyncio
nest_asyncio.apply()
import ipywidgets as widgets
from IPython.display import display, HTML
import asyncio
import datetime as dt

class AsyncButton(widgets.Button):
    def __init__(self, description: str, on_click: Callable[[Any], Coroutine[Any, Any, Any]], icon: str = "", disabled: bool = False):
        super().__init__(description=description, icon=icon, disabled=disabled)
        self.on_click_async = on_click
        self.on_click(self._handle_on_click)
        
    def _handle_on_click(self, change):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.on_click_async(change))
        return result
    
    @abstractmethod
    async def on_click_effect(self, change) -> Any:
        ...
    










next_button = widgets.Button(description='Next')

# display(send_button, chat_output)
def _render_message(content: str, role: str = "user", created_at: dt.datetime | None = None) -> str:
        """Render a single message."""
        bg_color = "#f8f9fa" if role == "user" else "#e3f2fd"
        role_display = role.capitalize()
        time_str = created_at.strftime("%H:%M:%S") if created_at else ""
        
        return f"""
        <div style="margin: 10px 0; padding: 10px; background: {bg_color}; 
                    border-radius: 4px; max-width: 80%; min-width: 400px;">
            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">
                {role_display} {time_str}
            </div>
            <div style="white-space: pre-wrap;">{content}</div>
        </div>
        """
        
def display_message(content: str, role: str = "user", created_at: dt.datetime | None = None):
    display(HTML(_render_message(content, role, created_at)))



def display_user_message(user_message):
    msg_html = widgets.HTML(
        value=block_to_html(TitleBlock(role="user", content=user_message)),
    )
    msg_box=widgets.HBox([msg_html, next_button])
    display(msg_box)
    
    
def display_assistant_message(assistant_message):
    content = assistant_message.content
    msg_html = widgets.HTML(
        # value=_render_message(content, "assistant", dt.datetime.now()),
        value=block_to_html(assistant_message)
    )
    display(msg_html)



class ChatOutput(widgets.Output):

    def __init__(self, context: Context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context

    # async def render_history(self):
    #     async with Context() as ctx:
    #         conv = ctx.last()        
    #         with self:
    #             for block in conv:
    #                 display(HTML(block_to_html(block)))
    async def render_history(self):
        conv = self.context.last()        
        with self:
            for block in conv:
                display(HTML(block_to_html(block)))





class ChatUI:
  
  
    def __init__(self, agent: Agent, context: Context):
        self.agent = agent
        self.context = context
        self.chat_output = ChatOutput(context=context)
        self.send_button = AsyncButton(description='Send', on_click=self.run_agent_handler)
        self.message_input = self.build_input(on_submit_handler=self.run_agent_handler)
        self.clear_session_btn = AsyncButton(description="New Session", on_click=self.clear_session, icon="plus")
        self.top_bar = widgets.HBox([self.clear_session_btn])
        self.chat_ui = widgets.VBox([self.top_bar, self.chat_output, widgets.HBox([self.message_input, self.send_button])])
        
    def build_input(self, on_submit_handler):
        message_input = widgets.Text(placeholder='Type your message here...')
        message_input.on_submit(on_submit_handler)
        return message_input
    
    async def run_agent_handler(self, change):
        user_message = self.message_input.value.strip()
        self.message_input.value = ""
        with self.chat_output:
            display_user_message(user_message)
            async with self.context.resume() as ctx:
                async for res in self.agent(ctx=ctx, message=user_message):
                    if res is not None:
                        display_assistant_message(res)
                else:
                    print("None")
            # async with Context() as ctx:
            #     async for res in self.agent(ctx=ctx, message=user_message):
            #         if res is not None:
            #             display_assistant_message(res)
            #         else:
            #             print("None")


    async def clear_session(self, change):
        with self.chat_output:
            self.context.clear_session()
            self.chat_output.clear_output(wait=True)
            print("")    
            # async with Context() as ctx:
            #     ctx.clear_session()
            #     self.chat_output.clear_output(wait=True)
            #     print("")    
    
    @staticmethod
    async def display(agent: Agent, context: Context):
        if not context.is_initialized:
            context.init()
        chat_ui = ChatUI(agent, context)
        await chat_ui.chat_output.render_history()
        display(chat_ui.chat_ui)









    @staticmethod
    async def display3(agent: Agent, context: Context):
        # ctx = Context(inputs=inputs).resume()
        # await ctx.__aenter__()
        async def run_ui(agent):
            for i in range(10):
                print(i)
                await asyncio.sleep(1)
            # async with context.resume() as ctx:
            #     chat_ui = ChatUI(agent)
            #     await chat_ui.chat_output.render_history()
            #     display(chat_ui.chat_ui)
            
        def thread_target(agent):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_ui(agent))
        
        thread = threading.Thread(target=thread_target, args=(agent,))
        thread.start()
        
        stop_event = asyncio.Event()
        
        # Create a background task to keep the event loop running
        async def process_events():
            while not stop_event.is_set():
                # Use a very short sleep to allow other tasks to run
                await asyncio.sleep(0.01)
                
        # Create multiple concurrent tasks to ensure responsiveness
        tasks = [asyncio.create_task(process_events()) for _ in range(3)]
        
        try:
            # Wait for all tasks
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            stop_event.set()
            for task in tasks:
                task.cancel()
                thread.join()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
    

    @staticmethod
    async def display2(agent: Agent, inputs: dict | None = None):
        chat_ui = ChatUI(agent)
        await chat_ui.chat_output.render_history()
        display(chat_ui.chat_ui)
        
        # Create an event to control the loop
        stop_event = asyncio.Event()
        
        # Create a background task to keep the event loop running
        async def process_events():
            while not stop_event.is_set():
                # Use a very short sleep to allow other tasks to run
                await asyncio.sleep(0.01)
                
        # Create multiple concurrent tasks to ensure responsiveness
        tasks = [asyncio.create_task(process_events()) for _ in range(3)]
        
        try:
            # Wait for all tasks
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            stop_event.set()
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  





# async def build_chat_ui():
#     chat_output = ChatOutput()

#     send_button = AsyncButton(description='Send', on_click=run_agent_handler)

#     new_session_btn = widgets.Button(description="New Session", icon="plus")


#     clear_session_btn = AsyncButton(
#         description="New Session", 
#         on_click=clear_session, 
#         icon="plus"
#     )       

#     top_bar = widgets.HBox([
#         clear_session_btn,
#         # widgets.Button(description="Forward"),
#     ])

#     chat_ui = widgets.VBox([
#         top_bar,
#         chat_output,
#         widgets.HBox([message_input, send_button])
#     ])

#     await chat_output.render_history()

#     return chat_ui






# from typing import Any, Callable, Coroutine
# from IPython.display import display, HTML
# import ipywidgets as widgets

# import asyncio
# import datetime as dt

# class ChatView:
#     """Interactive chat view for Jupyter notebooks using ipywidgets."""
    
#     def __init__(self, agent_fn: Callable[[Context, str], Coroutine[Any, Any, Any]]):
#         """
#         Initialize chat view.
        
#         Args:
#             agent_fn: Async function that takes (context, message) and returns agent response
#         """
#         self.agent_fn = agent_fn
#         self.context = Context()
#         self.context.init()
        
#         # Create widgets
#         self.messages_output = widgets.Output()
#         self.input_text = widgets.Text(
#             placeholder='Type a message...',
#             layout=widgets.Layout(width='80%')
#         )
#         self.send_button = widgets.Button(
#             description='Send',
#             button_style='primary',
#             layout=widgets.Layout(width='100px')
#         )
        
#         # Layout
#         self.input_box = widgets.HBox([
#             self.input_text, 
#             self.send_button
#         ])
#         self.chat_box = widgets.VBox([
#             self.messages_output,
#             self.input_box
#         ], layout=widgets.Layout(width='800px', margin='20px auto'))
        
#         # Bind events
#         self.send_button.on_click(self._on_send_clicked)
#         self.input_text.on_submit(self._on_send_clicked)
        
#     def _render_message(self, content: str, role: str = "user", created_at: dt.datetime | None = None) -> str:
#         """Render a single message."""
#         bg_color = "#f8f9fa" if role == "user" else "#e3f2fd"
#         role_display = role.capitalize()
#         time_str = created_at.strftime("%H:%M:%S") if created_at else ""
        
#         return f"""
#         <div style="margin: 10px 0; padding: 10px; background: {bg_color}; 
#                     border-radius: 4px; max-width: 80%;">
#             <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">
#                 {role_display} {time_str}
#             </div>
#             <div style="white-space: pre-wrap;">{content}</div>
#         </div>
#         """
    
#     def _on_send_clicked(self, _):
#         """Handle send button click or input submission."""
#         message = self.input_text.value
#         if not message.strip():
#             return
            
#         # Clear input
#         self.input_text.value = ''
        
#         # Get current event loop or create new one
#         try:
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
        
#         # Run message handling in the event loop
#         if loop.is_running():
#             # We're inside a notebook cell
#             future = asyncio.create_task(self._handle_message(message))
#             asyncio.create_task(self._wait_and_handle_errors(future))
#         else:
#             # We're in a button click
#             loop.run_until_complete(self._handle_message(message))
            
#     async def _wait_and_handle_errors(self, future):
#         """Wait for a future and handle any errors."""
#         try:
#             await future
#         except Exception as e:
#             with self.messages_output:
#                 display(HTML(self._render_message(
#                     content=f"Error: {str(e)}",
#                     role="system"
#                 )))
    
#     async def _handle_message(self, message: str):
#         """Process a message through the agent."""
#         try:
#             # Display user message
#             with self.messages_output:
#                 display(HTML(self._render_message(message, "user")))
            
#             # Process with agent
#             async with self.context as ctx:
#                 async for response in self.agent_fn(ctx=ctx, message=message):
#                     if response and response.content:
#                         with self.messages_output:
#                             display(HTML(self._render_message(
#                                 content=response.content,
#                                 role=response.role,
#                                 created_at=response.created_at
#                             )))
#         except Exception as e:
#             with self.messages_output:
#                 display(HTML(self._render_message(
#                     content=f"Error: {str(e)}",
#                     role="system"
#                 )))
    
#     def display(self):
#         """Display the chat interface."""
#         display(self.chat_box)
        
#     def clear(self):
#         """Clear the chat history."""
#         self.context = Context()
#         self.context.init()
#         self.messages_output.clear_output()
        
#     async def send(self, message: str):
#         """Programmatically send a message."""
#         await self._handle_message(message)
