import pytest
import pytest_asyncio
from chatboard import view
from chatboard.prompt.legacy.mvc import render_block
from chatboard.prompt.legacy.chat_prompt import ChatPrompt

prompt = ChatPrompt.decorator_factory()


async def assert_basic_prompt(example_prompt, system_message, user_message, **kwargs):
    ex_ctx = await example_prompt.to_ex_ctx(**kwargs)
    messages = ex_ctx.messages
    # messages = await example_prompt(output_messages=True, **kwargs)
    for m in messages:
        print(f"-----{m.role}----")
        print(m.content)
    
    assert len(messages) == 2
    assert messages[0].content == system_message
    assert messages[1].content == user_message



# @pytest.mark.asyncio
# async def test_basic_view():
#     @view()
#     def example_view():
#         return "this is an example view"

#     out, _,_ = render_block(example_view())
#     assert out == "this is an example view"
    
    
    
# @pytest.mark.asyncio
# async def test_basic_list_view():
#     @view()
#     def example_list_view():
#         return [
#             "this is an example view 1",
#             "this is an example view 2",
#             "this is an example view 3",
#         ]

#     out, _,_ = render_block(example_list_view())
#     assert out == "this is an example view 1\nthis is an example view 2\nthis is an example view 3"
is_traceable = False

@pytest.mark.asyncio
async def test_basic_prompt():
    @prompt(is_traceable=is_traceable)
    def example_prompt():
        return "this is a test"

    # messages = await example_prompt(output_messages=True)
    ex_ctx = await example_prompt.to_ex_ctx()
    messages = ex_ctx.messages
    print(messages[0].content)
    assert messages[0].content == "this is a test"

user_message_list = "this is a test 1\nthis is a test 2\nthis is a test 3"
    
@pytest.mark.asyncio
async def test_basic_prompt_tuple():
    @prompt(is_traceable=is_traceable)
    def example_tuple_prompt():
        return (
            "this is a test 1",
            "this is a test 2",
            "this is a test 3"
        )
        
    # messages = await example_tuple_prompt(output_messages=True)
    ex_ctx = await example_tuple_prompt.to_ex_ctx()
    messages = ex_ctx.messages
    assert messages[0].content == user_message_list

    
    
@pytest.mark.asyncio
async def test_mixed_prompt_list():
    @view()
    def test_view(num: int):
        return "this is a test " + str(num)

    @prompt(is_traceable=is_traceable)
    def example_list_prompt():
        return [
            test_view(1),
            "this is a test 2",
            test_view(3),
            "this is a test 4"
        ]

    # messages = await example_list_prompt(output_messages=True)
    ex_ctx = await example_list_prompt.to_ex_ctx()
    messages = ex_ctx.messages
    assert len(messages) == 4
    assert messages[0].content == "this is a test 1"
    assert messages[1].content == "this is a test 2"
    assert messages[2].content == "this is a test 3"
    assert messages[3].content == "this is a test 4"
    

@pytest.mark.asyncio
async def test_basic_propertie():
    @prompt(
        background="the is test background",
        is_traceable=is_traceable,
    )
    def example_prompt():
        return "this is a test"

    await assert_basic_prompt(example_prompt, "the is test background", "this is a test")
    

system_message = "this is a test background 1\nthis is a test background 2"

@pytest.mark.asyncio
async def test_basic_propertie_list():
    @prompt(
        background=[
            "this is a test background 1",
            "this is a test background 2",
        ],
        is_traceable=is_traceable,
    )
    def example_prompt():
        return "this is a test"
    await assert_basic_prompt(example_prompt, system_message, "this is a test")
    
    
    
@pytest.mark.asyncio
async def test_basic_propertie_function():
    def background_fn():
        return "the is test background"

    @prompt(
        background=background_fn,
        is_traceable=is_traceable,
    )
    def example_prompt():
        return "this is a test"

    await assert_basic_prompt(example_prompt, "the is test background", "this is a test")
    
    

@pytest.mark.asyncio
async def test_basic_propertie_async_function():
    async def background_fn():
        return "the is test background"

    @prompt(
        background=background_fn,
        is_traceable=is_traceable
    )
    def example_prompt():
        return "this is a test"

    await assert_basic_prompt(example_prompt, "the is test background", "this is a test")
    
    

@pytest.mark.asyncio
async def test_basic_propertie_view():
    @view()
    def background_view():
        return "the is test background"


    @prompt(
        background=background_view,
        is_traceable=is_traceable
    )
    def example_prompt():
        return "this is a test"

    await assert_basic_prompt(example_prompt, "the is test background", "this is a test")
    
    
@pytest.mark.asyncio
async def test_basic_propertie_list_view():
    @view()
    def background_view():
        return [
            "this is a test background 1",
            "this is a test background 2",
        ]


    @prompt(
        background=background_view,
        is_traceable=is_traceable
    )
    def example_prompt():
        return "this is a test"

    await assert_basic_prompt(example_prompt, system_message, "this is a test")
    
    
    
@pytest.mark.asyncio
async def test_basic_propertie_function_list_view():    
    @view()
    def background_view(number):
        return f"this is a test background {number}"
            

    def background_fn():
        return [background_view(1), background_view(2)]


    @prompt(
        background=background_fn,
        is_traceable=is_traceable
    )
    def example_prompt():
        return "this is a test"

    await assert_basic_prompt(example_prompt, system_message, "this is a test")
    
    
@pytest.mark.asyncio
async def test_basic_propertie_replacement():
    @view()
    def background_view():
        return "this is a {{ value }} background"
            
    def background_fn():
        return background_view()

    @prompt(
        background=background_fn,
        is_traceable=is_traceable
    )
    def example_prompt():
        return "this is a {{ value }}"


    await assert_basic_prompt(example_prompt, "this is a test background", "this is a test", value="test")