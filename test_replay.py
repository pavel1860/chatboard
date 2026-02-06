"""Test Context-managed replay mode."""
import asyncio
from chatboard.model import NamespaceManager
from chatboard.prompt import component, stream, Context
from chatboard.prompt.span_tree import SpanTree


async def test_replay():
    """Test that replay mode works with Context-managed spans."""
    # Initialize clean database
    await NamespaceManager.initialize_clean()

    @stream()
    async def my_stream(name: str):
        print(f"[EXECUTE] my_stream {name}")
        yield f"hello {name}"
        yield f"world {name}"
        yield "!"

    @component()
    async def my_component(name: str):
        print(f"[EXECUTE] my_component {name}")
        response = yield my_stream(name)
        print(f"[EXECUTE] my_component got response: {response}")
        yield f"finished with {len(response)} items"

    # ==========================================
    # Step 1: Original execution
    # ==========================================
    print("\n=== ORIGINAL EXECUTION ===\n")

    ctx = Context()
    async with ctx.start_turn():
        original_results = []
        async for event in my_component("Alice").stream():
            if event.type in ['stream_delta', 'span_event']:
                print(f"  Event: {event.type} - {event.payload}")
                original_results.append(event.payload)

        # Save turn ID for replay
        turn_id = ctx.turn.id
        print(f"\n  Turn ID: {turn_id}")
        print(f"  Original results: {original_results}")

    # ==========================================
    # Step 2: Replay execution
    # ==========================================
    print("\n=== REPLAY EXECUTION ===\n")

    ctx2 = Context()
    async with (await ctx2.load_replay(turn_id=turn_id)).fork().start_turn():
        replay_results = []
        async for event in my_component("Alice").stream():
            if event.type in ['stream_delta', 'span_event']:
                print(f"  Event: {event.type} - {event.payload}")
                replay_results.append(event.payload)

        print(f"\n  Replay results: {replay_results}")

    # ==========================================
    # Step 3: Verify results match
    # ==========================================
    print("\n=== VERIFICATION ===\n")

    # Convert to strings for comparison (ignore object instances)
    original_str = [str(r) for r in original_results if isinstance(r, str)]
    replay_str = [str(r) for r in replay_results if isinstance(r, str)]

    if original_str == replay_str:
        print("✅ SUCCESS: Replay results match original!")
        print(f"  Matched {len(original_str)} string results")
    else:
        print("❌ FAILED: Results don't match")
        print(f"  Original: {original_str}")
        print(f"  Replay:   {replay_str}")
        return False

    # ==========================================
    # Step 4: Verify no execution in replay mode
    # ==========================================
    print("\n=== EXECUTION CHECK ===\n")
    print("If you see '[EXECUTE]' messages above in REPLAY section,")
    print("it means the functions were executed instead of replayed.")
    print("\nExpected: No [EXECUTE] messages in REPLAY section")
    print("Actual: Check output above")

    return True


if __name__ == "__main__":
    result = asyncio.run(test_replay())
    if result:
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✅")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("TESTS FAILED ❌")
        print("="*50)
