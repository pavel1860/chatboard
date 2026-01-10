from __future__ import annotations
from typing import Iterator, TYPE_CHECKING

from .span import Span, BlockChunk

if TYPE_CHECKING:
    pass


class BlockText:
    """
    Linked list of Spans.

    BlockText owns all Spans it contains (Span.owner = BlockText).
    Blocks reference Spans but don't own them.

    The linked list uses sentinel nodes (head/tail) for easier insertion
    at boundaries.
    """

    __slots__ = ["_head", "_tail", "_count"]

    def __init__(self):
        # Sentinel spans for easy insertion at boundaries
        self._head = Span(id="<<HEAD>>")  # Sentinel start (not included in iteration)
        self._tail = Span(id="<<TAIL>>")  # Sentinel end (not included in iteration)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._count = 0

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __iter__(self) -> Iterator[Span]:
        """Iterate over all spans (excluding sentinels)."""
        current = self._head.next
        while current is not self._tail:
            yield current
            current = current.next

    def __len__(self) -> int:
        """Number of spans (excluding sentinels)."""
        return self._count

    def __bool__(self) -> bool:
        """True if contains any spans."""
        return self._count > 0

    # -------------------------------------------------------------------------
    # Span Creation
    # -------------------------------------------------------------------------

    def create_span(
        self,
        content: str | list[BlockChunk] | None = None,
        after: Span | None = None,
    ) -> Span:
        """
        Create a new span owned by this BlockText.

        Args:
            content: Optional initial content (string or chunks)
            after: Insert after this span (default: append to end)

        Returns:
            The newly created span
        """
        span = Span(owner=self)

        if content is not None:
            if isinstance(content, str):
                span.chunks = [BlockChunk(content=content)]
            else:
                span.chunks = content

        if after is None:
            self.append(span)
        else:
            self.insert_after(after, span)

        return span

    # -------------------------------------------------------------------------
    # Insertion
    # -------------------------------------------------------------------------

    def append(self, span: Span) -> Span:
        """
        Append span to end of list.

        Sets span.owner to this BlockText.

        Returns:
            The appended span
        """
        return self.insert_before(self._tail, span)

    def prepend(self, span: Span) -> Span:
        """
        Prepend span to start of list.

        Sets span.owner to this BlockText.

        Returns:
            The prepended span
        """
        return self.insert_after(self._head, span)

    def insert_after(self, after: Span, span: Span) -> Span:
        """
        Insert span after the given span.

        Sets span.owner to this BlockText.

        Args:
            after: The span to insert after
            span: The span to insert

        Returns:
            The inserted span
        """
        next_span = after.next

        after.next = span
        span.prev = after
        span.next = next_span
        next_span.prev = span

        span.owner = self
        self._count += 1

        return span

    def insert_before(self, before: Span, span: Span) -> Span:
        """
        Insert span before the given span.

        Sets span.owner to this BlockText.

        Args:
            before: The span to insert before
            span: The span to insert

        Returns:
            The inserted span
        """
        return self.insert_after(before.prev, span)

    # -------------------------------------------------------------------------
    # Removal
    # -------------------------------------------------------------------------

    def remove(self, span: Span) -> Span:
        """
        Remove span from list.

        Clears span.owner and linked list pointers.

        Args:
            span: The span to remove

        Returns:
            The removed span
        """
        if span.owner is not self:
            raise ValueError("Span is not owned by this BlockText")

        prev_span = span.prev
        next_span = span.next

        prev_span.next = next_span
        next_span.prev = prev_span

        span.prev = None
        span.next = None
        span.owner = None

        self._count -= 1

        return span

    def clear(self) -> None:
        """Remove all spans from the list."""
        current = self._head.next
        while current is not self._tail:
            next_span = current.next
            current.prev = None
            current.next = None
            current.owner = None
            current = next_span

        self._head.next = self._tail
        self._tail.prev = self._head
        self._count = 0

    # -------------------------------------------------------------------------
    # Text Access
    # -------------------------------------------------------------------------

    def text(self) -> str:
        """
        Get full text by concatenating all spans.

        Note: This gives spans in linked list order. For correct
        rendering with tree structure, use Block.render().

        Returns:
            Concatenated text of all spans
        """
        return "".join(span.text for span in self)
    
    
    def text_between(self, start: Span, end: Span) -> str:
        """
        Get text between start and end.
        """
        return "".join(span.text for span in self.spans_between(start, end))
    


    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    @property
    def first(self) -> Span | None:
        """Get first span, or None if empty."""
        if self._count == 0:
            return None
        return self._head.next

    @property
    def last(self) -> Span | None:
        """Get last span, or None if empty."""
        if self._count == 0:
            return None
        return self._tail.prev

    def spans_between(self, start: Span, end: Span) -> list[Span]:
        """
        Get all spans between start and end (inclusive).

        Args:
            start: Starting span
            end: Ending span

        Returns:
            List of spans from start to end
        """
        spans = []
        current = start
        while current is not None and current is not self._tail:
            spans.append(current)
            if current is end:
                break
            current = current.next
        return spans

    def contains(self, span: Span) -> bool:
        """Check if span is in this BlockText."""
        return span.owner is self

    # -------------------------------------------------------------------------
    # Copy / Fork
    # -------------------------------------------------------------------------

    def fork(self, spans: list[Span] | None = None) -> "BlockText":
        """
        Create a new BlockText with copies of spans.

        If spans is None, copies all spans in this BlockText.
        Copied spans have no owner until added to the new BlockText.

        Args:
            spans: Optional list of spans to copy (default: all)

        Returns:
            New BlockText with copied spans
        """
        new_block_text = BlockText()

        source_spans = spans if spans is not None else list(self)
        for span in source_spans:
            new_span = span.copy()
            new_block_text.append(new_span)

        return new_block_text

    def copy(self) -> "BlockText":
        """
        Create a deep copy of this BlockText.

        Alias for fork() with no arguments.

        Returns:
            New BlockText with all spans copied
        """
        return self.fork()

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def extend(self, spans: list[Span], after: Span | None = None) -> list[Span]:
        """
        Insert multiple spans after the given span.

        Args:
            spans: List of spans to insert
            after: Insert after this span (default: append to end)

        Returns:
            The inserted spans
        """
        if after is None:
            after = self._tail.prev

        current = after
        for span in spans:
            self.insert_after(current, span)
            current = span

        return spans

    def extend_from(self, other: "BlockText", copy: bool = True) -> list[Span]:
        """
        Extend this BlockText with spans from another BlockText.

        Args:
            other: Source BlockText
            copy: If True, copy spans. If False, move spans (clears other).

        Returns:
            The added spans
        """
        if copy:
            spans = [span.copy() for span in other]
        else:
            spans = list(other)
            other.clear()

        return self.extend(spans)

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"BlockText(spans={self._count})"

    def debug(self) -> str:
        """Debug representation showing all spans."""
        lines = [f"BlockText(spans={self._count}):"]
        for i, span in enumerate(self):
            line = f"  [{span.id}] {span}"
            # if span.prev:
            #     line += f" prev={span.prev.id}"
            # if span.next:
            #     line += f" next={span.next.id}"

            # line = f"  [{id(span)}] {span}"
            # if span.prev:
            #     line += f" prev={id(span.prev)}"
            # if span.next:
            #     line += f" next={id(span.next)}"

            lines.append(line)
        return "\n".join(lines)
    

    def print_debug(self):
        """Print debug representation."""
        print(self.debug())

    def assert_chain_integrity(self) -> None:
        """
        Assert that all spans are properly chained with no dangling spans.

        Raises:
            AssertionError: If the chain is broken or inconsistent.
        """
        # 1. Traverse forward from head to tail and count spans
        forward_count = 0
        forward_spans: set[int] = set()
        current = self._head.next
        prev = self._head

        while current is not self._tail:
            # Check backward link
            assert current.prev is prev, (
                f"Broken backward link: span {current.id} has prev={current.prev.id if current.prev else None}, "
                f"expected {prev.id}"
            )
            # Check owner
            assert current.owner is self, (
                f"Span {current.id} has wrong owner: {id(current.owner)} != {id(self)}"
            )
            # Track for duplicate detection
            span_id = id(current)
            assert span_id not in forward_spans, (
                f"Cycle detected: span {current.id} appears twice in forward traversal"
            )
            forward_spans.add(span_id)

            forward_count += 1
            prev = current
            current = current.next

            # Safety: prevent infinite loop
            assert forward_count <= self._count + 1, (
                f"Forward traversal exceeded count ({self._count}), possible cycle"
            )

        # Check that we ended at tail
        assert current is self._tail, (
            f"Forward traversal did not end at tail, ended at {current.id if current else None}"
        )
        # Check tail's prev link
        assert self._tail.prev is prev, (
            f"Tail's prev is {self._tail.prev.id if self._tail.prev else None}, expected {prev.id}"
        )

        # 2. Traverse backward from tail to head and count spans
        backward_count = 0
        backward_spans: set[int] = set()
        current = self._tail.prev
        next_span = self._tail

        while current is not self._head:
            # Check forward link
            assert current.next is next_span, (
                f"Broken forward link: span {current.id} has next={current.next.id if current.next else None}, "
                f"expected {next_span.id}"
            )
            # Track for duplicate detection
            span_id = id(current)
            assert span_id not in backward_spans, (
                f"Cycle detected: span {current.id} appears twice in backward traversal"
            )
            backward_spans.add(span_id)

            backward_count += 1
            next_span = current
            current = current.prev

            # Safety: prevent infinite loop
            assert backward_count <= self._count + 1, (
                f"Backward traversal exceeded count ({self._count}), possible cycle"
            )

        # Check that we ended at head
        assert current is self._head, (
            f"Backward traversal did not end at head, ended at {current.id if current else None}"
        )
        # Check head's next link
        assert self._head.next is next_span, (
            f"Head's next is {self._head.next.id if self._head.next else None}, expected {next_span.id}"
        )

        # 3. Verify counts match
        assert forward_count == self._count, (
            f"Forward count ({forward_count}) != stored count ({self._count})"
        )
        assert backward_count == self._count, (
            f"Backward count ({backward_count}) != stored count ({self._count})"
        )
        assert forward_spans == backward_spans, (
            f"Forward and backward traversals found different spans"
        )