# Remove Root Span Refactor - Implementation Plan

## Summary
Remove the root span (path="0") and make Turn the root container. Top-level spans start from path="1", "2", etc.

---

## Models - COMPLETED ✅

### SpanValue ✅
- Added `turn_id` field (nullable, mutually exclusive with `span_id`)
- Added `path` field (LTREE)
- Added `index` property (computed from path)

### Artifact ✅
- Added `span_id` field (creation context)

### Turn ✅
- Added `values` RelationField (turn-level values)

### ExecutionSpan ✅
- Added `artifacts` RelationField (artifacts created in this span)
- `parent_span_id` already nullable

---

## SpanTree Changes - IN PROGRESS

### Core Changes Needed

#### 1. Path Computation
**Current**: All paths start with "0" (root span)
```python
Root: "0"
Child: "0.1"
Grandchild: "0.1.2"
```

**New**: Top-level spans start from "1"
```python
Top-level span 1: "1"
Top-level span 2: "2"
Child of span 1: "1.1"
Value of span 1: "1.0", "1.1", "1.2"
```

#### 2. SpanTree.__init__
**Current**:
```python
path = ".".join([str(i) for i in self.path])  # Uses self.path property
self.root = ExecutionSpan(..., path=path, parent_span_id=parent.id if parent else None)
```

**Issues**:
- `self.path` property depends on `parent.path` which may not be set yet
- Creates circular dependency

**Solution**:
```python
if isinstance(target, ExecutionSpan):
    self.root = target
else:
    # Compute path based on parent
    if parent is None:
        # Top-level span - need to count siblings
        # But we don't have turn context here!
        path = "1"  # Placeholder - will be set when saved
    else:
        # Child span
        child_index = len(parent.children)
        path = f"{parent.root.path}.{child_index}"

    self.root = ExecutionSpan(
        name=target,
        span_type=span_type,
        tags=tags,
        path=path,
        parent_span_id=parent.id if parent else None
    )
```

**Problem**: Top-level spans don't know their index without turn context!

#### 3. SpanTree.init_new - REMOVE
**Current**:
```python
@classmethod
async def init_new(cls, name: str, ...):
    span = await ExecutionSpan(name=name, path="0").save()
    return cls(span)
```

**Action**: DELETE this method - it creates root spans

#### 4. SpanTree.log_value - PARTIALLY DONE ✅
**Current updates**:
- ✅ Added `_value_index` counter in `__init__`
- ✅ Compute `value_path = f"{self.root.path}.{self._value_index}"`
- ✅ Increment `self._value_index`
- ✅ Added `span_id=self.id` to Artifact creation
- ✅ Added `path=value_path` to SpanValue creation

**Still needed**:
- Update `log_value2` if it's still used

#### 5. SpanTree.add_child - NEEDS UPDATE
**Current**:
```python
async def add_child(self, name: str, ...):
    span_tree = await SpanTree(name, span_type, tags, index=len(self.children), parent=self).save()
    await self.log_value(span_tree, io_kind="output")
    return span_tree
```

**Issue**: Path computation in `__init__` may be wrong

**Solution**:
```python
async def add_child(self, name: str, span_type: str = "component", tags: list[str] = []):
    # Compute child path
    child_index = len(self.children)
    child_path = f"{self.root.path}.{child_index}"

    # Create child span
    child_span = await ExecutionSpan(
        name=name,
        span_type=span_type,
        tags=tags,
        path=child_path,
        parent_span_id=self.id
    ).save()

    # Wrap in SpanTree
    span_tree = SpanTree(child_span, parent=self, index=child_index)

    # Log as value
    await self.log_value(span_tree, io_kind="output")

    return span_tree
```

#### 6. SpanTree.from_turn - MAJOR UPDATE NEEDED
**Current**: Loads root span at path="0"
```python
async def _from_turn(cls, turn_id: int, ...):
    spans = await ExecutionSpan.query(...).where(turn_id=...)
    # Loads all spans including root at "0"
```

**New**: Support Turn as root OR ExecutionSpan as root
```python
async def from_turn(cls, turn_id: int, span_id: int | None = None, branch_id: int | None = None):
    """
    Load span tree from turn.

    If span_id is None: Return TurnSpanTree (Turn as root)
    If span_id is provided: Return SpanTree starting from that span
    """
    if span_id is None:
        # Load turn with all top-level spans
        turn = await Turn.get(turn_id)
        return TurnSpanTree(turn, branch_id=branch_id)
    else:
        # Load specific span subtree (existing behavior)
        return await cls._load_span_subtree(span_id, branch_id)
```

#### 7. New Class: TurnSpanTree
**Purpose**: Wrap Turn as root container

```python
class TurnSpanTree:
    """
    SpanTree where Turn is the root.
    Top-level spans are direct children.
    """
    def __init__(self, turn: Turn, branch_id: int | None = None):
        self.turn = turn
        self.branch_id = branch_id or turn.branch_id
        self._values = []  # Turn-level values
        self._children = []  # Top-level spans
        self._value_index = 0  # For turn-level values
        self._span_index = 0  # For top-level spans

    @property
    def id(self):
        return f"turn_{self.turn.id}"

    @property
    def path(self):
        return ""  # Turn is root

    @property
    def children(self) -> list[SpanTree]:
        """Top-level spans"""
        return self._children

    async def add_child(self, name: str, span_type: str = "component", tags: list[str] = []):
        """Create top-level span"""
        # Compute path for top-level span
        span_path = str(self._span_index + 1)  # "1", "2", "3", ...
        self._span_index += 1

        # Create span
        span = await ExecutionSpan(
            name=name,
            span_type=span_type,
            tags=tags,
            path=span_path,
            parent_span_id=None  # No parent - top level
        ).save()

        # Wrap in SpanTree
        span_tree = SpanTree(span, parent=None, index=self._span_index - 1)
        self._children.append(span_tree)

        return span_tree

    async def log_value(self, target: Any, ...):
        """Log turn-level value"""
        value_path = str(self._value_index)  # "0", "1", "2", ...
        self._value_index += 1

        # Create SpanValue with turn_id
        value = await SpanValue(
            turn_id=self.turn.id,
            span_id=None,
            path=value_path,
            kind=...,
            artifact_id=...,
            ...
        ).save()

        # Add to values
        v = Value(value, target)
        self._values.append(v)
        return v
```

#### 8. Context Integration
**Where SpanTrees are created**:

**Option A**: Context creates TurnSpanTree
```python
class Context:
    async def start_turn(self):
        async with self._branch.start_turn() as turn:
            self.turn = turn
            self.turn_span_tree = TurnSpanTree(turn)  # NEW
            yield self
```

**Option B**: Keep current pattern, remove root span creation
- Current code might create root spans in components
- Need to audit all `SpanTree()` creations

---

## Migration Strategy

### Database Changes
1. Drop database (as you said)
2. Run with new schema

### Code Changes Order

1. ✅ **Models updated**
2. **SpanTree updates**:
   - [x] Add `_value_index` counter
   - [x] Update `log_value` to use path
   - [ ] Delete `init_new` method
   - [ ] Update `add_child` path computation
   - [ ] Create `TurnSpanTree` class
   - [ ] Update `from_turn` to support Turn as root
3. **Context updates**:
   - [ ] Remove root span creation
   - [ ] Integrate TurnSpanTree
4. **Component updates**:
   - [ ] Audit all SpanTree() creations
   - [ ] Fix path computations

---

## Testing Strategy

### Unit Tests
1. Test top-level span creation (path="1", "2", etc.)
2. Test child span creation (path="1.1", "1.2", etc.)
3. Test SpanValue paths (path="1.0", "1.1", etc.)
4. Test turn-level values (path="0", "1", "2", etc.)

### Integration Tests
1. Test multi-step agent (pirate example)
2. Test single component execution
3. Test nested spans (3+ levels)
4. Test `from_turn` loading

---

## Open Questions

1. **Do we need TurnSpanTree as separate class?**
   - Pro: Clean separation
   - Con: More complexity
   - Alternative: Make SpanTree work with Turn OR ExecutionSpan as root

2. **How to handle top-level span indexing?**
   - Current: Count from parent
   - Issue: No parent for top-level spans
   - Solution: TurnSpanTree tracks `_span_index`

3. **Backward compatibility?**
   - Decision: No - breaking change, drop database

---

## Next Steps

Choose approach:

**Option 1**: Create TurnSpanTree class (cleaner separation)
**Option 2**: Make SpanTree polymorphic (root: Turn | ExecutionSpan)

Which do you prefer?
