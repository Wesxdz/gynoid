# Layout System Documentation

## Components

### Position (Local/World)
```cpp
typedef Vector2 Position;  // {x, y}
struct Local {};  // Tag for local position (relative to parent)
struct World {};  // Tag for world position (absolute)
```
- `Position, Local` - Position relative to parent entity
- `Position, World` - Absolute position computed from hierarchy

### UIElementSize
```cpp
struct UIElementSize {
    float width, height;
};
```
- Represents the size of a UI element
- Set by renderables, measured text, or layout systems

### UIElementBounds
```cpp
struct UIElementBounds {
    float xmin, ymin, xmax, ymax;
};
```
- Absolute world coordinates of element's bounding box
- Computed from `Position, World` + `UIElementSize`

### LayoutBox
```cpp
struct LayoutBox {
    enum Direction { Horizontal, Vertical };
    Direction dir = Horizontal;
    float padding = 0.0f;
    float move_dir = 1.0f;  // 1 = right/down, -1 = left/up
};
```
- Positions children sequentially (horizontal or vertical)
- Updates own `UIElementSize` based on children's sizes

### UIContainer
```cpp
struct UIContainer {
    int pad_horizontal;
    int pad_vertical;
};
```
- Entity that sizes itself based on children's extent
- Adds padding around children
- Used with RoundedRectRenderable or CustomRenderable

### Expand
```cpp
struct Expand {
    bool x_enabled;
    float pad_left, pad_right;
    float x_percent;  // 0.0 to 1.0
    bool y_enabled;
    float pad_top, pad_bottom;
    float y_percent;
};
```
- Makes entity expand to fill parent's bounds
- Excluded from parent's bounds/extent calculation

---

## 3-Phase Layout System

### Phase 1: Initialize Sizes & Propagate World Positions (PreFrame/OnLoad)

**Phase 1a: Hierarchical Positioning** (OnLoad)
- Query: Position Local, parent Position World → Position World
- Logic: `world = local + parent_world`
- Uses cascade for breadth-first parent-to-child order

**Phase 1b: Bounds Calculation** (OnLoad)
- Query: Position World, UIElementSize → UIElementBounds
- Logic: `bounds = {world.x, world.y, world.x + size.width, world.y + size.height}`

**Phase 1c: Size from Renderables** (PreFrame)
- Query: UIElementSize, various Renderables
- Logic: Copy renderable dimensions to UIElementSize
- Skips RoundedRectRenderable if entity has UIContainer

### Phase 2: Layout (PostLoad)

**LayoutBoxSystem** (PostLoad, cascade().desc() for bottom-up)
- Query: LayoutBox, UIElementSize
- For each LayoutBox entity (children processed before parents):
  1. Iterate direct children with UIElementSize
  2. Set child's Position Local based on accumulated progress
  3. **Propagate world positions to child and all descendants**
  4. Accumulate main_progress and cross_max
  5. Set own UIElementSize from accumulated dimensions

**UIContainer Systems** (PostLoad)
- Query: UIContainer, Renderable, UIElementSize
- For each UIContainer entity:
  1. Calculate extent from children's Position Local + UIElementSize
  2. Set renderable.width/height = extent + padding
  3. Set UIElementSize = renderable dimensions

### Phase 3: Bounds Containment (PostLoad)

**bubbleUpBoundsSystem / bubbleUpBoundsSecondarySystem** (PostLoad)
- Query: UIElementBounds, parent UIElementBounds
- Logic: Expand parent bounds to contain child (if child doesn't have Expand)

---

## Helper Functions

### propagate_world_positions(entity)
Recursively updates Position World for entity and all descendants.
Called by LayoutBoxSystem after positioning each child.

### calculate_children_extent(parent, max_x, max_y, found)
Calculates bounding box from children's Position Local + UIElementSize.
Used by UIContainer to size itself without depending on world bounds.

### propagate_bounds_containment(parent)
Recursively propagates bounds containment from children to parent.
Children's bounds are finalized before contributing to parent.

---

## Key Design Principles

1. **Local coordinates for sizing**: UIContainer and LayoutBox use Position Local + UIElementSize to calculate extent, not UIElementBounds. This avoids feedback loops from stale world bounds.

2. **Immediate position propagation**: When LayoutBox positions a child, it immediately propagates world positions to that child's descendants. This ensures nested LayoutBoxes see correct world positions.

3. **Bottom-up layout**: LayoutBoxSystem uses cascade().desc() to process children before parents. A child LayoutBox calculates its size before its parent tries to position it.

4. **Clear phase separation**:
   - Phase 1: Sizes and positions initialized
   - Phase 2: Layout systems modify positions and sizes
   - Phase 3: Bounds calculated and containment propagated
