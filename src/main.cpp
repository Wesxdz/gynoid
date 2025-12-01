#include <iostream>
#include <cmath>
#include <vector>
#include <stack>
#include <algorithm>
#include <variant>
#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <flecs.h>
#include <nanovg.h>
#include <nanovg_gl.h>
#include <float.h>
#include <unordered_map>
#include <raymath.h>

/*
These luminous phenomena still manifest themselves
from time to time, as when a new idea opening up possibilities
strikes me, but they are no longer exciting, being of relatively
small intensity. When I close my eyes I invariably observe first,
a background of very dark and uniform blue, not unlike the
sky on a clear but starless night. In a few seconds this field
becomes animated with innumerable scintillating flakes of
green, arranged in several layers and advancing towards me.
Then there appears, to the right, a beautiful pattern of two
systems of parallel and closely spaced lines, at right angles to
one another, in all sorts of colors with yellow-green and gold
predominating. Immediately thereafter the lines grow brighter and the whole is thickly sprinkled with dots of twinkling
light. This picture moves slowly across the field of vision and
in about ten seconds vanishes to the left, leaving behind a
ground of rather unpleasant and inert grey which quickly
gives way to a billowy sea of clouds, seemingly trying to mould
themselves in living shapes. It is curious that I cannot project a
form into this grey until the second phase is reached. Every
time, before falling asleep, images of persons or objects flit
before my view. When I see them I know that I am about to lose
consciousness. If they are absent and refuse to come it means a
sleepless night.
*/

// ECS Components

// struct Position {
//     float x, y;
// };

typedef Vector2 Position;

struct Edge
{
    Position p0;
    Position p1;
};

struct Local {};
struct World {};

struct Velocity {
    float dx, dy;
};

struct UIElementSize
{
    float width, height;
};

struct UIElementBounds {
    float xmin, ymin, xmax, ymax;
};

struct RenderStatus {
    bool visible;
};

struct HorizontalLayoutBox
{
  float x_progress;
  float padding = 0.0f;
};

struct VerticalLayoutBox 
{
  float y_progress;
  float padding;
};


struct RectRenderable {
    float width, height;
    bool stroke;
    uint32_t color;
};

struct RoundedRectRenderable {
    float width, height, radius;
    bool stroke;
    uint32_t color;
};

struct TextRenderable {
    std::string text;
    std::string fontFace;
    float fontSize;
    uint32_t color;
    int alignment;
};

struct ImageCreator
{
    std::string path;
    float scaleX, scaleY;
};

struct ImageRenderable
{
    int imageHandle;
    float scaleX, scaleY;

    float width, height;
};

struct ZIndex {
    int layer;
};

struct Window {
    GLFWwindow* handle;
    int width, height;
};

struct CursorState
{
    double x, y;
};

struct AddTagOnLeftClick{};
struct ShowEditorPanels {};

struct LeftClickEvent {};
struct Dragging {};
struct DynamicPartition {};
struct DynamicMerge {};
struct LeftReleaseEvent {};
struct RightClick {};
struct RightRelease {};

struct AddTagOnHoverEnter {};
struct HoverEnterEvent {};

struct AddTagOnHoverExit {};
struct HoverExitEvent {};

struct CloseEditorSelector {};
struct SetMenuHighlightColor {};
struct SetMenuStandardColor {};

struct Graphics {
    NVGcontext* vg;
};

enum class EditorType
{
    Bookshelf,
    MelSpectrogram,
    VNCStream,
    CameraVideoFeed,
    RobotActuators,
    VirtualHumanoid,
    EpisodicMemoryTimeline,
    Reification,
    ReadingRepresentation,
    ProgramSynthesis,
    Servers,
    Healthbar,
    Inventory,
    Scheduling,
    Chat,
    ComputeProvisioning,
    Backup
};

// Node representing the editor area...
struct EditorNodeArea
{
    float width, height;
};

struct EditorShiftRegion
{
    UIElementBounds bounds;
    int cursor_type; // ex, GLFW_CROSSHAIR_CURSOR
    flecs::entity split_target;
};

struct EditorModifyPartitionRegion
{
    UIElementBounds bounds;
    flecs::entity split_target;
};

struct EditorRoot 
{
    std::vector<EditorShiftRegion> shift_regions;
    std::vector<EditorModifyPartitionRegion> modify_partition_regions;
};

struct EditorVisual {};
struct EditorOutline {};

struct UpperNode {};
struct LowerNode {};

struct LeftNode {};
struct RightNode {};

struct VerticalAlign
{
    float percent;
};

struct HorizontalAlign
{
    float percent;
};

struct SelfAlign
{
    float horizontal;
    float vertical;
};

// Expand Rect or RoundedRect to UIElement bounds of parent
// with some padding
struct Expand
{
    bool x_enabled;
    float pad_left, pad_right;
    float x_percent; // 0.0 to 1.0

    bool y_enabled;
    float pad_top, pad_bottom;
    float y_percent;
};

struct EditorLeafData
{
    EditorType editor_type;
};

enum class PanelSplitType
{
    Horizontal,
    Vertical
};

struct DragContext {
    bool active;
    flecs::entity target;
    PanelSplitType dim;
    float startPercent;
};

struct PanelSplit 
{
    // use percent instead of pixel count for proportional expansion when
    // window size changes
    float percent; // 0 to 100
    PanelSplitType dim;
};

enum class RenderType {
    Rectangle,
    RoundedRectangle,
    Text,
    Image
};

flecs::world world;

void create_editor(flecs::entity leaf, EditorNodeArea& node_area, flecs::world world, flecs::entity UIElement)
{
    leaf.set<EditorLeafData>({EditorType::VNCStream});

    auto editor_visual = world.entity()
        .is_a(UIElement)
        .set<Position, Local>({1.0f, 1.0f})
        .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, false, 0x010222})
        .child_of(leaf);

    leaf.add<EditorVisual>(editor_visual);

    auto editor_outline = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x111222FF})
        .set<ZIndex>({5});

    leaf.add<EditorOutline>(editor_outline);

    // Add 'expand to parent UIElement bounds with padding'
    auto editor_canvas = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({8.0f, 23.0f})
        .set<RectRenderable>({node_area.width-2-8.0f, node_area.height-2-23.0f, false, 0x121212FF})
        .set<Expand>({true, 8.0f, 8.0f, 1.0f, true, 27.0f, 4.0f, 1.0f})
        .set<ZIndex>({8});

    auto editor_icon_bkg = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<Position, Local>({8.0f, 2.0f})
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, false, 0x282828FF})
        .add<AddTagOnLeftClick, ShowEditorPanels>()
        .set<ZIndex>({4});

    auto editor_icon = world.entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({2.0f, 0.0f})
        .set<ImageCreator>({"../assets/graphstar_icon.png", 1.0f, 1.0f})
        .set<ZIndex>({12});

    auto editor_dropdown = world.entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<Position, Local>({22.0f, 8.0f})
        .set<ImageCreator>({"../assets/arrow_down.png", 1.0f, 1.0f})
        .set<ZIndex>({12});
        
    world.entity()
        .is_a(UIElement)
        .child_of(editor_icon_bkg)
        .set<RoundedRectRenderable>({32.0f, 20.0f, 4.0f, true, 0x5f5f5fFF})
        .set<ZIndex>({6});

    // TODO: Editor type entities...

    if (leaf.has<EditorLeafData>() && leaf.get<EditorLeafData>().editor_type == EditorType::Bookshelf)
    {
    // For bookshelf, we should create the horizontal layout with all the books
    auto bookshelf_layer = world.entity()
        .is_a(UIElement)
        .child_of(editor_visual)
        .set<HorizontalLayoutBox>({0.0f, 8.0f});
        
    world.entity()
    .is_a(UIElement)
    .child_of(bookshelf_layer)
    .set<Position, Local>({4.0f, 23.0f})
    .set<ImageCreator>({"../assets/cover_james.jpg", 1.0f, 1.0f})
    .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 27.0f, 0.0f, 1.0f})
    .set<ZIndex>({10});

    world.entity()
    .is_a(UIElement)
    .child_of(bookshelf_layer)
    .set<Position, Local>({4.0f, 23.0f})
    .set<ImageCreator>({"../assets/cover_cognitive_theory.jpg", 1.0f, 1.0f})
    .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 27.0f, 0.0f, 1.0f})
    .set<ZIndex>({10});

    world.entity()
    .is_a(UIElement)
    .child_of(bookshelf_layer)
    .set<Position, Local>({4.0f, 23.0f})
    .set<ImageCreator>({"../assets/cover_soar.jpg", 1.0f, 1.0f})
    .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 27.0f, 0.0f, 1.0f})
    .set<ZIndex>({10});

    world.entity()
    .is_a(UIElement)
    .child_of(bookshelf_layer)
    .set<Position, Local>({4.0f, 23.0f})
    .set<ImageCreator>({"../assets/cover_readings_in_kr.jpg", 1.0f, 1.0f})
    .set<Expand>({false, 4.0f, 4.0f, 1.0f, true, 27.0f, 0.0f, 1.0f})
    .set<ZIndex>({10});
    }
    
}

void merge_editor(flecs::entity non_leaf, flecs::world world, flecs::entity UIElement)
{
    non_leaf.remove<PanelSplit>();
    non_leaf.remove<LeftNode>(flecs::Wildcard);
    non_leaf.remove<RightNode>(flecs::Wildcard);
    non_leaf.remove<UpperNode>(flecs::Wildcard);
    non_leaf.remove<LowerNode>(flecs::Wildcard);
    non_leaf.children([](flecs::entity child) 
    {
        child.destruct();
    });
    EditorNodeArea& intermediate_area = non_leaf.ensure<EditorNodeArea>();
    create_editor(non_leaf, intermediate_area, world, UIElement);
}

void split_editor(PanelSplit split, flecs::entity leaf, flecs::world world, flecs::entity UIElement)
{
    // Destroy any existing visual
    flecs::entity existing_visual = leaf.target<EditorVisual>();
    if (existing_visual.is_valid() && existing_visual.is_alive())
    {
        leaf.remove<EditorLeafData>();
        existing_visual.destruct();
    }
    leaf.set<PanelSplit>(split);
    const EditorNodeArea* node_area = leaf.try_get<EditorNodeArea>();
    // We want to reuse the existing editor_area by moving it from the leaf to the left_editor...
    if (split.dim == PanelSplitType::Horizontal)
    {   
        EditorNodeArea left_node_area = {node_area->width*split.percent, node_area->height};
        auto left_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(left_node_area)
            .add(flecs::OrderedChildren);
        create_editor(left_editor_leaf, left_node_area, world, UIElement);
        leaf.add<LeftNode>(left_editor_leaf);

        EditorNodeArea right_node_area = {node_area->width*(1.0f-split.percent), node_area->height};
        auto right_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({node_area->width*split.percent, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(right_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(right_editor_leaf, right_node_area, world, UIElement);
        leaf.add<RightNode>(right_editor_leaf);
    }
    if (split.dim == PanelSplitType::Vertical)
    {   
        EditorNodeArea upper_node_area = {node_area->width, node_area->height*split.percent};
        auto upper_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, 0.0f})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(upper_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(upper_editor_leaf, upper_node_area, world, UIElement);
        leaf.add<UpperNode>(upper_editor_leaf);
        
        EditorNodeArea lower_node_area = {node_area->width, node_area->height*(1-split.percent)};
        auto lower_editor_leaf = world.entity()
            .child_of(leaf)
            .set<Position, Local>({0.0f, node_area->height*split.percent})
            .set<Position, World>({0.0f, 0.0f})
            .set<EditorNodeArea>(lower_node_area) 
            .add(flecs::OrderedChildren);
        create_editor(lower_editor_leaf, lower_node_area, world, UIElement);
        leaf.add<LowerNode>(lower_editor_leaf);
    }
}

struct RenderCommand {
    Position pos;
    std::variant<RoundedRectRenderable, RectRenderable, TextRenderable, ImageRenderable> renderData;
    RenderType type;
    int zIndex;

    bool operator<(const RenderCommand& other) const {
        return zIndex < other.zIndex;
    }
};

struct RenderQueue {
    std::vector<RenderCommand> commands;

    void clear() {
        commands.clear();
    }

    void addRectCommand(const Position& pos, const RectRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Rectangle, zIndex});
    }

    void addRoundedRectCommand(const Position& pos, const RoundedRectRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::RoundedRectangle, zIndex});
    }

    void addTextCommand(const Position& pos, const TextRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Text, zIndex});
    }

    void addImageCommand(const Position& pos, const ImageRenderable& renderable, int zIndex) {
        commands.push_back({pos, renderable, RenderType::Image, zIndex});
    }

    void sort() {
        std::sort(commands.begin(), commands.end());
    }
};

void error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    Window& window_comp = world.lookup("GLFWState").ensure<Window>();
    window_comp.width = width;
    window_comp.height = height;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

bool point_in_bounds(float x, float y, UIElementBounds bounds)
{
    return (x >= bounds.xmin && x <= bounds.xmax && y >= bounds.ymin && y <= bounds.ymax);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    // TODO: Move to Observer?
    CursorState& cursor_state = world.lookup("GLFWState").ensure<CursorState>();
    // TODO: Query for hoverable UIElement 
    flecs::query hoverable_elements = world.query_builder<AddTagOnHoverEnter, AddTagOnHoverExit, UIElementBounds>()
    .term_at(0).second(flecs::Wildcard).optional()
    .term_at(1).second(flecs::Wildcard).optional()
    .build();

    hoverable_elements.each([&](flecs::entity ui_element, AddTagOnHoverEnter, AddTagOnHoverExit, UIElementBounds& bounds) {
        bool in_bounds_prior = point_in_bounds(cursor_state.x, cursor_state.y, bounds);
        bool in_bounds_post = point_in_bounds(xpos, ypos, bounds);
        
        if (!in_bounds_prior && in_bounds_post && ui_element.has<AddTagOnHoverEnter>(flecs::Wildcard))
        {   
            world.event<HoverEnterEvent>()
            .id<UIElementBounds>()
            .entity(ui_element)
            .enqueue();
        } else if (in_bounds_prior && !in_bounds_post && ui_element.has<AddTagOnHoverExit>(flecs::Wildcard))
        {
            // TODO: Store hover state...
            world.event<HoverExitEvent>()
            .id<UIElementBounds>()
            .entity(ui_element)
            .enqueue();  
        }
    });

    cursor_state.x = xpos;
    cursor_state.y = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    flecs::entity glfw_state = world.lookup("GLFWState");
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            flecs::query interactive_elements = world.query_builder<AddTagOnLeftClick, UIElementBounds>()
            .term_at(0).second(flecs::Wildcard)
            .build();
            // TODO: Eventually this should use a more efficient partition bound check as the first layer
            const CursorState* cursor_state = world.lookup("GLFWState").try_get<CursorState>();
            interactive_elements.each([&](flecs::entity ui_element, AddTagOnLeftClick, UIElementBounds& bounds) {
                if (point_in_bounds(cursor_state->x, cursor_state->y, bounds))
                {
                    world.event<LeftClickEvent>()
                    .id<UIElementBounds>()
                    .entity(ui_element)
                    .emit();
                } 
            });
            world.event<LeftClickEvent>()
            .id<CursorState>()
            .entity(glfw_state)
            .emit();
        } else if (action == GLFW_RELEASE)
        {
            world.event<LeftReleaseEvent>()
            .id<CursorState>()
            .entity(glfw_state)
            .emit();
        }
    }
}

float point_distance_to_edge(Position p, Position a, Position b)
{
    // 1. Get vector from line start (a) to point (p)
    Vector2 v = Vector2Subtract(p, a);
    // 2. Get vector of the edge itself (a -> b)
    Vector2 edge = Vector2Subtract(b, a);
    // 3. Project v onto edge (returns normalized position t along the line)
    float edge_len_sq = Vector2LengthSqr(edge);
    if (edge_len_sq == 0.0f) return Vector2Distance(p, a); // Safety check
    float t = Vector2DotProduct(v, edge) / edge_len_sq;
    // 4. Clamp t to ensure we stay on the segment (0 to 1)
    t = Clamp(t, 0.0f, 1.0f);
    // 5. Calculate closest point: a + (edge * t)
    Vector2 closest_point = Vector2Add(a, Vector2Scale(edge, t));
    // 6. Return distance
    return Vector2Distance(p, closest_point);
}

int main(int, char *[]) {
    glfwSetErrorCallback(error_callback);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "ECS Graphstar", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, 800, 600);
    NVGcontext* vg = nvgCreateGL2(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
    if (vg == NULL) {
        std::cerr << "Failed to initialize NanoVG" << std::endl;
        glfwTerminate();
        return -1;
    }

    world.component<Position>();
    world.component<Velocity>();
    world.component<RectRenderable>();
    world.component<TextRenderable>();
    world.component<ImageCreator>();
    world.component<ImageRenderable>();
    world.component<ZIndex>();
    world.component<Window>();
    world.component<CursorState>();
    world.component<Graphics>().add(flecs::Singleton);
    world.component<RenderQueue>();
    world.component<UIElementBounds>();
    world.component<UIElementSize>();

    world.component<EditorNodeArea>();
    world.component<PanelSplit>();

    world.component<HorizontalAlign>();
    world.component<VerticalAlign>();
    world.component<SelfAlign>();
    world.component<Expand>();

    world.component<HorizontalLayoutBox>();
    world.component<VerticalLayoutBox>();

    world.component<DragContext>().add(flecs::Singleton);
    world.set<DragContext>({false, flecs::entity::null(), PanelSplitType::Horizontal, 0.0f});

    world.observer<ImageCreator, Graphics>()
    .event(flecs::OnSet)
    .each([&](flecs::entity e, ImageCreator& img, Graphics& graphics)
    {
        int imgHandle = nvgCreateImage(graphics.vg, ("../assets/" + img.path).c_str(), 0);

        if (imgHandle == -1) {
            std::cerr << "Failed to load " << img.path << std::endl;
        }
        e.set<ImageRenderable>({imgHandle, img.scaleX, img.scaleY, 0.0f, 0.0f});
    });

    world.observer<ImageRenderable, Graphics>()
    .event(flecs::OnSet)
    .each([&](flecs::entity e, ImageRenderable& img, Graphics& graphics)
    {
        int nativeWidth, nativeHeight;
        nvgImageSize(graphics.vg, img.imageHandle, &nativeWidth, &nativeHeight);
        img.width = nativeWidth * img.scaleX;
        img.height = nativeHeight * img.scaleY;
    });

    double cursorXPos, cursorYPos;
    glfwGetCursorPos(window, &cursorXPos, &cursorYPos);

    auto glfwStateEntity = world.entity("GLFWState")
        .set<Window>({window, 800, 600})
        .set<CursorState>({cursorXPos, cursorYPos});

    auto graphicsEntity = world.entity("Graphics")
        .set<Graphics>({vg});

    auto renderQueueEntity = world.entity("RenderQueue")
        .set<RenderQueue>({});

    auto UIElement = world.prefab("UIElement")
        .set<Position, Local>({0.0f, 0.0f})
        .set<Position, World>({0.0f, 0.0f})
        .set<UIElementBounds>({FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX})
        .set<UIElementSize>({0.0f, 0.0f})
        .set<RenderStatus>({true})
        .set<ZIndex>({0});

    // TODO: Possibly load this from string
    std::vector<std::string> editor_types = 
    {
        "Void",
        // "ECS Graph", // Entity component relationship
        "Peach Core",
        // "Gynoid",
        "Embodiment",
        "Language Game", // Queue or Stream
        "Vision",
        "Hearing",
        "Memory",
        "Bookshelf",
    };

    world.observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<CloseEditorSelector>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        std::cout << "Hover exit editor selector region" << std::endl;
        e.destruct();
    });

    world.observer<UIElementBounds, AddTagOnHoverEnter>()
    .term_at(1).second<SetMenuHighlightColor>()
    .event<HoverEnterEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverEnter)
    {
        RoundedRectRenderable& bkg = e.ensure<RoundedRectRenderable>();
        bkg.color = 0x585858FF;
    });

    world.observer<UIElementBounds, AddTagOnHoverExit>()
    .term_at(1).second<SetMenuStandardColor>()
    .event<HoverExitEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnHoverExit)
    {
        RoundedRectRenderable& bkg = e.ensure<RoundedRectRenderable>();
        bkg.color = 0x383838FF;
    });

    world.observer<UIElementBounds, AddTagOnLeftClick>()
    .term_at(1).second<ShowEditorPanels>()
    .event<LeftClickEvent>()
    .each([&](flecs::entity e, UIElementBounds& bounds, AddTagOnLeftClick)
    {
        std::cout << "Left mouse click event on " << e.id() << std::endl;
        // Popup a menu to selector editor type (or close it)
        bool has_close_child = false;
        e.children([&](flecs::entity child) 
        {
            if (child.has<AddTagOnHoverExit, CloseEditorSelector>())
            {
                child.destruct();
                has_close_child = true;
            }
        });
        if (has_close_child) return;

        auto editor_hover_region = world.entity()
        .is_a(UIElement)
        .child_of(e)
        .add<AddTagOnHoverExit, CloseEditorSelector>();

        auto editor_icon_bkg_square = world.entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 10.0f})
        .set<RectRenderable>({32.0f, 12.0f, false, 0x282828FF})
        .set<ZIndex>({7});

        auto editor_type_selector = world.entity()
        .is_a(UIElement)
        .child_of(editor_hover_region)
        .set<Position, Local>({-1.0f, 19.0f});

        auto editor_type_selector_square_corner = world.entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<RectRenderable>({16.0f, 16.0f, false, 0x282828FF})
        .set<ZIndex>({30});

        auto editor_type_selector_bkg = world.entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<RoundedRectRenderable>({196.0f, 256.0f, 4.0f, false, 0x282828FF})
        .set<Expand>({false, 0, 0, 1.0f, true, 0.0f, -4.0f, 1.0f})
        .set<ZIndex>({30});

        auto editor_type_list = world.entity()
        .is_a(UIElement)
        .child_of(editor_type_selector)
        .set<VerticalLayoutBox>({0.0f, 2.0f})
        .set<Position, Local>({4.0f, 4.0f});

        for (std::string editor_type_name : editor_types)
        {
            // When you click on these elements,
            // change the EditorType
            // Remove any existing type scene content
            // and load the new default scene...
            auto edtior_type_btn = world.entity()
            .is_a(UIElement)
            .child_of(editor_type_list)
            .set<RoundedRectRenderable>({196.0f-12.0f, 20.0f, 2.0f, false, 0x383838FF})
            .set<Position, Local>({2.0f, 0.0f})
            .add<AddTagOnHoverEnter, SetMenuHighlightColor>()
            .add<AddTagOnHoverExit, SetMenuStandardColor>()
            .set<ZIndex>({38});


            world.entity()
            .is_a(UIElement)
            .child_of(edtior_type_btn)
            .set<TextRenderable>({editor_type_name.c_str(), "ATARISTOCRAT", 16.0f, 0xFFFFFFFF, NVG_ALIGN_TOP | NVG_ALIGN_LEFT})
            .set<Position, Local>({4.0f, 2.0f})
            .set<ZIndex>({40});
        }

    });

    // Create text entities with different z-indices
    // auto text1 = world.entity("Text1")
    //     .is_a(UIElement)
    //     .set<Position, Local>({400.0f, 100.0f})
    //     .set<TextRenderable>({"Behind boxes", "ATARISTOCRAT", 24.0f, 0xFFFFFFFF, NVG_ALIGN_CENTER})
    //     .set<ZIndex>({0});

    auto movementSystem = world.system<Position, Velocity>()
    .term_at(0).second<Local>()
        .each([](flecs::iter& it, size_t i, Position& pos, Velocity& vel) {
            float deltaTime = it.delta_system_time();

            pos.x += vel.dx * deltaTime;
            pos.y += vel.dy * deltaTime;
        });

    // Hierarchical positioning system - computes world positions from local positions
    auto hierarchicalQuery = world.query_builder<const Position, const Position*, Position>()
        .term_at(0).second<Local>()      // Local position
        .term_at(1).second<World>()      // Parent world position
        .term_at(2).second<World>()      // This entity's world position
        .term_at(1).parent().cascade()   // Get parent position in breadth-first order
        .build();

    auto hierarchicalSystem = world.system()
        .kind(flecs::OnLoad)  // Run after layout systems to compute world positions
        .each([&]() {
            // std::cout << "Update hierarchy" << std::endl;
            hierarchicalQuery.each([](const Position& local, const Position* parentWorld, Position& world) {
                world.x = local.x;
                world.y = local.y;
                if (parentWorld) {
                    world.x += parentWorld->x;
                    world.y += parentWorld->y;
                }
            });
        });

    int editor_padding = 3.0f;
    int editor_edge_hover_dist = 8.0f;

    auto editor_root = world.entity("editor_root")
        .set<Position, Local>({0.0f, 28.0f})
        .set<Position, World>({0.0f, 0.0f})
        .set<EditorNodeArea>({800.0f, 600.0f - 28.0f}) // TOOD: Observer to update root to window width/height updates
        .add<EditorRoot>()
        .add(flecs::OrderedChildren);

    auto editor_header = world.entity()
        .is_a(UIElement)
        .set<ImageCreator>({"../assets/ecs_header.png", 1.0f, 1.0f})
        .set<ZIndex>({5});

    // create_editor(editor_root, world, UIElement);
    split_editor({0.5, PanelSplitType::Horizontal}, editor_root, world, UIElement);
    auto right_node = editor_root.target<RightNode>();
    split_editor({0.35, PanelSplitType::Vertical}, right_node, world, UIElement);
    auto left_node = editor_root.target<LeftNode>();
    split_editor({0.25, PanelSplitType::Vertical}, left_node, world, UIElement);

    auto propagateEditorRoot = world.system<Window, EditorNodeArea, EditorRoot>("EditorPropagate")
    .term_at(0).src(glfwStateEntity)
    .kind(flecs::PreFrame)
    .run([](flecs::iter& it)
    {
        while (it.next()) {
            auto window = it.field<const Window>(0);
            auto node_area = it.field<EditorNodeArea>(1);
            for (auto i : it) {
                node_area[i].width = window[i].width;
                node_area[i].height = window[i].height-28.0f; // minus the header...
            }
            flecs::entity editor_root = world.lookup("editor_root");

            std::stack<flecs::entity> editors_to_visit;
            editors_to_visit.push(editor_root);
            while (!editors_to_visit.empty())
            {
                flecs::entity editor = editors_to_visit.top();
                editors_to_visit.pop();

                bool is_leaf = editor.has<EditorVisual>(flecs::Wildcard);
                EditorNodeArea& node_area = editor.ensure<EditorNodeArea>();
                // Update EditorVisual/EditorOutline if the node is a leaf
                // This could be refactored to an independent system...
                if (is_leaf)
                {
                    flecs::entity visual = editor.target<EditorVisual>();
                    // TODO: Only modify first two params...
                    visual.set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, false, 0x010222});
                    flecs::entity outline = editor.target<EditorOutline>();
                    outline.set<RoundedRectRenderable>({node_area.width-2, node_area.height-2, 4.0f, true, 0x5f5f5fFF});
                }

                PanelSplit* split = editor.try_get_mut<PanelSplit>();
                // Update child EditorNodeAreas if the panel is split
                if (split)
                {
                    if (split->dim == PanelSplitType::Horizontal)
                    {
                        flecs::entity left_node = editor.target<LeftNode>();
                        left_node.set<EditorNodeArea>({node_area.width*split->percent, node_area.height});
                        editors_to_visit.push(left_node);
                        flecs::entity right_node = editor.target<RightNode>();
                        right_node.set<EditorNodeArea>({node_area.width*(1-split->percent), node_area.height});
                        right_node.set<Position, Local>({node_area.width*split->percent, 0.0f});
                        editors_to_visit.push(right_node);
                    } else
                    {
                        flecs::entity upper_node = editor.target<UpperNode>();
                        upper_node.set<EditorNodeArea>({node_area.width, node_area.height*split->percent});
                        editors_to_visit.push(upper_node);
                        flecs::entity lower_node = editor.target<LowerNode>();
                        lower_node.set<EditorNodeArea>({node_area.width, node_area.height*(1-split->percent)});
                        lower_node.set<Position, Local>({0.0f, node_area.height*split->percent});
                        editors_to_visit.push(lower_node);
                    }
                }
            }
        }
    });

    auto sizeCalculationSystem = world.system<UIElementSize>()
        .kind(flecs::PreFrame)
        .each([&](flecs::entity e, UIElementSize& size) {

            if (e.has<RectRenderable>()) {
                auto rect = e.get<RectRenderable>();
                size.width = rect.width;
                size.height = rect.height;
            }
            else if (e.has<RoundedRectRenderable>()) {
                auto rect = e.get<RoundedRectRenderable>();
                size.width = rect.width;
                size.height = rect.height;
                //std::cout << "Setting size to" << size.width << std::endl;
            } else if (e.has<ImageRenderable>()) {
                auto img = e.get<ImageRenderable>();
                size.width = img.width;
                size.height = img.height;
            } else if (e.has<TextRenderable>()) {
                auto text = e.get<TextRenderable>();
                // Approximate text bounds
                float approxWidth = text.text.length() * text.fontSize * 0.6f;
                float approxHeight = text.fontSize;
                size.width = approxWidth;
                size.height = approxHeight;
            }
        });

	world.system<HorizontalLayoutBox>("ResetHProgress")
        .kind(flecs::PreFrame)
	    .each([](flecs::entity e, HorizontalLayoutBox& box)
	    {
	      box.x_progress = 0.0f;
	      e.children([&](flecs::entity child)
	      {
	        Position& pos = child.ensure<Position, Local>();
	        pos.x = box.x_progress;
	        const UIElementSize* size = child.try_get<UIElementSize>();
	        if (size) {
	          box.x_progress += size->width + box.padding;
	        }
	      });
	    });


    world.system<VerticalLayoutBox, UIElementSize>("ResetVProgress")
    .kind(flecs::PreFrame)
    .each([](flecs::entity e, VerticalLayoutBox& box, UIElementSize& container_size)
    {
        box.y_progress = 0.0f;
        float max_width = 0.0f; // Track the widest child

        e.children([&](flecs::entity child)
        {
            Position& pos = child.ensure<Position, Local>();
            pos.y = box.y_progress;
            
            // Renamed variable to avoid shadowing 'container_size'
            const UIElementSize* child_size = child.try_get<UIElementSize>();
            
            if (child_size) {
                // std::cout << box.y_progress << std::endl;
                box.y_progress += child_size->height + box.padding;
                
                // Update max width
                if (child_size->width > max_width) {
                    max_width = child_size->width;
                }
            }
        });

        container_size.height = box.y_progress;
        container_size.width = max_width; // Apply the calculated width
    });

    auto boundsCalculationSystem = world.system<Position, UIElementBounds, UIElementSize>()
        .term_at(0).second<World>()
        .kind(flecs::PostLoad) 
        .each([&](flecs::entity e, Position& worldPos, UIElementBounds& bounds, UIElementSize& size) {
            // Reset bounds to invalid state at start of each frame
            bounds.xmin = worldPos.x;
            bounds.ymin = worldPos.y;
            bounds.xmax = worldPos.x;
            bounds.ymax = worldPos.y;

            if (size.width > 0 && size.height > 0)
            {
                bounds.xmax += size.width;
                bounds.ymax += size.height;
            }
        });

    auto cursorEvents = world.observer<CursorState, EditorRoot>()
        .event<LeftClickEvent>()
        .term_at(1).src(editor_root)
        .each([&UIElement](flecs::iter& it, size_t i, CursorState& cursor_state, EditorRoot& editor_root) {
            std::cout << "Left click at " << cursor_state.x << ", " << cursor_state.y << std::endl;

            bool in_modify_region = false;
            for (EditorModifyPartitionRegion& partition_region : editor_root.modify_partition_regions)
            {
                if (point_in_bounds((float)cursor_state.x, (float)cursor_state.y, partition_region.bounds))
                {
                    in_modify_region = true; 
                    split_editor({0.05, PanelSplitType::Horizontal}, partition_region.split_target, world, UIElement);
                    partition_region.split_target.add<Dragging>().add<DynamicPartition>();
                }
            }
            if (!in_modify_region)
            {
                for (EditorShiftRegion& shift_region : editor_root.shift_regions)
                {
                    // std::cout << "Check shift region" << std::endl;
                    if (point_in_bounds((float)cursor_state.x, (float)cursor_state.y, shift_region.bounds))
                    {
                        shift_region.split_target.add<Dragging>();
                    }
                }
            }
        });

    world.system<Position, EditorNodeArea, PanelSplit, CursorState>()
        .term_at(0).second<World>()
        .term_at(3).src(glfwStateEntity)
        .with<Dragging>()
        .each([](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, PanelSplit& panel_split, CursorState& cursor_state)
        {
            // TODO: Propagate percent update to children to keep them 'in place'
            if (panel_split.dim == PanelSplitType::Horizontal)
            {
                panel_split.percent = std::clamp(((float)cursor_state.x-world_pos.x)/node_area.width, 0.05f, 0.95f);
            } 
            else if (panel_split.dim == PanelSplitType::Vertical)
            {
                panel_split.percent = std::clamp(((float)cursor_state.y-world_pos.y)/node_area.height, 0.05f, 0.95f);
            }
            // std::cout << "Evaluate drag update" << std::endl;
        });

    world.system<Position, EditorNodeArea, PanelSplit*, CursorState>()
        .term_at(0).second<World>()
        .term_at(2).optional()
        .term_at(3).src(glfwStateEntity)
        .with<Dragging>()
        .with<DynamicPartition>()
        .each([&UIElement](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, PanelSplit* panel_split, CursorState& cursor_state)
        {
            // First, we need to determine if the dynamic partition hover is hovered over the same node area it was spawned in
            // If not, then we should merge it with its sibling
            if (!point_in_bounds(cursor_state.x, cursor_state.y, {world_pos.x, world_pos.y, world_pos.x + node_area.width, world_pos.y + node_area.height}))
            {
                if (!e.has<DynamicMerge>())
                {
                    e.add<DynamicMerge>();
                    merge_editor(e, world, UIElement);
                    // TODO: Temporary merger to allow reversal?
                    merge_editor(e.parent(), world, UIElement);
                }
            } else
            {
                if (e.has<DynamicMerge>())
                {
                    e.remove<DynamicMerge>();
                    split_editor({0.05, PanelSplitType::Horizontal}, e, world, UIElement);
                } else
                {
                // TODO: Consider the case of child panel splits of the same dimension as multiple siblings
                // 1. Define the boundaries of the node
                float left   = world_pos.x;
                float right  = world_pos.x + node_area.width;
                float top    = world_pos.y;
                float bottom = world_pos.y + node_area.height;

                // 2. Calculate distance to vertical edges (Left/Right)
                float dist_left  = std::abs(cursor_state.x - left);
                float dist_right = std::abs(cursor_state.x - right);
                float min_dist_x = std::min(dist_left, dist_right);

                // 3. Calculate distance to horizontal edges (Top/Bottom)
                float dist_top    = std::abs(cursor_state.y - top);
                float dist_bottom = std::abs(cursor_state.y - bottom);
                float min_dist_y  = std::min(dist_top, dist_bottom);

                // 4. Compare: Is the cursor physically closer to a side edge, or a top/bottom edge?
                if (min_dist_x < min_dist_y) {
                    // Closer to Left or Right edge -> We want to split vertically ( | )
                    if (panel_split->dim == PanelSplitType::Horizontal)
                    {
                        flecs::entity prev_left = e.target<LeftNode>();
                        flecs::entity prev_right = e.target<RightNode>();
                        e.remove<LeftNode>(prev_left);
                        e.remove<RightNode>(prev_right);
                        e.add<UpperNode>(prev_left);
                        e.add<LowerNode>(prev_right);
                        panel_split->dim = PanelSplitType::Vertical;
                    }
                } else {
                    // Closer to Top or Bottom edge -> We want to split horizontally ( - )
                    if (panel_split->dim == PanelSplitType::Vertical)
                    {
                        flecs::entity prev_upper = e.target<UpperNode>();
                        flecs::entity prev_lower = e.target<LowerNode>();
                        e.remove<UpperNode>(prev_upper);
                        e.remove<LowerNode>(prev_lower);
                        e.add<LeftNode>(prev_upper);
                        e.add<RightNode>(prev_lower);
                        panel_split->dim = PanelSplitType::Horizontal;
                    }
                }
            }
            }
        });

    auto query_dragging = world.query_builder<PanelSplit>()
    .with<Dragging>()
    .build();

    world.observer<CursorState, EditorRoot>()
        .event<LeftReleaseEvent>()
        .term_at(1).src(editor_root)
        .each([&query_dragging](flecs::entity e, CursorState& cursor_state, EditorRoot& editor_root) {
            std::cout << "Left mouse release" << std::endl;
            query_dragging.each([](flecs::iter& it, size_t row, PanelSplit& panel_split) {
                flecs::entity e_drag = it.entity(row);
                e_drag.remove<Dragging>();
                e_drag.remove<DynamicMerge>(); // Remove dynamics too if they exist
                e_drag.remove<DynamicPartition>();
                std::cout << "Remove dragging" << std::endl;
            });
        });

    auto bubbleUpBoundsQuery = world.query_builder<UIElementBounds, UIElementBounds*, RenderStatus*>()
        .term_at(1).parent().up()  // Parent UIElementBounds
        .term_at(2).optional()          // Optional RenderStatus
        .build();

    auto bubbleUpBoundsSystem = world.system<UIElementBounds, UIElementBounds*, RenderStatus*>()
        .kind(flecs::PostLoad) 
        .term_at(1).parent().up()
        .term_at(2).optional()
        .each([&](flecs::entity e, UIElementBounds& bounds, UIElementBounds* parent_bounds, RenderStatus* render) {
            if (parent_bounds && (!render || render->visible)) {
                
                const Expand* expand = e.try_get<Expand>();

                if (bounds.xmax == 0)
                {
                    // Inherit bounds if non renderable
                    bounds.xmin = parent_bounds->xmin;
                    bounds.xmax = parent_bounds->xmax;
                    bounds.ymin = parent_bounds->ymin;
                    bounds.ymax = parent_bounds->ymax;
                }

                // Handle Horizontal Bubble Up
                // Only let the child set the parent's width if the child IS NOT expanding in X
                if (!expand || !expand->x_enabled) {
                    parent_bounds->xmin = std::min(parent_bounds->xmin, bounds.xmin);
                    parent_bounds->xmax = std::max(parent_bounds->xmax, bounds.xmax);
                }

                // Handle Vertical Bubble Up
                // Only let the child set the parent's height if the child IS NOT expanding in Y
                if (!expand || !expand->y_enabled) {
                    parent_bounds->ymin = std::min(parent_bounds->ymin, bounds.ymin);
                    parent_bounds->ymax = std::max(parent_bounds->ymax, bounds.ymax);
                }
                }
        });

    auto bubbleUpBoundsSecondarySystem = world.system<UIElementBounds, UIElementBounds*, RenderStatus*>()
        .kind(flecs::PostLoad) 
        .term_at(1).parent().up()
        .term_at(2).optional()
        .each([&](flecs::entity e, UIElementBounds& bounds, UIElementBounds* parent_bounds, RenderStatus* render) {
            if (parent_bounds && (!render || render->visible)) {
                
                const Expand* expand = e.try_get<Expand>();

                if (bounds.xmax == 0)
                {
                    // Inherit bounds if non renderable
                    bounds.xmin = parent_bounds->xmin;
                    bounds.xmax = parent_bounds->xmax;
                    bounds.ymin = parent_bounds->ymin;
                    bounds.ymax = parent_bounds->ymax;
                }

                // Handle Horizontal Bubble Up
                // Only let the child set the parent's width if the child IS NOT expanding in X
                if (!expand || !expand->x_enabled) {
                    parent_bounds->xmin = std::min(parent_bounds->xmin, bounds.xmin);
                    parent_bounds->xmax = std::max(parent_bounds->xmax, bounds.xmax);
                }

                // Handle Vertical Bubble Up
                // Only let the child set the parent's height if the child IS NOT expanding in Y
                if (!expand || !expand->y_enabled) {
                    parent_bounds->ymin = std::min(parent_bounds->ymin, bounds.ymin);
                    parent_bounds->ymax = std::max(parent_bounds->ymax, bounds.ymax);
                }
                }
        });

    world.system<UIElementBounds*, RectRenderable, Expand>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, RectRenderable& rect, Expand& expand) {
        // std::cout << bounds->xmin << std::endl;
        // std::cout << bounds->xmax << std::endl;
        
        if (expand.x_enabled)
        {
            rect.width = (bounds->xmax - bounds->xmin - (expand.pad_left + expand.pad_right))*expand.x_percent;
        }
        if (expand.y_enabled)
        {
            rect.height = (bounds->ymax - bounds->ymin - (expand.pad_top + expand.pad_bottom))*expand.y_percent;
        }
    });

    world.system<UIElementBounds*, RoundedRectRenderable, Expand>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, RoundedRectRenderable& rect, Expand& expand) {
        if (expand.x_enabled)
        {
            rect.width = (bounds->xmax - bounds->xmin - (expand.pad_left + expand.pad_right))*expand.x_percent;
        }
        if (expand.y_enabled)
        {
            rect.height = (bounds->ymax - bounds->ymin - (expand.pad_top + expand.pad_bottom))*expand.y_percent;
        }
    });

    world.system<UIElementBounds*, ImageRenderable, Expand, Graphics>()
    .term_at(0).parent()
    .kind(flecs::PreUpdate)
    .each([&](flecs::entity e, UIElementBounds* bounds, ImageRenderable& sprite, Expand& expand, Graphics& graphics) {        
        if (bounds)
        {
            int img_width, img_height;
            nvgImageSize(graphics.vg, sprite.imageHandle, &img_width, &img_height);
            if (expand.x_enabled)
            {
                sprite.width = (bounds->xmax - bounds->xmin - (expand.pad_left + expand.pad_right))*expand.x_percent;
                if (!expand.y_enabled)
                {
                    // TODO: Keep the ratio the same!
                    sprite.height = sprite.width/img_width * img_height;
                }
            }
            if (expand.y_enabled)
            {
                sprite.height = (bounds->ymax - bounds->ymin - (expand.pad_top + expand.pad_bottom))*expand.y_percent;
                if (!expand.x_enabled)
                {
                    // TODO: Keep the ratio the same!
                    sprite.width = sprite.height/img_height * img_width;
                }
            }
        }
    });

    auto debugRenderBounds = world.system<RenderQueue, UIElementBounds>()
    .term_at(0).src(renderQueueEntity)
    .each([](flecs::entity e, RenderQueue& render_queue, UIElementBounds& bounds) 
    {
        RectRenderable debug_bound {bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin, true, 0xFFFF00FF};
        //render_queue.addRectCommand({bounds.xmin, bounds.ymin}, debug_bound, 100);
    });

    auto roundedRectQueueSystem = world.system<Position, RoundedRectRenderable, ZIndex>()
    .term_at(0).second<World>()
    .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, Position& pos, RoundedRectRenderable& renderable, ZIndex& zIndex) {
            RenderQueue& queue = world.ensure<RenderQueue>();
            queue.addRoundedRectCommand(pos, renderable, zIndex.layer);
        });


    auto rectQueueSystem = world.system<Position, RectRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
        .each([&](flecs::entity e, Position& pos, RectRenderable& renderable, ZIndex& zIndex) {
            RenderQueue& queue = world.ensure<RenderQueue>();
            queue.addRectCommand(pos, renderable, zIndex.layer);
        });

    auto textQueueSystem = world.system<Position, TextRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, TextRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world.ensure<RenderQueue>();
        queue.addTextCommand(pos, renderable, zIndex.layer);
    });

    auto imageQueueSystem = world.system<Position, ImageRenderable, ZIndex>()
    .kind(flecs::PostUpdate)
    .term_at(0).second<World>()
    .each([&](flecs::entity e, Position& pos, ImageRenderable& renderable, ZIndex& zIndex) {
        RenderQueue& queue = world.ensure<RenderQueue>();
        queue.addImageCommand(pos, renderable, zIndex.layer);
    });


    world.system<Position, EditorNodeArea, EditorLeafData, EditorRoot>()
    .term_at(0).second<World>()
    .term_at(3).src(editor_root)
    .run([](flecs::iter& it)
    {
        auto editor_root = world.lookup("editor_root").try_get_mut<EditorRoot>();
        editor_root->modify_partition_regions.clear();
        while (it.next()) {
            it.each();
        }
    }, [](flecs::entity e, Position& world_pos, EditorNodeArea& node_area, EditorLeafData& leaf_data, EditorRoot& editor_root) 
    {
        editor_root.modify_partition_regions.push_back({{world_pos.x, world_pos.y, world_pos.x + 8.0f, world_pos.y + 24.0f}, e});
    });

    int scale_region_dist = 8;
    
    world.system<Window, CursorState, EditorNodeArea, PanelSplit, Position, EditorRoot>()
    .term_at(0).src(glfwStateEntity)
    .term_at(1).src(glfwStateEntity)
    .term_at(4).second<World>()
    .term_at(5).src(editor_root)
    .run([scale_region_dist](flecs::iter& it)
    {
        auto editor_root = world.lookup("editor_root").try_get_mut<EditorRoot>();
        editor_root->shift_regions.clear();
        while (it.next()) {
            it.each();
        }
        
        auto window = it.field<Window>(0);
        auto cursor_state = it.field<CursorState>(1);
        

        // DEBUG BOUNDS
        // for (EditorShiftRegion& shift_region : editor_root->shift_regions)
        // {
        //     RenderQueue& queue = world.ensure<RenderQueue>();
        //     RectRenderable debug_rect;
        //     debug_rect.width = shift_region.bounds.xmax - shift_region.bounds.xmin; 
        //     debug_rect.height = shift_region.bounds.ymax - shift_region.bounds.ymin;
        //     debug_rect.color = 0xFF00FFFF;
        //     debug_rect.stroke = true;
        //     queue.addRectCommand({shift_region.bounds.xmin, shift_region.bounds.ymin}, debug_rect, 100);
        // }

        // std::cout << cursor_state->x << ", " << cursor_state->y << std::endl;

        glfwSetCursor(window->handle, NULL);

        bool in_modify_region = false;
        for (EditorModifyPartitionRegion& partition_region : editor_root->modify_partition_regions)
        {
            if (point_in_bounds((float)cursor_state->x, (float)cursor_state->y, partition_region.bounds))
            {
                GLFWcursor* cursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
                glfwSetCursor(window->handle, cursor);
                in_modify_region = true;
            }
        }
        if (!in_modify_region)
        {
            for (EditorShiftRegion& shift_region : editor_root->shift_regions)
            {
                if (point_in_bounds((float)cursor_state->x, (float)cursor_state->y, shift_region.bounds))
                {
                    GLFWcursor* cursor = glfwCreateStandardCursor(shift_region.cursor_type);
                    glfwSetCursor(window->handle, cursor);
                }
            }
        }
        
        // TODO: If the cursor location is within one of those rectangles, then create a tag indicating the scale target node
        // and change the curorsor type
        
        // GLFWcursor* cursor = glfwCreateStandardCursor(GLFW_RESIZE_EW_CURSOR);
        // GLFWcursor* cursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
        // glfwSetCursor(window, cursor);
    },
    [scale_region_dist](flecs::entity e, Window& window, CursorState& cursor_state, EditorNodeArea& node_area, PanelSplit& panel_split, Position& world_pos, EditorRoot& editor_root) {
        UIElementBounds bounds;
        if (panel_split.dim == PanelSplitType::Horizontal)
        {
            float line_xpos = world_pos.x + panel_split.percent * node_area.width;
            bounds.xmin = line_xpos - scale_region_dist;
            bounds.ymin = world_pos.y;
            bounds.xmax = line_xpos + scale_region_dist;
            bounds.ymax = world_pos.y + node_area.height;
            editor_root.shift_regions.push_back({bounds, GLFW_RESIZE_EW_CURSOR, e});
        } else if (panel_split.dim == PanelSplitType::Vertical)
        {
            float line_ypos = world_pos.y + panel_split.percent * node_area.height;
            bounds.xmin = world_pos.x;
            bounds.ymin = line_ypos - scale_region_dist;
            bounds.xmax = world_pos.x + node_area.width;
            bounds.ymax = line_ypos + scale_region_dist;
            editor_root.shift_regions.push_back({bounds, GLFW_RESIZE_NS_CURSOR, e});
        }
    });

    auto renderExecutionSystem = world.system<RenderQueue, Graphics>()
        .kind(flecs::PostUpdate)
        .each([&](flecs::entity e, RenderQueue& queue, Graphics& graphics) {
            queue.sort();

            for (const auto& cmd : queue.commands) {
                switch (cmd.type) {
                    case RenderType::RoundedRectangle: {
                        const auto& rect = std::get<RoundedRectRenderable>(cmd.renderData);
                        nvgBeginPath(graphics.vg);
                        nvgRoundedRect(graphics.vg, cmd.pos.x, cmd.pos.y, rect.width, rect.height, rect.radius);

                        uint8_t r = (rect.color >> 24) & 0xFF;
                        uint8_t g = (rect.color >> 16) & 0xFF;
                        uint8_t b = (rect.color >> 8) & 0xFF;

                        if (rect.stroke)
                        {
                            nvgStrokeColor(graphics.vg, nvgRGB(r, g, b));
                            nvgStroke(graphics.vg);
                        } else
                        {
                            nvgFillColor(graphics.vg, nvgRGB(r, g, b));
                            nvgFill(graphics.vg);
                        }
                        break;
                    }
                    case RenderType::Rectangle: {
                        const auto& rect = std::get<RectRenderable>(cmd.renderData);
                        nvgBeginPath(graphics.vg);
                        nvgRect(graphics.vg, cmd.pos.x, cmd.pos.y, rect.width, rect.height);

                        uint8_t r = (rect.color >> 24) & 0xFF;
                        uint8_t g = (rect.color >> 16) & 0xFF;
                        uint8_t b = (rect.color >> 8) & 0xFF;

                        if (rect.stroke)
                        {
                            nvgStrokeColor(graphics.vg, nvgRGB(r, g, b));
                            nvgStroke(graphics.vg);
                        } else
                        {
                            nvgFillColor(graphics.vg, nvgRGB(r, g, b));
                            nvgFill(graphics.vg);
                        }
                        break;
                    }
                    case RenderType::Text: {
                        const auto& text = std::get<TextRenderable>(cmd.renderData);
                        nvgFontSize(graphics.vg, text.fontSize);
                        nvgFontFace(graphics.vg, text.fontFace.c_str());
                        nvgTextAlign(graphics.vg, text.alignment);

                        uint8_t r = (text.color >> 24) & 0xFF;
                        uint8_t g = (text.color >> 16) & 0xFF;
                        uint8_t b = (text.color >> 8) & 0xFF;

                        nvgFillColor(graphics.vg, nvgRGB(r, g, b));
                        nvgText(graphics.vg, cmd.pos.x, cmd.pos.y, text.text.c_str(), nullptr);
                        break;
                    }
                    case RenderType::Image: {
                        const auto& image = std::get<ImageRenderable>(cmd.renderData);
                        if (image.imageHandle != -1) {
                            NVGpaint imgPaint = nvgImagePattern(graphics.vg, cmd.pos.x, cmd.pos.y,
                                                              image.width, image.height, 0.0f,
                                                              image.imageHandle, 1.0); // image.alpha
                            nvgBeginPath(graphics.vg);
                            nvgRect(graphics.vg, cmd.pos.x, cmd.pos.y, image.width, image.height);
                            nvgFillPaint(graphics.vg, imgPaint);
                            nvgFill(graphics.vg);
                        }
                        break;
                    }
                }
            }

            queue.clear();
        });

    int fontHandle = nvgCreateFont(vg, "ATARISTOCRAT", "../assets/ATARISTOCRAT.ttf");

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        int winWidth, winHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        float devicePixelRatio = (float)fbWidth / (float)winWidth;

        glViewport(0, 0, fbWidth, fbHeight);
        // glClearColor(22.0f/255.0f, 22.0f/255.0f, 22.0f/255.0f, 0.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glfwStateEntity.set<Window>({window, winWidth, winHeight});

        nvgBeginFrame(vg, winWidth, winHeight, devicePixelRatio);
        
        world.defer_begin();
        glfwPollEvents();
        world.defer_end();
        world.progress();

        nvgEndFrame(vg);

        glfwSwapBuffers(window);
    }

    nvgDeleteGL2(vg);
    glfwTerminate();
    return 0;
}
