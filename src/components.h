#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>

#include <nanovg.h>
#include <flecs.h>
#include <yoga/Yoga.h>

#include <raymath.h>

// ECS Components
struct TimeInterval
{
    float startTime;
    float endTime;
    float get_duration() {
        return endTime - startTime;
    }
};

// There must be some system which takes in a TimeInterval and TimeIntervalToSpaceSegment and modifies the RectRenderable width
struct TimeIntervalToSpaceSegment{};

struct TextSize { float w, h; };

typedef Vector2 Position;
struct Edge
{
    Position p0;// LibVNC
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

struct DebugRenderBounds {};
struct UIElementBounds {
    float xmin, ymin, xmax, ymax;
};

// What the Peach Core panel knows about a supporting process.
//
// The first three are what Thornfield can observe from outside: no process, a
// process we just spawned, a process that exists. Ready is different in kind --
// only the server knows when its model is loaded and it is answering, so it has
// to say so. It reports over the announce socket in server_registry.h, and
// until it does it sits at Running.
enum class ServerStatus
{
    Offline,
    Starting,
    Running,
    Ready,
};

// One entry from assets/config/servers.json, held by a registry entity that
// outlives any panel. See that file for what each field means.
struct ServerDef
{
    std::string id;
    std::string label;
    std::string icon;
    std::string description;

    std::string script;
    std::string cwd;
    std::vector<std::string> args;

    std::string conda_env;              // "" = whatever python3 resolves to
    std::string launch = "direct";      // "direct" | "conda-run"
    std::string socket;
    std::string match;                  // /proc/<pid>/cmdline needle

    bool autostart = false;
};

struct ServerState
{
    ServerStatus status = ServerStatus::Offline;

    // The process id, and whether Thornfield is the one that started it. A
    // foreign process can be reported but not stopped -- we do not own it, and
    // killing something the user launched by hand is not ours to do.
    int pid = 0;
    bool owned = false;

    double started_at = 0.0;            // world time, for the warmup grace

    // Set by the server's own announcement, cleared when the process goes away.
    // last_report is world time; once servers announce periodically rather than
    // once, it is what lets a wedged server decay back out of Ready.
    bool ready = false;
    double last_report = 0.0;
};

// Icon entity --ServerOf--> registry entity. Panels come and go; the registry
// entity and its process do not, so the icon only ever holds this reference.
struct ServerOf {};

struct RenderStatus {
    bool visible;
    // float transparency = 1.0f;
};

// A generic config component for panel to provide them contextual state
// TODO: Each option string should have an associated description that can allow for semantic search and
// easy natural language editable options...
struct PanelOption
{
    std::string name;
    bool active;
    std::string description;
};

struct PanelOptionData
{
    size_t index;
};

// The options should show up as checkboxes
struct PanelState 
{
    std::vector<PanelOption> options; 
    std::unordered_map<std::string, uint32_t> option_indices;

    void add_option(PanelOption& option)
    {
        if (!option_indices.count(option.name))
        {
            option_indices[option.name] = options.size();
            options.push_back(option); 
        }
    }

    bool is_option_active(std::string option_name)
    {
        return option_indices.count(option_name) && options[option_indices[option_name]].active;
    }
};

// ============================================================================
// Yoga (flexbox) layout
// ----------------------------------------------------------------------------
// Marker tag: any entity with this participates in the Yoga layout tree.
// An OnAdd observer creates its YogaNode; an OnRemove observer frees it.
// Entities without UIYoga are invisible to the Yoga tree even when their
// flecs parent is a Yoga node, so Yoga can be adopted subtree by subtree.
struct UIYoga {};

struct YogaNode {
    YGNodeRef node = nullptr;
};

struct UISize { float w = YGUndefined, h = YGUndefined; };
struct UIMaxSize { float w = YGUndefined, h = YGUndefined; };
struct UIMinSize { float w = YGUndefined, h = YGUndefined; };

// Absolute positioning with per-edge offsets (any edge = NaN means "unset").
// Used for "fill parent" and "fill parent with padding" cases -- the Yoga
// equivalent of the old Expand component.
struct UIAbsoluteEdges {
    float left = YGUndefined;
    float top = YGUndefined;
    float right = YGUndefined;
    float bottom = YGUndefined;
};

// Aspect ratio (width / height). When set, Yoga uses it to compute the
// opposite axis from the known one.
struct UIAspectRatio { float ratio = YGUndefined; };

// Flex Container Properties (Parent-level)
struct UIFlexContainer {
    YGFlexDirection direction = YGFlexDirectionColumn;
    YGWrap wrap = YGWrapNoWrap;
    YGJustify justify = YGJustifyFlexStart;
    YGAlign items = YGAlignFlexStart;
    float gap = 0.0f;
};

// Flex Item Properties (Child-level)
struct UIFlexItem {
    float grow = 0.0f;
    float shrink = 1.0f;
    YGAlign alignSelf = YGAlignAuto;
};

// Box Model
struct UIMargin  { float top, right, bottom, left; };
struct UIPadding { float top, right, bottom, left; };

// Adopts a subtree that is still laid out by the legacy pipeline
// (LayoutBox / UIContainer / Expand) as a single leaf in a Yoga row or column.
// Yoga takes the node's size from whatever size the old layout computed for the
// entity and only decides *where* it goes; nothing inside it is touched, and
// YogaReadback will not write its renderable size back. This is how calibrated
// widgets -- badges, triple blocks -- get flex spacing without being rebuilt.
struct UIYogaLegacyLeaf {};

// Opt-in intrinsic sizing for a Yoga image leaf: style width/height are pinned
// to the image's native pixel size (times ImageRenderable scale). Without this
// an image leaf carries only an aspect ratio, so it needs a definite size from
// somewhere -- a flex stretch, a parent, or this tag.
struct UINativeImageSize {};

// Caps a Yoga image leaf at its native pixel size (times ImageRenderable
// scale), so flex stretching can shrink it but never upscale past 1:1.
// Overrides UIMaxSize when both are present.
struct UIMaxNativeImageSize {};

// Sizes a Yoga image leaf to the largest size that fits inside its parent's
// content box while keeping the image's native aspect ratio -- flexbox
// "contain". Pair it with a parent that centres (JustifyCenter / AlignCenter)
// to place the result.
//
// This can't be expressed with aspectRatio + UIMaxSize: when a max constraint
// clamps one axis, Yoga leaves the other at its styled value instead of
// re-deriving it from the ratio, and the image comes out skewed. Computing both
// axes here keeps the proportions exact and, because it happens in the same
// PreFrame pass as the rest of layout, without the frame of lag the old
// Expand/Constrain pair had.
struct UIContainParent {
    float pad_left = 0.0f, pad_top = 0.0f, pad_right = 0.0f, pad_bottom = 0.0f;
    bool allow_upscale = false;
};

// Bridges a legacy-laid-out parent into a Yoga root: each frame the parent's
// UIElementSize is copied into this entity's UISize, minus the given insets.
// This is how a Yoga subtree fills an editor panel whose own sizing is still
// owned by the old Expand/LayoutBox pipeline, so panels can be ported to Yoga
// one at a time. Position<Local> of the root should match pad_left/pad_top.
struct UIFillParent {
    float pad_left = 0.0f, pad_top = 0.0f, pad_right = 0.0f, pad_bottom = 0.0f;
};

// Remembers what a Yoga text leaf was last measured with. Yoga caches measure
// results, so the node is only marked dirty when the text actually changes --
// otherwise every text node would force a full relayout every frame.
struct YogaTextCache {
    std::string text;
    std::string fontFace;
    float fontSize = 0.0f;
    float wrapWidth = 0.0f;
    float scaleY = 0.0f;
};

// ============================================================================

// Unified layout box - used for both horizontal and vertical layouts
struct LayoutBox
{
  enum Direction { Horizontal, Vertical };
  Direction dir = Horizontal;
  float padding = 0.0f;
  float move_dir = 1.0f; // 1 = right/down, -1 = left/up
};

struct ColumnFitConstraint
{
    int count;
};

struct FitChildren 
{
    float scale_factor;
};

struct FlowLayoutBox
{
  float x_progress;
  float y_progress;
  float padding = 0.0f;
  float line_height = 0.0f;
  float line_spacing = 0.0f;
};

struct RenderGradient
{
    uint32_t start;
    uint32_t end;
};

struct RectRenderable {
    float width, height;
    bool stroke;
    uint32_t color;
};

// Forward declaration
struct RenderCommand; 

// Example, diamond
struct CustomRenderable
{
    float width, height;
    bool stroke;
    uint32_t color;
    uint32_t gradient_start = 0;
    uint32_t gradient_end = 0;

    std::function<void(NVGcontext*, const RenderCommand*, const CustomRenderable&)> render_function;
};

struct RoundedRectRenderable {
    float width, height, radius;
    bool stroke;
    uint32_t color;
};

struct LineRenderable {
    float x1, y1;
    float x2, y2;
    
    float thickness;
    uint32_t color;
};

struct QuadraticBezierRenderable {
    float x1, y1;
    float cx, cy;
    float x2, y2;
    
    float thickness;
    uint32_t color;
};

struct DiurnalHour
{
    size_t segment;
};

struct DynamicTextWrap 
{
    float pad;
};

struct DynamicTextWrapContainer {};

struct TextRenderable {
    std::string text;
    std::string fontFace;
    float fontSize;
    uint32_t color;
    float scaleY = 1.0f;
    float wrapWidth = 0.0f;  // 0 = no wrapping, >0 = wrap at this width
    // True when wrapWidth was derived from Yoga's offered layout bound rather
    // than authored (badge_apply_wrap, input fields). Authored widths are
    // never touched; a derived one is re-derived when the bound moves, so a
    // widening panel un-wraps text that no longer needs it.
    bool autoWrapped = false;
};

struct ImageCreator
{
    std::string path;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    NVGcolor tint = nvgRGBA(255, 255, 255, 255);
};

struct ImageRenderable
{
    int imageHandle;
    float scaleX, scaleY;

    float width, height;
    NVGcolor tint = nvgRGBA(255, 255, 255, 255);

    // Texture offset for smooth scrolling (shifts UV sampling, not element position)
    float texOffsetX = 0.0f;
    float texOffsetY = 0.0f;
};

// Scroll position of a chat transcript, in pixels from the bottom. Zero is the
// newest message resting on the panel's lower edge, which is where a chat wants
// to sit; scrolling up reveals history. Lives on the message list.
struct ChatScroll { float offset = 0.0f; };

// Vertical scroll for a top-anchored list: the same shape as ChatScroll, minus
// the transcript's bottom anchoring. Put it on a content container that carries
// UIAbsoluteEdges and lives inside a clipped parent -- the container overflows
// its parent by design, PanelScrollSystem slides it and clamps to the overflow,
// and the parent's scissor hides what runs past. Any panel with more rows than
// room gets scrolling by adding these two components.
struct PanelScroll { float offset = 0.0f; };

// The slider of a top-anchored list's scrollbar, pointing at the PanelScroll it
// reports -- the counterpart to ChatScrollbarOf, so the system that sizes it
// needs no fixed hierarchy between the two.
struct PanelScrollbarOf {};

// The slider of a transcript's scrollbar. Points at the ChatScroll it reports,
// so the system that sizes it needs no fixed hierarchy between the two.
struct ChatScrollbarOf {};

// A playable clip, drawn as its styled spectrogram. The image is the strip's
// own ImageRenderable; this carries what the strip needs to behave as a player
// -- which file to play, and how long it is, so a playhead can be positioned
// without asking the audio device about a clip it may not have loaded.
struct SpecStrip
{
    std::string wav_path;
    std::string palette;      // empty = spectrogram::default_palette()
};

// Marks the ImageRenderable whose texture is the offscreen render3d scene (see
// panel3d.h). The panel's laid-out bounds drive the viewport's resolution and
// its hit test, so the tag is how that per-frame sync finds the element.
struct Panel3DViewport {};

// The Holodeck panel's 3D viewport element -- the screen3d module's offscreen
// image, kept sized/fed by Screen3DSyncSystem the way Panel3DViewport is.
struct Screen3DViewport {};

// The Droid panel's 3D tooltip: a screen-sized badge on the canvas that rides
// the cursor (bottom-left corner on the pointer) while a scene part is
// hovered (panel3d::has_hover). Parked far off-canvas otherwise.
struct DroidPartTooltip {};

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

struct CallbackOnLeftClick {
    std::function<void(flecs::entity)> onClicked;
};

// Tracks the pointer event currently being dispatched. Click targets under the
// cursor are visited in priority order -- highest ZIndex first, deepest in the
// hierarchy first on ties -- and a handler that acts on the click marks it
// consumed, which stops the event reaching anything beneath. Without this,
// every overlapping element receives the same click: clicking an option row in
// a popup also "clicks" the dropdown icon whose bubbled-up bounds contain the
// popup, which immediately toggles the menu shut.
struct UIInputDispatch {
    bool consumed = false;
};

struct AddTagOnLeftClick{};
struct ShowEditorPanels {};
struct ShowPanelState {};
struct SetPanelEditorType {};
struct ToggleBooleanOption {};
struct SelectServer {};

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

struct ServerHUDOverlay {};
struct ServerDescription
{
    ecs_entity_t selected;
};
struct ShowServerHUDOverlay {};
struct HideServerHUDOverlay {};

struct HighlightBFOInheritanceHierarchy {};
struct ResetBFOSprites {};

struct CloseEditorSelector {};
struct SetMenuHighlightColor {};
struct SetMenuStandardColor {};

// Crop entities beyond panel edge
struct ScissorContainer {};

struct CheckBox {};
struct ChatMessage {
    std::string author;
    std::string text;
    // Index into g_annotatable_messages when this message has an annotation
    // row, -1 otherwise (command lines). What lets the saved transcript carry
    // each message's annotation template alongside its raw text.
    int annotatable = -1;
};

struct ChatState {
    std::vector<ChatMessage> messages;
    std::string draft;
    bool input_focused;
    // Caret position in the draft, in bytes (input is ASCII-filtered).
    // Everything that mutates the draft clamps against its current size, so a
    // stale value after clear/replace degrades to "caret at the end".
    int cursor = 0;
    // Selection anchor: the fixed end of a drag or shift-extend, with the
    // caret as the moving end. -1 (or equal to the caret, or out of range)
    // means no selection -- so stale values self-degrade to none.
    int sel_anchor = -1;
};

struct ChatMessageView {
    int index;
};

// Discord-style row hover: a transcript row carrying this tints faintly while
// the mouse is over it. The row keeps a permanent zero-alpha fill and the
// hover system only writes the colour, so hovering never churns archetypes.
// `partner` couples a message bubble to its interpretation stack (and back):
// hovering either lights both, because they are one utterance shown twice --
// once as said, once as understood.
struct ChatHoverRow {
    flecs::entity partner;
};

// Marks a rendered text as sweep-selectable, Discord-style: mouse drag over
// it selects a substring (hit-tested against the renderer's own wrapping) and
// Ctrl+C copies it. One selection exists at a time, app-wide.
struct SelectableText {};

// One hover action chip (edit / delete) on a transcript message. Chips exist
// permanently at zero alpha and light only while their host row is hovered --
// the hover system writes colours, the click handler guards on actual hover
// so an invisible chip can never be clicked.
struct ChatMessageAction {
    int index = -1;           // into ChatState.messages
    bool erase = false;       // delete rather than edit
    flecs::entity host;       // the hover row this chip belongs to
    flecs::entity label;      // the chip's text, for visibility toggling
};

struct FocusChatInput {};
struct SendChatMessage {};

// The badge under the Interlocutor's annotation cursor -- what the selected
// stop literally displays -- or empty when the cursor is on plain text or
// annotation mode is off. Defined in main.cpp beside the annotator state it
// reads; declared here so panel modules can key off the selection.
std::string annotator_selected_badge();

// The binding symbol and colour a named entity's badge wears on-screen -- its
// unique shorthand id, recovered from the annotated messages that bound it.
// Returns false for names with no bound identity (pure types, unannotated
// words); such badges keep whatever colours their panel gives them.
bool entity_screen_style(const std::string& name, std::string* symbol, uint32_t* color);

// The relationship under the Interlocutor's annotation cursor as
// "source\trelation\ttarget", or empty when the cursor is elsewhere. Defined
// in main.cpp beside the annotator state it reads.
std::string annotator_selected_triple();

// The comprehension the annotation cursor is on -- the [[Q ]] frame the Q
// gesture wraps a span in -- as its member words, tab separated. Empty when
// the cursor is not inside a frame. The frame is an intension: it carries its
// own id and a second id for the extension it denotes, so this is how a panel
// reads "the query this sentence just declared". Only the surface words are
// published, because what a term MEANS is the executing context's business --
// an ARC tag here, a world type elsewhere. Defined in main.cpp.
std::string annotator_selected_comprehension();

// Which role of an annotated relationship the MOUSE is over: 0 source, 1 the
// relation, 2 target, -1 none. Joined badges count as the slot they fuse
// into. Defined in main.cpp.
int annotator_hovered_slot();

// Same encoding for the annotation CURSOR's position: the role of the
// selected stop, with an entity badge bound into a relationship's end
// counting as that end. Defined in main.cpp.
int annotator_selected_slot();

// The inner row of a badge, where its label and symbols live. Published so a
// caller can nest another badge *inside* this one -- a relationship block that
// holds its target, Scratch-style -- rather than only placing badges beside
// each other. The badge entity itself is the block; this is its slot.
struct BadgeContent {
    flecs::entity row;
    // The label inside that row. Carried rather than searched for: code that
    // re-cuts a badge needs to measure the label, and scanning the row for "the
    // first child with text" silently finds nothing the moment the internals
    // change shape.
    flecs::entity label;
};

// The Lexicon panel: shows the word currently selected in the annotator as a
// strip of character glyphs -- the subword representation the neurosymbolic
// rule layers will operate over. `shown` is what the strip currently renders,
// so the panel only rebuilds when the selection actually moves to a new word.
// One row of a paradigm in the Lexicon panel: a word form's current
// characters and the entity type it is annotated as. The type is free text
// naming any world type -- Singular, Plural, PastTense, whatever the user
// types -- created on demand when the paradigm commits, never an enum.
struct LexiconForm {
    std::string text;
    // One or more type names separated by '|' -- "Plural|Singular" is a
    // disjunction, rendered as one badge per disjunct with the OR glyph
    // between. The commit path splits it back into one tag per type on a
    // single world entity.
    std::string type;
    // Set on rows the machine proposed; X on such a row is a rejection --
    // a negative demonstration -- rather than a character edit.
    bool proposed = false;
};

struct LexiconPanel {
    flecs::entity glyph_row;   // the column the paradigm rows build into
    std::string shown;

    // The paradigm: row 0 is the word from the sentence (surface form, not
    // editable); further rows are variants, spawned by Down past the end and
    // shaped by letter edits. Direction of any taught pair comes from the
    // Number annotations, never from row position.
    std::vector<LexiconForm> forms;

    // Character-level selection: which row, and which characters within it.
    bool selecting = false;
    int row = 0;
    int sel_start = 0;
    int sel_end = 0;

    // Tab switches the selected row between its two fields: characters, or
    // the type annotation being typed.
    bool typing_type = false;

    // Whether the FST's proposal has pre-filled the paradigm for this word;
    // reset when the word changes so a stale proposal can't dress up as new.
    bool prefilled = false;

    bool dirty = false;
};

// The Transducer panel: the neurosymbolic rule series the morphology server
// has synthesized from demonstrations -- each rule's rewrite, its evidence,
// and its confidence after exceptions have testified against it. `shown_sig`
// is a fingerprint of what is rendered, so the list rebuilds only when the
// machine actually relearns.
struct TransducerPanel {
    flecs::entity list;
    std::string shown_sig;
};

// Marks a sentence's POS tier strip, so a late tag reply can find and replace
// the tier without tearing down the sentence it floats above.
struct PosTier {};

// A co-reference arc in the annotated sentence: the same symbol appearing in
// two places -- an entity badge and a relationship slot bound to it -- drawn as
// a curve between them, the way dependency parses draw them. The arc stores
// which two UI entities it connects; a system re-derives the curve geometry
// from their bounds every frame, so the arc follows layout instead of being
// baked to positions that a wrap or resize would orphan.
struct CorefArc {
    flecs::entity from;   // the anchor: entity badge or reified block
    flecs::entity to;     // the referring slot glyph
};

// Backs one row of the Entities panel. The badge entity is a presentation
// element built by create_badge(); this carries what it stands for, so a click
// handler can recover the subject without reading it back out of a TextRenderable.
struct EntityBadge {
    std::string name;
    size_t index = 0;
    flecs::entity panel;   // leaf carrying the EntitySelection this row reports to
};

// Which row of an Entities panel is selected, and what the inspector subpanel
// currently shows. Two fields rather than an event: the rebuild is driven by
// comparing them, so a selection made while the panel is being rebuilt can't be
// missed, and setting the same index twice costs nothing.
//
// `selected` and `ranked_for` are indices into the *backing data*, never row
// positions, so they survive the list being reordered underneath them.
struct EntitySelection {
    flecs::entity inspector;   // subpanel the selected entity is built into
    flecs::entity list;        // list content, marked dirty so rows can restyle
    size_t selected = SIZE_MAX;
    size_t shown = SIZE_MAX;

    // Which selection and relation the current ordering was computed for, and
    // whether a request is in flight. Together they stop the panel re-asking
    // the embedding server for a ranking it already has.
    size_t ranked_for = SIZE_MAX;
    std::string ranked_query;
    bool rank_pending = false;

    // Semantic relation currently in force, submitted from the panel's semantic
    // bar. Held here rather than on the bar because the bar is a view -- this is
    // the panel's state, and it has to outlive a rebuild of the panel chrome.
    std::string relation;
};

// The entities an Entities panel is showing, pulled from a separate headless
// flecs world over the entity_query bridge.
//
// The panel owns its result set rather than reading a global, so two panels can
// hold two different queries against the same world. `names` and `detail` are
// index-aligned; a row is an index into both.
struct EntitySet {
    // Three stages of the same string: what the panel should be showing, what
    // is in flight, and what actually produced the current rows. A request goes
    // out whenever `wanted` has moved ahead of `expr` and nothing is pending.
    std::string wanted;
    std::string requested;
    std::string expr;

    std::vector<std::string> names;

    // Index-aligned with names, one list per entity. Kept apart because the
    // inspector draws each kind differently: tags and component types as plain
    // badges, relations as an arrow badge followed by its target.
    std::vector<std::vector<std::string>> tags;
    std::vector<std::vector<std::string>> components;

    // Each entry is "Relation\tTarget" -- tab, because flecs paths contain
    // colons and a relation target is often a path.
    std::vector<std::vector<std::string>> relations;

    // Last reply from the bridge: "OK", "OFFLINE" (world not running),
    // "ERROR" (bad query), or empty before the first request.
    std::string status;
    std::string error;

    bool loaded = false;
};

// Natural-language relation typed into an Entities panel, applied to whatever
// entity is selected: select "Giraffe", type "lives nearby", and the list
// reorders by that relation rather than by bare name similarity.
//
// `draft` is what is being typed; `applied` is what was last submitted. Keeping
// them apart means the list only re-ranks on Enter, not on every keystroke --
// each rank is a round trip to the embedding server.
// Lives on the input bar entity itself, not the panel leaf, because a panel has
// two of them -- a flecs query bar and a semantic one -- and flecs allows only
// one instance of a component per entity.
struct EntityQuery {
    flecs::entity panel;         // leaf this bar belongs to
    flecs::entity input_text;    // TextRenderable kept in sync with the draft
    std::string draft;
    std::string applied;
    bool focused = false;

    // What submitting the bar does. All kinds share the keyboard plumbing;
    // only the effect of Enter differs.
    enum Kind {
        Semantic,   // re-rank the loaded rows by proximity / relation
        Flecs,      // re-query the headless world for a new set of rows
        Supertype,  // assert <lattice focus> SubtypeOf <typed name> in the world
        Subtype,    // assert <typed name> SubtypeOf <lattice focus> in the world
        Instance,   // ensure <typed name> as an individual tagged <lattice focus>
    } kind = Semantic;

    std::string placeholder;
};

struct FocusEntityQuery {};

// A supertype or subtype name submitted from a Lattice panel's entry shell,
// parked on the panel until its module sends it to the world. Components
// rather than direct calls because the Enter key is handled in main.cpp's
// shared input plumbing, which should not know how the lattice talks to the
// bridge. Two components, not a flag: both entries can be mid-flight at once.
struct SupertypeSubmit {
    std::string name;
};
struct SubtypeSubmit {
    std::string name;
};
struct InstanceSubmit {
    std::string name;
};

// A cell query the chat's `arc` verb submitted to an ArcTask panel, parked
// until the module sends it to the solver bridge. The panel has no input of
// its own: queries are said in the Interlocutor.
struct ArcQuerySubmit {
    std::string query;
};

// ---------------------------------------------------------------- context

// What a context IS, as an entity: a thing statements can be made with regard
// to. `evaluator` names the registered query_contexts entry that knows how to
// answer for it, so the entity carries its own dispatch rather than the chat
// having to know what kinds of context exist.
struct ContextInfo {
    std::string evaluator;   // query_contexts id, e.g. "arc"
    std::string label;       // "ArcTask"
    std::string detail;      // what it is scoped to right now, e.g. the task id
    // The shorthand glyph the context wears when it is bound into a slot, same
    // register as any other entity's identity chip. A letter or digit; empty
    // would draw the wildcard, which is what an UNbound slot means.
    std::string symbol;
};

// A string some context recognises, declared by whatever owns that context's
// meaning rather than mirrored here: the ARC solver says it knows "Sun", that it
// is a colour, and that its own index for it is 4. One entity per term, related
// Context to the context that declared it, so a term is a thing in the world --
// bindable by the annotator, placeable in the type lattice, and scoped, so two
// tasks that spell things differently cannot collide.
//
// `color` is the one field the DECLARER does not decide: how a term looks is
// Thornfield's business, resolved at import from whatever the declarer's index
// means (an ARC colour value maps to the palette the cells are drawn with, so a
// Sun badge is the same yellow as a Sun cell). Zero means no opinion.
struct VocabTerm {
    std::string text;
    std::string kind;        // "color" | "state" | "abstraction" | ...
    int index = -1;          // the declarer's own numbering, -1 when it has none
    uint32_t color = 0;

    // The shorthand this term has been wearing. Remembered so a term keeps one
    // chip across sentences instead of taking whatever digit happened to be
    // free -- a symbol is per-sentence, but the thing it stands for is not.
    // Given up when another term in the same sentence already holds it.
    std::string symbol;
};

// Provenance: `X --Context--> context entity`. What a conversation is scoped to,
// and what everything the conversation produces was produced within.
//
// Never spoken. You do not say "Rose Poly in ArcTask" -- you establish that what
// you are doing happens within the ArcTask, and then you just say "Rose Poly".
// So this is ambient rather than an argument: a relationship on the
// conversation, read by a comprehension when it needs to know where it is being
// asked.
//
// Deliberately NOT exclusive, and registered transitive:
//
//   * Multiple -- a conversation can carry several contexts at once, and a
//     comprehension declared in it is asked of each. Nothing here caps the
//     count; the drop gesture toggles one context without disturbing the rest.
//   * Hierarchical -- a context entity wears Context itself, pointing at the
//     broader one it sits within (a task within a dataset, a dataset within a
//     project). Because the relation composes, the effective scope of a
//     conversation is the transitive closure, most specific first, and a context
//     in the middle of a chain need not know how to evaluate anything: it can be
//     pure containment with evaluators only at the leaves.
struct Context {};

// A badge that carries a context and can be dragged onto a conversation to
// scope it. Worn by a panel's identity badge, so the thing you drag is the
// thing that names the panel.
struct ContextBadge {
    flecs::entity context;   // the ContextInfo entity this badge stands for
};

// Somewhere a context badge can be dropped: a conversation, named by its leaf.
// `base_color` is the target's own resting fill, so the lit state while a drag
// hovers it can be handed back afterwards without the drag code knowing what
// colour the panel chose.
struct ContextDropTarget {
    flecs::entity chat_leaf;
    uint32_t base_color = 0x050505FF;
};

// A context badge held by the cursor. Singleton; an invalid badge means nothing
// is being dragged.
//
// `ghost` is the copy that follows the cursor -- without it a drag is invisible
// and you are asked to believe the editor is tracking something. It appears only
// once the cursor has moved past a small threshold, so a plain click on the
// badge is still a plain click.
struct ContextDrag {
    flecs::entity badge;
    flecs::entity ghost;
    float press_x = 0.0f;
    float press_y = 0.0f;
    bool dragging = false;
    bool over_target = false;
};

// Display order for an Entities panel: row position -> index into the backing
// data. Empty means identity, which is also what it falls back to whenever a
// ranking is unavailable, so the list is never blocked on the server.
struct EntityOrder {
    std::vector<size_t> order;

    size_t data_index(size_t row) const {
        return row < order.size() ? order[row] : row;
    }
};

// A vertical list whose rows exist as entities only while they are on screen.
//
// The backing data can hold a million rows: every row is exactly row_pitch tall,
// so the window of visible indices is arithmetic rather than a search, and only
// that window is ever built. Bounding the entity count -- not just the draw call
// count -- is the point, because the legacy layout pass walks the whole UI
// hierarchy every frame, so a row that merely exists offscreen costs as much as
// a visible one.
//
// Lives on the row container itself rather than the panel leaf, so it dies with
// the canvas children when the panel type changes and can never be left
// pointing at destroyed rows.
struct VirtualList {
    // Clips the rows and supplies the visible height. Rows overflow it by at
    // most one pitch, which the scissor crops.
    flecs::entity viewport;

    // Builds row `index` under `parent`. The list owns nothing about what a row
    // looks like; it only guarantees every row it asks for is row_pitch tall.
    std::function<void(flecs::entity parent, size_t index)> build_row;

    float row_pitch = 28.0f;
    float pad = 8.0f;
    size_t item_count = 0;

    // Pixels scrolled from the top of the virtual content. Written by input,
    // clamped by the list system against the current viewport height.
    float scroll = 0.0f;

    // The window currently built as entities. realized stays false until the
    // viewport has a measured height, so the first build waits for layout.
    size_t first_realized = 0;
    size_t realized_count = 0;
    bool realized = false;

    // Forces a rebuild of the current window even though it hasn't moved. Set
    // when something outside the list changes how a row should look -- the
    // selection moving, for instance.
    bool dirty = false;
};


// Particle animation for grid triangles - moving with velocity to collide at target
struct TriangleParticle {
    float targetX, targetY, targetZ;  // Final grid position (collision point)
    float localX, localY, localZ;     // Rotated local offset from tetrahedron centroid
    float vx, vy, vz;  // Velocity
    float collisionTime;  // When this particle reaches its target
    float u, v;  // UV coordinates (don't animate)
    float elapsedTime;  // Current time
    float hitTime;  // Time when particle first hit/locked (for glow effect)
    float baryX, baryY, baryZ;  // Barycentric coordinates for edge detection
    int vertexIndex;  // Which vertex in the buffer
    bool locked;  // Has reached target
    bool isCentral;  // Part of central triangle (for impact glow intensity)
    float pulseScale;  // Random scale variation for pulse (0.6 - 1.4)
    float pulseRotation;  // Random rotation for pulse asymmetry
};

// Noise tetrahedron that flies past without joining grid
struct NoiseTetrahedron {
    float x, y, z;      // Current position (centroid)
    float vz;           // Velocity towards camera
    float scale;        // Size multiplier
    // Random rotation (axis-angle)
    float axisX, axisY, axisZ;
    float rotAngle;
};

struct TimeEventRowChannel 
{   
    int scaleForMinimumCount;
};

struct CopyChildHeight {};

struct Graphics {
    NVGcontext* vg;

    // 3D rendering resources
    GLuint fbo;
    GLuint fboTexture;
    GLuint fboDepthRenderBuffer;
    GLuint planeVAO;
    GLuint planeVBO;
    GLuint planeEBO;
    GLuint gridVAO;
    GLuint gridVBO;
    GLuint gridEBO;
    GLuint shaderProgram;
    GLuint greyTexture;  // 1x1 grey texture for noise tetrahedrons
    float tiltAngle;
    int uiWidth;
    int uiHeight;
    bool useGridMode;
    float gridModeTransitionTimer;  // Timer for delayed transition to plane mode
    bool allParticlesLocked;        // Track if all particles are locked
    int gridVertexCount;

    // Particle system data
    std::vector<TriangleParticle> particles;
    std::vector<float> gridVertices;  // Store vertices for dynamic updates

    // Noise tetrahedrons (fly past, don't join grid)
    GLuint noiseVAO;
    GLuint noiseVBO;
    std::vector<NoiseTetrahedron> noiseParticles;
    std::vector<float> noiseVertices;
    int noiseVertexCount;

    // FTL deceleration state
    float decelerationTime;      // Time since start of deceleration
    float decelerationDuration;  // Total deceleration period
};

struct ParentClass {};
struct BFOSprite {};


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
struct EditorHeader {};
struct EditorOutline {};
struct EditorCanvas {};
struct EditorLeaf {};

struct UpperNode {};
struct LowerNode {};

struct LeftNode {};
struct RightNode {};

struct Align
{
    float self_horizontal;
    float self_vertical;
    float horizontal;
    float vertical;
};

// Expand Rect or RoundedRect to UIElement bounds of parent
// with some padding
struct Expand
{
    bool x_enabled;
    // TODO: Use a padding primitive rather than placing these on expand...
    float pad_left, pad_right;
    float x_percent; // 0.0 to 1.0

    bool y_enabled;
    float pad_top, pad_bottom;
    float y_percent;

    // If true (ImageRenderable only), do not upscale beyond native image size
    // (after applying ImageRenderable.scaleX/scaleY).
    bool cap_to_intrinsic = false;
};

// This is a special system tag to reduce UIElementBounds for LayoutBox
struct ContractBoundsPostLayout {};

struct UIContainer
{
    int pad_horizontal;
    int pad_vertical;
};

// Post expand layer to 'fit within editor panel bounds'
struct Constrain
{
    bool fit_x; // Scale x to fit within bounds (maintain ratio)
    bool fit_y; // Scale y to fit within bounds (maintain ratio)
};

// What the actual fuck is this?
struct ProportionalConstraint {
    float max_width;
    float max_height;
};

enum class EditorType
{
    Void,
    PeachCore,
    Condensate,
    ImaginaryInterlocutor,
    VNCStream,
    Healthbar,
    // Respawn,
    // Genome,
    InSilicoLab,
    Droid,
    Vision,
    Hearing,
    Memory,
    Bookshelf,
    Episodic,
    BFO,
    SceneGraph,
    DataFusion,
    Entities,
    Lexicon,
    Transducer,
    // A VNC stream hung as a plane in a walkable 3D room (screen3d.h).
    // Appended at the end: saved layouts store these as ints.
    Holodeck,
    // Type lattices: where an entity sits under subsumption, and how the
    // lattices of several entities line up when they form an analogy.
    Lattice,
    // Triple generalization: the seven adornments of a triple (Ego, Alter,
    // Bridge, ...) and their extensions -- see type_generalization.png.
    TypeGen,
    // An ARC-AGI task: train/test IO grid pairs rendered as a two-column
    // table, cells selected by click or by nlp_query against the Jane Eyre
    // solver bridge.
    ArcTask,
    // The solver's scene graph for the task in context, as an indented tree.
    VarRegistry,
    // SystemNavigator,

    // Bookshelf,
    // MelSpectrogram,
    // VNCStream,
    // CameraVideoFeed,
    // RobotActuators,
    // VirtualHumanoid,
    // EpisodicMemoryTimeline,
    // Reification,
    // ReadingRepresentation,
    // ProgramSynthesis,
    // Servers,
    // Healthbar,
    // Inventory,
    // Scheduling,
    // Chat,
    // ComputeProvisioning,
    // Backup
};

// This needs to be refactored to be a direct enum relationship in flecs once you grow up and become a competent person
struct EditorLeafData
{
    EditorType editor_type;
};

struct FilmstripChannel{};
// Relationship tag to link mel spec display to its source renderer entity
struct MelSpecSource{};
// Determine how to chunk and choose which frames to display

enum class FilmstripMode {
    Uniform,     // Frames added at fixed time intervals, adjacent layout
    Stegosaurus  // Frames added on DINO spikes, time-based positioning with overlaps
};

// Component to track when a filmstrip frame was captured (for time-based positioning)
struct FilmstripFrameTime {
    double capture_time;  // glfwGetTime() when captured
};

struct FilmstripData
{
    // TODO: This should probably be a dynamic value based on container width...
    int frame_limit;
    // Should they be stored as entities or paths?
    std::vector<flecs::entity> frames;
    // Scroll tracking for realtime left-to-right scroll over 24 seconds
    float scroll_offset = 0.0f;  // Current scroll progress (0.0 to 1.0 of one frame width)
    float elapsed_time = 0.0f;   // Time elapsed in current scroll cycle
    size_t total_frames_added = 0;  // Counter incremented each time a frame is pushed
    size_t last_seen_frame_count = 0;  // To detect when total_frames_added changes
    static constexpr float SCROLL_DURATION = 24.0f;  // Seconds for full container to scroll

    FilmstripMode mode = FilmstripMode::Uniform;

    // FilmstripMode::Stegosaurus parameters
    float spike_threshold = 0.15f;   // Minimum cosDiff to trigger a spike (0.0-1.0)
    float last_dino_value = 0.0f;    // Last DINO cosDiff value seen
    float spike_cooldown = 1.0f;     // Minimum seconds between spike captures
    float time_since_spike = 0.0f;   // Time since last spike capture
    bool pending_capture = false;    // Flag to signal capture should happen
    double pending_spike_time = 0.0; // Time when the spike was detected (for accurate frame positioning)

    // FilmstripMode::Uniform parameters: track mel spec scroll position for sync
    size_t last_capture_scroll_commands = 0;  // totalScrollCommands at last frame capture
};

// Timestamped data point for line chart
struct LineChartPoint {
    float value;
    double capture_time;  // Wall clock time when this value was captured
};

// Line chart channel for streaming data visualization (e.g., DINO cosine similarity)
struct LineChartData
{
    std::vector<LineChartPoint> points;  // Circular buffer of timestamped values
    size_t write_pos = 0;       // Current write position in circular buffer
    size_t capacity = 0;        // Max number of data points
    float min_value = 0.0f;     // Min value for normalization
    float max_value = 1.0f;     // Max value for normalization
    uint32_t fill_color = 0xFFFFFF40;  // RGBA fill color (semi-transparent white)
    uint32_t line_color = 0xFFFFFFFF;  // RGBA line color (solid white)
    float sample_interval = 0.0f;  // Seconds between samples (0 = every frame)
    float time_since_sample = 0.0f;  // Time accumulator for sampling
    float processing_lag = 0.0f;  // Additional lag in seconds (e.g., DINO inference time)

    static constexpr float WINDOW_DURATION = 24.0f;  // Total time window in seconds (matches filmstrip)

    void push(float value, double timestamp) {
        if (capacity == 0) return;
        LineChartPoint point = {value, timestamp};
        if (points.size() < capacity) {
            points.push_back(point);
        } else {
            points[write_pos] = point;
        }
        write_pos = (write_pos + 1) % capacity;
    }

    // Get point at index (0 = oldest, size-1 = newest)
    LineChartPoint get(size_t index) const {
        if (points.empty() || index >= points.size()) return {0.0f, 0.0};
        if (points.size() < capacity) {
            return points[index];
        }
        return points[(write_pos + index) % capacity];
    }

    size_t size() const { return points.size(); }
};

// Tag for LineChart channel type
struct LineChartChannel{};

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
    Image,
    Line,
    QuadraticBezier,
    CustomRenderable,
};

