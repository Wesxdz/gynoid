#pragma once

// The function library: the set of things the gynoid can be *asked to do*, as
// opposed to things it can be told. A capability registers itself here and
// becomes callable by anything that can name it -- a typed command line, a
// language model emitting tool calls, or a custom parser of your own that never
// produces JSON at all.
//
// The registry deliberately does not know who is calling. There are two entry
// points, and neither is privileged:
//
//   invoke(call)         structured call, name + named arguments. What a model
//                        targets, and what any programmatic caller should use.
//   dispatch_text(line)  a typed line -- "Speak as dramatic: hello there" --
//                        matched against the registered names and handed to the
//                        function's own parser. The chat box is one frontend
//                        for this, not its owner.
//
// Arguments are strings by design. Every transport that reaches here already
// carries strings (msgpack maps to the Python servers, typed input, model
// output), the declared types in FunctionParam are what a model reads to choose
// values, and coercion belongs in the function that knows what it wants. A
// variant type here would be a third representation to convert between.
//
// Registration mirrors panels::register_panel -- same shape, same reasoning:
// the capability owns its definition and the monolith owns nothing.

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace functions {

using Args = std::map<std::string, std::string>;

struct Param
{
    const char* name;
    const char* type;          // "string" | "number" | "boolean"
    const char* description;   // read by a model choosing a value
    bool required = false;
};

// What a call reports back. Long-running work (speech, a model round trip)
// resolves after this returns, so `ok` means the call was accepted and
// dispatched, not that its effect has finished. A function whose effect can
// fail later should say so through its own channel.
struct Result
{
    bool ok = false;
    std::string message;
};

struct Def
{
    const char* name;          // lowercase, the word a caller types or emits
    const char* description;   // one line; this is what a model selects on
    std::vector<Param> params;

    // The call itself.
    std::function<Result(const Args&)> invoke;

    // Optional: turn the remainder of a typed line into arguments. When absent,
    // the whole remainder becomes the first required parameter, which is what
    // makes a one-argument function work with no parser of its own.
    std::function<Args(const std::string& tail)> parse;
};

void register_function(const Def& def);

// Exact name match, case-insensitive. Null when there is no such function.
const Def* find(const std::string& name);

// Everything registered, for a caller that wants to choose among them.
const std::vector<Def>& all();

// Call by name. Reports a miss rather than throwing, because the caller may be
// a model that invented the name.
Result invoke(const std::string& name, const Args& args);

// Try to read `line` as a call: leading word matched against the registry, the
// rest handed to that function's parser. Returns false when the line does not
// begin with a known function, leaving it to be treated as ordinary text.
bool dispatch_text(const std::string& line, Result* out = nullptr);

// JSON Schema description of the library, for a model that expects tool
// definitions. A custom caller can ignore this entirely and read all() instead.
std::string describe_json();

} // namespace functions
