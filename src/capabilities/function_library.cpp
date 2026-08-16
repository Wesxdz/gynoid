#include "function_library.h"

#include <algorithm>
#include <cctype>

#include <nlohmann/json.hpp>

namespace functions {
namespace {

// Function-local rather than a file-scope global: registration happens from
// static initializers and module constructors whose order across translation
// units is unspecified, and this guarantees the vector exists before the first
// one runs.
std::vector<Def>& registry()
{
    static std::vector<Def> defs;
    return defs;
}

std::string lower(const std::string& s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

std::string trim(const std::string& s)
{
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

// Default text parser: the whole tail is the first required parameter. This is
// what lets "Speak hello there" work without the speak function owning a parser.
Args default_parse(const Def& def, const std::string& tail)
{
    for (const Param& p : def.params) {
        if (p.required) return {{p.name, trim(tail)}};
    }
    return def.params.empty() ? Args{} : Args{{def.params.front().name, trim(tail)}};
}

} // namespace

void register_function(const Def& def)
{
    if (!def.name || !def.invoke) return;

    // Re-registering a name replaces it, so a capability can be reloaded
    // without the registry accumulating stale duplicates.
    std::vector<Def>& defs = registry();
    const std::string key = lower(def.name);
    for (Def& existing : defs) {
        if (lower(existing.name) == key) { existing = def; return; }
    }
    defs.push_back(def);
}

const Def* find(const std::string& name)
{
    const std::string key = lower(trim(name));
    for (const Def& def : registry()) {
        if (lower(def.name) == key) return &def;
    }
    return nullptr;
}

const std::vector<Def>& all()
{
    return registry();
}

Result invoke(const std::string& name, const Args& args)
{
    const Def* def = find(name);
    if (!def) return {false, "no function named '" + name + "'"};

    for (const Param& p : def->params) {
        // Empty counts as missing. A text parser producing "" for the tail of a
        // bare "Speak" is the common case, and reporting that as a satisfied
        // argument would let every function repeat the same check.
        const auto it = args.find(p.name);
        if (p.required && (it == args.end() || it->second.empty())) {
            return {false, std::string("'") + def->name + "' needs " + p.name};
        }
    }
    return def->invoke(args);
}

bool dispatch_text(const std::string& line, Result* out)
{
    const std::string text = trim(line);
    if (text.empty()) return false;

    // The leading word names the function; everything after it is that
    // function's business, not the registry's.
    const auto split = text.find_first_of(" \t\n");
    const std::string head = text.substr(0, split);
    const std::string tail = (split == std::string::npos) ? "" : text.substr(split + 1);

    const Def* def = find(head);
    if (!def) return false;

    const Args args = def->parse ? def->parse(tail) : default_parse(*def, tail);
    const Result result = invoke(def->name, args);
    if (out) *out = result;
    return true;
}

std::string describe_json()
{
    nlohmann::json tools = nlohmann::json::array();
    for (const Def& def : registry()) {
        nlohmann::json properties = nlohmann::json::object();
        nlohmann::json required = nlohmann::json::array();
        for (const Param& p : def.params) {
            properties[p.name] = {{"type", p.type}, {"description", p.description}};
            if (p.required) required.push_back(p.name);
        }
        tools.push_back({
            {"name", def.name},
            {"description", def.description},
            {"input_schema", {
                {"type", "object"},
                {"properties", properties},
                {"required", required},
            }},
        });
    }
    return tools.dump(2);
}

} // namespace functions
