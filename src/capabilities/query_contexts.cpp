#include "query_contexts.h"

#include <algorithm>

namespace {

// Function-local static so contexts registering during world import never race a
// global's construction order -- same reasoning as panels::registry.
std::vector<query_contexts::Context>& contexts()
{
    static std::vector<query_contexts::Context> v;
    return v;
}

bool is_available(const query_contexts::Context& context)
{
    // No predicate means the context is always willing to be asked.
    return !context.available || context.available();
}

} // namespace

namespace query_contexts {

void register_context(Context context)
{
    for (Context& existing : contexts()) {
        if (existing.id == context.id) {
            existing = std::move(context);   // a reopened panel replaces its own
            return;
        }
    }
    contexts().push_back(std::move(context));
}

void unregister_context(const std::string& id)
{
    std::vector<Context>& list = contexts();
    list.erase(std::remove_if(list.begin(), list.end(),
                              [&](const Context& c) { return c.id == id; }),
               list.end());
}

const std::vector<Context>& all()
{
    return contexts();
}

std::vector<std::string> available_ids()
{
    std::vector<std::string> ids;
    for (const Context& context : contexts())
        if (is_available(context)) ids.push_back(context.id);
    return ids;
}

Result execute(const std::string& id, const Comprehension& query)
{
    for (const Context& context : contexts()) {
        if (context.id != id) continue;
        if (!context.execute) return {false, context.id + " cannot run queries"};
        if (!is_available(context)) return {false, context.label + " is not ready"};
        return context.execute(query);
    }
    return {false, "no query context named " + id};
}

Result execute_sole(const Comprehension& query)
{
    const std::vector<std::string> ready = available_ids();

    if (ready.empty()) {
        // Registered-but-unavailable is a different problem from nothing being
        // open, and the difference is what tells someone what to do next.
        if (contexts().empty())
            return {false, "no query context open"};
        std::string names;
        for (const Context& context : contexts()) {
            if (!names.empty()) names += ", ";
            names += context.label;
        }
        return {false, "no context ready to answer (" + names + ")"};
    }

    if (ready.size() > 1) {
        std::string names;
        for (const std::string& id : ready) {
            if (!names.empty()) names += ", ";
            names += id;
        }
        return {false, "several contexts could answer (" + names + "); name one"};
    }

    return execute(ready.front(), query);
}

} // namespace query_contexts
