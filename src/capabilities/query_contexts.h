#pragma once

// Query contexts: the things an annotated comprehension can be RUN AGAINST.
//
// The Q gesture in the Interlocutor turns an annotated span into an intension --
// "Sun polyominoes" as a query rather than a remark. An intension on its own
// selects nothing; it has to be evaluated somewhere. A context is that
// somewhere: the ARC panel evaluates terms against the loaded task's cells, and
// later the entity world, an episode, or a document can each answer the same
// comprehension their own way.
//
// The registry deliberately knows nothing about what a term means. It hands the
// context the words exactly as they were said -- plural, inflected, unresolved
// -- because resolving "polyominoes" to something selectable is the context's
// own job: it owns the vocabulary, it owns whatever it has been taught, and it
// is the one that can say which words it failed to understand. Nothing about a
// word's meaning belongs in this file.
//
// Same shape as panels::register_panel and functions::register_function, for the
// same reason: the capability owns its definition and the monolith owns nothing.
//
// ---- Adding a context later ----
// Fill a Context and register it (typically from a panel module's constructor,
// beside its register_panel call):
//
//   query_contexts::register_context({
//       "episode", "Episode",
//       [] { return an_episode_is_loaded(); },                 // optional
//       [](const query_contexts::Comprehension& q) { ... },
//   });
//
// Two rules keep that robust. Read the Comprehension fields you understand and
// ignore the rest -- the struct grows as more of the annotation is published, and
// growth must never break a context that predates it. And re-check your own
// state inside execute(): a context outlives any one panel instance, so being
// registered is not proof of being usable.

#include <functional>
#include <string>
#include <vector>

namespace query_contexts {

// One term of a comprehension, as the annotation bound it. `word` is the
// surface form; `binding_symbol` is the shorthand id its badge wears, empty for
// an unbound word.
struct Term
{
    enum class Role {
        Member,     // a bare term inside the frame
        Source,     // the source end of an annotated relationship
        Relation,   // the relationship itself
        Target,     // its target end
    };

    std::string word;
    std::string binding_symbol;
    Role role = Role::Member;
};

// A comprehension handed over for evaluation.
//
// Additive by design: a context reads the fields it understands and ignores the
// rest, so publishing more of the annotation later cannot break an existing
// context. Prefer adding a field here over widening execute()'s signature.
struct Comprehension
{
    std::vector<Term> terms;

    // The frame's two ids, as the Q gesture assigns them: the intension is the
    // query as an entity, the extension is the set it denotes. Carried so a
    // context can reify its answer against them -- cache a result set, name it,
    // relate it back to the sentence that asked.
    std::string intension_symbol;
    std::string extension_symbol;

    // The span as it was said, for messages a person has to read.
    std::string source_text;

    // The words in order, for a context that does not care about roles.
    std::vector<std::string> words() const
    {
        std::vector<std::string> out;
        out.reserve(terms.size());
        for (const Term& term : terms) out.push_back(term.word);
        return out;
    }
};

struct Result
{
    bool ok = false;
    std::string message;   // what happened, in the user's terms
};

struct Context
{
    std::string id;      // stable key; registering the same id again replaces it
    std::string label;   // shown when choosing among contexts

    // Whether this context can answer right now -- a task loaded, a server up.
    // Empty means "always". Used to choose among contexts and to tell a person
    // why nothing ran; execute() must still re-check, since availability can
    // lapse between the question and the answer.
    std::function<bool()> available;

    // Evaluates the comprehension. Reports acceptance, not completion: a context
    // that has to ask a server answers later, through its own panel.
    std::function<Result(const Comprehension& query)> execute;
};

void register_context(Context context);
void unregister_context(const std::string& id);

// Everything registered, for a caller that wants to choose among them.
const std::vector<Context>& all();

// The ids that would answer right now.
std::vector<std::string> available_ids();

// Run against one context by id. Reports a miss rather than throwing, because
// the id may name a context that was never registered or is not ready.
Result execute(const std::string& id, const Comprehension& query);

// Run against the only context that could answer. With none, or with several,
// it declines and says so -- picking silently would run the query somewhere the
// user was not looking.
Result execute_sole(const Comprehension& query);

} // namespace query_contexts
