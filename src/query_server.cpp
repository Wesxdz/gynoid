#include "query_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <signal.h>
#include <setjmp.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <cfloat>

namespace query_server {

// Global state for the query server
static flecs::world* g_world = nullptr;
static spatial::SpatialIndexManager* g_spatial_index = nullptr;
// Component serializers removed - now using flecs reflection with entity.to_json()
static SpatialQueryHandler g_spatial_handler = nullptr;

// Thread-local jump buffer for error recovery
thread_local jmp_buf error_jump_buffer;
thread_local bool error_handler_set = false;

// Signal handler for catching segfaults during query processing
void query_error_handler(int sig) {
    if (error_handler_set) {
        longjmp(error_jump_buffer, 1);
    }
}


// Serialized responses are built into one buffer and sent in a single write.
// 16KB truncated an unfiltered listing partway through flecs' own builtins,
// before any of the world's entities were reached -- the tail was silently
// dropped, so a query looked like it had simply returned less.
static constexpr size_t QUERY_RESPONSE_CAPACITY = 1u << 20;

// Text carried by entities the editor creates. Registered here rather than by
// the host application so any world running a query server can accept created
// entities without first defining a schema for them. Named "Text" explicitly,
// since the C++ symbol would otherwise surface in the UI as query_server::EntityText.
struct EntityText {
    std::string value;
};

// When an entity was created, as seconds since the epoch. Stored as a number
// because that is what it is -- an instant, comparable and sortable. Turning it
// into something a person wants to read is the display layer's job, not this
// one's, so nothing here formats it.
struct EntityCreatedAt {
    double time;
};

// create <Type> <text...>
//
// Everything after the type name is the text, spaces included, so no quoting or
// escaping is needed for the thing this exists to carry: a typed message.
// The type is created on demand if the world has never seen it, which is what
// lets the editor introduce a kind of entity the world was not compiled with.
static bool handle_create(int client_socket, const char* request) {
    if (strncmp(request, "create ", 7) != 0) return false;

    const char* cursor = request + 7;
    while (*cursor == ' ') cursor++;

    const char* type_end = cursor;
    while (*type_end && *type_end != ' ') type_end++;

    std::string type_name(cursor, type_end - cursor);
    const char* text = type_end;
    while (*text == ' ') text++;

    char response[512];
    if (type_name.empty()) {
        snprintf(response, sizeof(response), "{\"error\": \"create needs a type\"}\n");
        send(client_socket, response, strlen(response), 0);
        return true;
    }

    flecs::entity type = g_world->lookup(type_name.c_str());
    if (!type.is_valid()) {
        type = g_world->entity(type_name.c_str());
    }

    // Anonymous: the text is the identity here, not a path-safe name. Entity
    // names are scoped and separator-sensitive, and a sentence is neither.
    flecs::entity created = g_world->entity();
    created.add(type);
    created.set<EntityText>({std::string(text)});
    created.set<EntityCreatedAt>({(double)std::time(nullptr)});

    snprintf(response, sizeof(response),
             "{\"status\": \"OK\", \"id\": %llu}\n",
             (unsigned long long)created.id());
    send(client_socket, response, strlen(response), 0);
    return true;
}

// The first identifier in `expr` that names nothing in this world, or empty.
//
// A query bar is typed into, and a name the world has never seen is the normal
// case rather than an error: "Chat" matches nothing until the first chat entity
// exists, and a user browsing types should get an empty list, not a failure.
// flecs itself refuses to compile such a query, so the check happens first.
//
// Deliberately conservative -- only bare identifiers are judged. Anything with
// an operator, a variable, or a wildcard is left for flecs to rule on.
static std::string first_unknown_identifier(const std::string& expr) {
    size_t pos = 0;
    while (pos <= expr.size()) {
        size_t end = expr.find(',', pos);
        if (end == std::string::npos) end = expr.size();

        std::string term = expr.substr(pos, end - pos);
        pos = end + 1;

        // Trim, then drop the operators that may lead a term.
        size_t b = term.find_first_not_of(" \t!?[|");
        if (b == std::string::npos) continue;
        size_t e = term.find_last_not_of(" \t]|");
        term = term.substr(b, e - b + 1);

        // A relation term names its relation before the parenthesis.
        size_t paren = term.find('(');
        if (paren != std::string::npos) term = term.substr(0, paren);

        if (term.empty() || term == "_" || term == "*" || term[0] == '$') continue;

        bool plain = true;
        for (char c : term) {
            if (!(isalnum((unsigned char)c) || c == '_' || c == '.' || c == ':')) {
                plain = false;
                break;
            }
        }
        if (!plain) continue;   // ranges, comparisons, anything else: flecs decides

        if (!g_world->lookup(term.c_str()).is_valid()) return term;
    }
    return std::string();
}

// True for entities flecs defines for its own bookkeeping -- traits, builtin
// components, module scopes. A query of "_" matches literally everything, so
// without this an unfiltered listing is ~40 entities of ECS plumbing before any
// of the world's own data. Anything under the `flecs` scope is internal.
static bool is_internal_entity(flecs::entity e) {
    if (e.has(flecs::Module) || e.has(flecs::Prefab) ||
        e.has<flecs::Member>() || e.has<flecs::Type>()) {
        return true;
    }
    // Anything scoped under the `flecs` module is the ECS describing itself.
    // Walked rather than matched on the path string, which is not reliably
    // prefixed for builtins.
    for (flecs::entity p = e.parent(); p.is_valid(); p = p.parent()) {
        if (p.id() == flecs::Flecs) return true;
    }
    return false;
}

// entity.to_json() carries name, tags, pairs and components, but not the entity
// id -- and a caller that wants to refer back to an entity needs the id, not a
// position in a result list. Spliced in rather than rebuilt, so flecs keeps
// owning the serialization of everything else.
static std::string entity_json_with_id(flecs::entity e) {
    std::string json = e.to_json().c_str();
    std::string prefix = "{\"id\": " + std::to_string((uint64_t)e.id()) + ", ";
    if (!json.empty() && json[0] == '{') {
        return prefix + json.substr(1);
    }
    return json;
}

// Initialize the query server
void initialize(flecs::world* world, spatial::SpatialIndexManager* spatial_index) {
    g_world = world;
    g_spatial_index = spatial_index;

    // Reflection metadata, so a created entity's text round-trips through
    // to_json() like any component the host declared itself.
    world->component<EntityText>("Text").member<std::string>("value");
    world->component<EntityCreatedAt>("CreatedAt").member<double>("time");
}

// Register component serializer
// register_component_serializer removed - now using flecs reflection

// Register spatial query handler
void register_spatial_handler(SpatialQueryHandler handler) {
    g_spatial_handler = handler;
}

// Helper function to remove a single field's queries from string
void remove_field_queries(std::string& result, const std::string& field_name) {
    while (true) {
        size_t field_pos = result.find(field_name);
        if (field_pos == std::string::npos) break;

        size_t end_pos = field_pos + field_name.length();

        // Skip whitespace after field name
        while (end_pos < result.length() && (result[end_pos] == ' ' || result[end_pos] == '\t')) {
            end_pos++;
        }

        bool erased = false;

        // Check for operators or parenthesis
        if (end_pos < result.length()) {
            // Handle field(min,max) pattern
            if (result[end_pos] == '(') {
                size_t close_paren = result.find(')', end_pos);
                if (close_paren != std::string::npos) {
                    result.erase(field_pos, close_paren - field_pos + 1);
                    erased = true;
                }
            }
            // Handle comparison operators: >=, <=, ==, >, <
            else if (end_pos < result.length() &&
                     (result[end_pos] == '>' || result[end_pos] == '<' || result[end_pos] == '=')) {
                // Skip operator (could be 1 or 2 characters)
                if (end_pos + 1 < result.length() && result[end_pos + 1] == '=') {
                    end_pos += 2;
                } else {
                    end_pos++;
                }

                // Skip whitespace after operator
                while (end_pos < result.length() && (result[end_pos] == ' ' || result[end_pos] == '\t')) {
                    end_pos++;
                }

                // Skip the number (digits, optional decimal point, optional sign)
                if (end_pos < result.length() && (result[end_pos] == '-' || result[end_pos] == '+')) {
                    end_pos++;
                }
                while (end_pos < result.length() &&
                       (isdigit(result[end_pos]) || result[end_pos] == '.')) {
                    end_pos++;
                }

                result.erase(field_pos, end_pos - field_pos);
                erased = true;
            }
        }

        // If we didn't erase anything, break to avoid infinite loop
        if (!erased) {
            break;
        }
    }
}

// Helper function to parse a single field's range constraints
bool parse_field_constraints(const char* query_str, const std::string& field_name,
                             std::unordered_set<uint64_t>& results, std::string& error_msg) {
    const float FIELD_MIN = FLT_MIN;
    const float FIELD_MAX = FLT_MAX;
    auto index = g_spatial_index->get_1d_index(field_name);
    if (!index) {
        return false; // Index not registered
    }

    bool found_any = false;
    float combined_min = FIELD_MIN;
    float combined_max = FIELD_MAX;

    // Search for all occurrences of this field in the query
    const char* search_pos = query_str;
    size_t field_len = field_name.length();

    while (true) {
        const char* field_start = strstr(search_pos, field_name.c_str());
        if (!field_start) break;

        // Check if this is a complete word match (not a substring within another word)
        // Character before field should be start of string, whitespace, or comma
        if (field_start != query_str) {
            char before = *(field_start - 1);
            if (before != ' ' && before != '\t' && before != ',') {
                // Not a word boundary, skip this match
                search_pos = field_start + 1;
                continue;
            }
        }

        const char* op_pos = field_start + field_len;
        // Skip whitespace
        while (*op_pos == ' ' || *op_pos == '\t') op_pos++;

        // Character after field should be whitespace or an operator
        if (*op_pos == '\0' || (*op_pos != '>' && *op_pos != '<' && *op_pos != '=' && *op_pos != '(')) {
            // Not followed by an operator, skip this match
            search_pos = field_start + field_len;
            continue;
        }

        float value;
        float min_val = FIELD_MIN;
        float max_val = FIELD_MAX;
        bool found_operator = false;

        // Check for >= operator
        if (strncmp(op_pos, ">=", 2) == 0) {
            if (sscanf(op_pos + 2, "%f", &value) == 1) {
                min_val = value;
                max_val = FIELD_MAX;
                found_operator = true;
            }
        }
        // Check for <= operator
        else if (strncmp(op_pos, "<=", 2) == 0) {
            if (sscanf(op_pos + 2, "%f", &value) == 1) {
                min_val = FIELD_MIN;
                max_val = value;
                found_operator = true;
            }
        }
        // Check for == operator
        else if (strncmp(op_pos, "==", 2) == 0) {
            if (sscanf(op_pos + 2, "%f", &value) == 1) {
                min_val = value;
                max_val = value;
                found_operator = true;
            }
        }
        // Check for > operator (must come after >=)
        else if (*op_pos == '>') {
            if (sscanf(op_pos + 1, "%f", &value) == 1) {
                min_val = value + 0.001f; // Exclusive
                max_val = FIELD_MAX;
                found_operator = true;
            }
        }
        // Check for < operator (must come after <=)
        else if (*op_pos == '<') {
            if (sscanf(op_pos + 1, "%f", &value) == 1) {
                min_val = FIELD_MIN;
                max_val = value - 0.001f; // Exclusive
                found_operator = true;
            }
        }
        // Check for field(min,max) pattern
        else if (*op_pos == '(') {
            if (sscanf(op_pos + 1, "%f,%f)", &min_val, &max_val) == 2) {
                found_operator = true;
            }
        }

        if (found_operator) {
            // Combine with previous constraints (intersection)
            combined_min = std::max(combined_min, min_val);
            combined_max = std::min(combined_max, max_val);
            found_any = true;
        }

        // Move search position forward
        search_pos = field_start + field_len;
    }

    if (found_any) {
        if (combined_min > combined_max) {
            error_msg = "Conflicting " + field_name + " constraints result in empty set";
            return false;
        }
        results = index->range_query(combined_min, combined_max);
        printf("DEBUG: Field '%s' query [%.2f, %.2f] returned %zu entities\n",
               field_name.c_str(), combined_min, combined_max, results.size());
        for (uint64_t eid : results) {
            printf("  - Entity ID: %lu\n", eid);
        }
        return true;
    }

    return false;
}

// Parse range queries (1D partitioning) for all supported fields
bool parse_range_query(const char* query_str, std::unordered_set<uint64_t>& results,
                      std::string& error_msg) {
    std::vector<std::string> fields = g_spatial_index->get_registered_fields();

    printf("DEBUG: parse_range_query called with query: '%s'\n", query_str);
    printf("DEBUG: Registered fields: ");
    for (const auto& f : fields) {
        printf("'%s' ", f.c_str());
    }
    printf("\n");

    bool found_any = false;
    std::unordered_set<uint64_t> combined_results;

    for (const auto& field : fields) {
        std::unordered_set<uint64_t> field_results;
        std::string field_error;

        if (parse_field_constraints(query_str, field, field_results, field_error)) {
            if (!field_error.empty()) {
                error_msg = field_error;
                return false;
            }

            if (!found_any) {
                // First field result
                combined_results = field_results;
                found_any = true;
            } else {
                // Intersect with previous results
                std::unordered_set<uint64_t> intersected;
                for (uint64_t eid : combined_results) {
                    if (field_results.find(eid) != field_results.end()) {
                        intersected.insert(eid);
                    }
                }
                combined_results = intersected;
            }
        }
    }

    if (found_any) {
        results = combined_results;
        return true;
    }

    return false;
}

// Parse string equality queries
bool parse_string_query(const char* query_str, std::unordered_set<uint64_t>& results,
                       std::string& error_msg) {
    std::vector<std::string> fields = g_spatial_index->get_registered_string_fields();

    printf("DEBUG: parse_string_query called with query: '%s'\n", query_str);
    printf("DEBUG: Registered string fields: ");
    for (const auto& f : fields) {
        printf("'%s' ", f.c_str());
    }
    printf("\n");

    bool found_any = false;
    std::unordered_set<uint64_t> combined_results;

    for (const auto& field : fields) {
        // Look for pattern: field=="value" or field == "value"
        const char* search_pos = query_str;
        size_t field_len = field.length();

        while (true) {
            const char* field_start = strstr(search_pos, field.c_str());
            if (!field_start) break;

            // Check word boundary before field
            if (field_start != query_str) {
                char before = *(field_start - 1);
                if (before != ' ' && before != '\t' && before != ',') {
                    search_pos = field_start + 1;
                    continue;
                }
            }

            const char* op_pos = field_start + field_len;
            // Skip whitespace
            while (*op_pos == ' ' || *op_pos == '\t') op_pos++;

            // Check for == operator
            if (strncmp(op_pos, "==", 2) != 0) {
                search_pos = field_start + field_len;
                continue;
            }
            op_pos += 2;

            // Skip whitespace after ==
            while (*op_pos == ' ' || *op_pos == '\t') op_pos++;

            // Expect opening quote
            if (*op_pos != '"') {
                search_pos = field_start + field_len;
                continue;
            }
            op_pos++;

            // Find closing quote
            const char* closing_quote = strchr(op_pos, '"');
            if (!closing_quote) {
                search_pos = field_start + field_len;
                continue;
            }

            // Extract the string value
            std::string value(op_pos, closing_quote - op_pos);
            printf("DEBUG: Found string query: %s==\"%s\"\n", field.c_str(), value.c_str());

            auto index = g_spatial_index->get_string_index(field);
            if (index) {
                std::unordered_set<uint64_t> field_results = index->exact_match(value);

                if (!found_any) {
                    combined_results = field_results;
                    found_any = true;
                } else {
                    // Intersect with previous results
                    std::unordered_set<uint64_t> intersected;
                    for (uint64_t eid : combined_results) {
                        if (field_results.find(eid) != field_results.end()) {
                            intersected.insert(eid);
                        }
                    }
                    combined_results = intersected;
                }
            }

            search_pos = closing_quote + 1;
        }
    }

    if (found_any) {
        results = combined_results;
        return true;
    }

    return false;
}

// Remove partition queries from string
std::string remove_partition_queries(const char* query_str) {
    std::string result = query_str;

    // Remove spatial: queries
    const char* spatial_start = strstr(result.c_str(), "spatial:");
    if (spatial_start) {
        const char* spatial_end = strchr(spatial_start, ')');
        if (spatial_end) {
            size_t start_pos = spatial_start - result.c_str();
            size_t end_pos = spatial_end - result.c_str() + 1;
            result.erase(start_pos, end_pos - start_pos);
        }
    }

    // Remove all field range queries from registered fields
    std::vector<std::string> fields = g_spatial_index->get_registered_fields();
    for (const auto& field : fields) {
        remove_field_queries(result, field);
    }

    // Remove all string equality queries from registered string fields
    std::vector<std::string> string_fields = g_spatial_index->get_registered_string_fields();
    for (const auto& field : string_fields) {
        // Remove pattern: field=="value" or field == "value"
        while (true) {
            size_t field_pos = result.find(field);
            if (field_pos == std::string::npos) break;

            // Check word boundary
            if (field_pos > 0) {
                char before = result[field_pos - 1];
                if (before != ' ' && before != '\t' && before != ',') {
                    // Not a word boundary, but we can't skip past it easily in string,
                    // so just break to avoid infinite loop
                    break;
                }
            }

            size_t op_pos = field_pos + field.length();
            while (op_pos < result.length() && (result[op_pos] == ' ' || result[op_pos] == '\t')) {
                op_pos++;
            }

            if (op_pos + 1 < result.length() && result.substr(op_pos, 2) == "==") {
                op_pos += 2;
                while (op_pos < result.length() && (result[op_pos] == ' ' || result[op_pos] == '\t')) {
                    op_pos++;
                }

                if (op_pos < result.length() && result[op_pos] == '"') {
                    op_pos++;
                    size_t closing_quote = result.find('"', op_pos);
                    if (closing_quote != std::string::npos) {
                        size_t erase_end = closing_quote + 1;
                        // Also consume trailing comma and/or whitespace
                        while (erase_end < result.length() &&
                               (result[erase_end] == ' ' || result[erase_end] == '\t' || result[erase_end] == ',')) {
                            erase_end++;
                            // If we consumed a comma, also consume whitespace after it
                            if (result[erase_end - 1] == ',') {
                                while (erase_end < result.length() &&
                                       (result[erase_end] == ' ' || result[erase_end] == '\t')) {
                                    erase_end++;
                                }
                                break; // Only consume one comma
                            }
                        }
                        result.erase(field_pos, erase_end - field_pos);
                        continue;
                    }
                }
            }
            break;
        }
    }

    // Clean up: trim commas and whitespace
    while (!result.empty() && (result.front() == ' ' || result.front() == ',' || result.front() == '\t'))
        result.erase(0, 1);
    while (!result.empty() && (result.back() == ' ' || result.back() == ',' || result.back() == '\t'))
        result.pop_back();

    return result;
}

// Query handler function
void handle_query(int client_socket, const char* query_str) {
    if (!g_world) {
        const char* error = "{\"error\": \"World not initialized\"}\n";
        send(client_socket, error, strlen(error), 0);
        return;
    }

    // Set up signal handler for this thread
    struct sigaction sa, old_sa;
    sa.sa_handler = query_error_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &old_sa);

    // Set up error recovery point
    if (setjmp(error_jump_buffer) != 0) {
        // We got here from a signal handler (segfault)
        error_handler_set = false;
        sigaction(SIGSEGV, &old_sa, nullptr);
        const char* error = "{\"error\": \"Query caused internal error (invalid query syntax or component)\"}\n";
        send(client_socket, error, strlen(error), 0);
        printf("Query caused segfault: %s\n", query_str);
        return;
    }

    error_handler_set = true;

    try {
        // Check for partition queries (both range and spatial)
        std::unordered_set<uint64_t> partition_filter;
        std::string partition_error;
        bool use_partition_filter = false;

        // Try range query (1D partitioning) first
        bool has_range = parse_range_query(query_str, partition_filter, partition_error);
        if (has_range) {
            use_partition_filter = true;
        }

        // Try string equality query
        std::unordered_set<uint64_t> string_results;
        std::string string_error;
        bool has_string = parse_string_query(query_str, string_results, string_error);
        if (has_string) {
            if (use_partition_filter) {
                // Intersect with existing partition filter
                std::unordered_set<uint64_t> intersected;
                for (uint64_t eid : partition_filter) {
                    if (string_results.find(eid) != string_results.end()) {
                        intersected.insert(eid);
                    }
                }
                partition_filter = intersected;
            } else {
                partition_filter = string_results;
                use_partition_filter = true;
            }
        }

        // Try spatial query (2D partitioning) using application handler
        if (g_spatial_handler && strstr(query_str, "spatial:") != nullptr) {
            std::unordered_set<uint64_t> spatial_results = g_spatial_handler(query_str, g_spatial_index);
            // Always use partition filter for spatial queries, even if results are empty
            if (use_partition_filter) {
                // Intersect both filters
                std::unordered_set<uint64_t> intersected;
                for (uint64_t eid : partition_filter) {
                    if (spatial_results.find(eid) != spatial_results.end()) {
                        intersected.insert(eid);
                    }
                }
                partition_filter = intersected;
            } else {
                partition_filter = spatial_results;
                use_partition_filter = true;
            }
        }

        // If any query parsing failed with an error, report it
        if (!partition_error.empty()) {
            char error[512];
            snprintf(error, sizeof(error), "{\"error\": \"Range query error: %s\"}\n", partition_error.c_str());
            send(client_socket, error, strlen(error), 0);

            // Clean up signal handler
            error_handler_set = false;
            sigaction(SIGSEGV, &old_sa, nullptr);
            return;
        }

        // Check if this is a pure partition query
        bool is_pure_partition = false;
        if (use_partition_filter) {
            std::string test_query = remove_partition_queries(query_str);
            printf("DEBUG: After removing partition queries: '%s' (empty=%d)\n",
                   test_query.c_str(), test_query.empty());
            if (test_query.empty()) {
                is_pure_partition = true;
            }
        }

        if (is_pure_partition) {
            // Build JSON response from partition results
            printf("DEBUG: Pure partition query, %zu entities in filter\n", partition_filter.size());
            for (uint64_t eid : partition_filter) {
                printf("  - Entity ID in partition_filter: %lu\n", eid);
            }

            std::vector<char> response_storage(QUERY_RESPONSE_CAPACITY);
            char* response = response_storage.data();
            int offset = 0;
            int remaining = response_storage.size() - offset;
            offset += snprintf(response + offset, remaining, "{\"results\": [\n");

            int count = 0;
            for (uint64_t eid : partition_filter) {
                flecs::entity e = g_world->entity(eid);

                // Skip flecs internal entities (metadata, modules, etc.)
                if (is_internal_entity(e)) {
                    continue;
                }

                if (offset >= response_storage.size() - 1000) break;

                if (count > 0) {
                    remaining = response_storage.size() - offset;
                    offset += snprintf(response + offset, remaining, ",\n");
                }

                // Use flecs reflection to serialize entity with all components
                remaining = response_storage.size() - offset;
                offset += snprintf(response + offset, remaining, "  %s", entity_json_with_id(e).c_str());
                count++;
            }

            offset += snprintf(response + offset, response_storage.size() - offset, "\n], \"count\": %d}\n", count);
            send(client_socket, response, strlen(response), 0);

            // Clean up signal handler
            error_handler_set = false;
            sigaction(SIGSEGV, &old_sa, nullptr);
            return;
        }

        // Parse and execute the flecs query
        std::string flecs_query_str = remove_partition_queries(query_str);

        // A name this world has never heard of yields nothing, rather than
        // failing the whole query.
        std::string unknown = first_unknown_identifier(flecs_query_str);
        if (!unknown.empty()) {
            char empty_response[256];
            snprintf(empty_response, sizeof(empty_response),
                     "{\"results\": [], \"count\": 0, \"unknown\": \"%s\"}\n",
                     unknown.c_str());
            send(client_socket, empty_response, strlen(empty_response), 0);
            error_handler_set = false;
            sigaction(SIGSEGV, &old_sa, nullptr);
            return;
        }

        // Skip if the cleaned query is empty
        if (flecs_query_str.empty()) {
            const char* error = "{\"error\": \"No flecs query components specified\"}\n";
            send(client_socket, error, strlen(error), 0);
            error_handler_set = false;
            sigaction(SIGSEGV, &old_sa, nullptr);
            return;
        }

        auto query = g_world->query_builder()
            .expr(flecs_query_str.c_str())
            .build();

        // Build JSON response
        std::vector<char> response_storage(QUERY_RESPONSE_CAPACITY);
            char* response = response_storage.data();
        int offset = 0;
        int remaining = response_storage.size() - offset;
        offset += snprintf(response + offset, remaining, "{\"results\": [\n");

        int count = 0;
        query.each([&](flecs::entity e) {
            // Apply partition filter if present
            if (use_partition_filter && partition_filter.find(e.id()) == partition_filter.end()) {
                return;
            }

            // Skip flecs internal entities (metadata, modules, etc.)
            if (is_internal_entity(e)) {
                return;
            }

            // Safety check for buffer overflow
            if (offset >= response_storage.size() - 1000) {
                return;
            }

            if (count > 0) {
                remaining = response_storage.size() - offset;
                offset += snprintf(response + offset, remaining, ",\n");
            }

            // Use flecs reflection to serialize entity with all components
            remaining = response_storage.size() - offset;
            offset += snprintf(response + offset, remaining, "  %s", entity_json_with_id(e).c_str());
            count++;
        });

        offset += snprintf(response + offset, response_storage.size() - offset, "\n], \"count\": %d}\n", count);

        send(client_socket, response, strlen(response), 0);

        // Clean up signal handler
        error_handler_set = false;
        sigaction(SIGSEGV, &old_sa, nullptr);
    } catch (const std::exception& e) {
        char error[512];
        snprintf(error, sizeof(error), "{\"error\": \"Query parsing failed: %s\"}\n", e.what());
        send(client_socket, error, strlen(error), 0);
        printf("Query error: %s\n", e.what());

        // Clean up signal handler
        error_handler_set = false;
        sigaction(SIGSEGV, &old_sa, nullptr);
    } catch (...) {
        const char* error = "{\"error\": \"Unknown error occurred while processing query\"}\n";
        send(client_socket, error, strlen(error), 0);
        printf("Unknown error occurred while processing query\n");

        // Clean up signal handler
        error_handler_set = false;
        sigaction(SIGSEGV, &old_sa, nullptr);
    }
}

// Thread function for handling client connections
void* client_handler(void* arg) {
    int client_socket = *(int*)arg;
    free(arg);

    char buffer[1024];
    ssize_t bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);

    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';

        // Remove trailing newline if present
        if (buffer[bytes_read - 1] == '\n') {
            buffer[bytes_read - 1] = '\0';
        }

        // An empty request means "everything". `_` is flecs' Any wildcard, which
        // matches each entity exactly once -- unlike `*`, which yields an entity
        // once per component it holds.
        const char* query = buffer;
        while (*query == ' ' || *query == '\t') query++;
        if (*query == '\0') query = "_";

        if (handle_create(client_socket, query)) {
            close(client_socket);
            return nullptr;
        }

        printf("Received query: %s\n", query);
        handle_query(client_socket, query);
    }

    close(client_socket);
    return nullptr;
}

// Socket server thread
void* socket_server_thread(void* arg) {
    int port = *(int*)arg;
    free(arg);

    int server_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Socket creation failed");
        return nullptr;
    }

    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_socket);
        return nullptr;
    }

    if (listen(server_socket, 5) < 0) {
        perror("Listen failed");
        close(server_socket);
        return nullptr;
    }

    printf("Server listening on port %d...\n", port);

    while (true) {
        int* client_socket = (int*)malloc(sizeof(int));
        *client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);

        if (*client_socket < 0) {
            perror("Accept failed");
            free(client_socket);
            continue;
        }

        printf("Client connected\n");

        pthread_t thread;
        pthread_create(&thread, nullptr, client_handler, client_socket);
        pthread_detach(thread);
    }

    close(server_socket);
    return nullptr;
}

// Start the server
void start_server(int port) {
    int* port_arg = (int*)malloc(sizeof(int));
    *port_arg = port;

    pthread_t server_thread;
    pthread_create(&server_thread, nullptr, socket_server_thread, port_arg);
    pthread_detach(server_thread);
}

} // namespace query_server
