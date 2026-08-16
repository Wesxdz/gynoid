#include "server_registry.h"

// Same GL prelude as main.cpp: components.h assumes the mel_spec glad is the
// one GL header, and GLFW must not drag in the system's.
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "components.h"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <msgpack.hpp>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "tradewinds.h"

using json = nlohmann::json;
using tradewinds::LinuxProcess;
using tradewinds::SpawnRequest;
using tradewinds::ZMQContext;

namespace servers {
namespace {

// Where the config lives, relative to the working directory Thornfield runs
// from (build/, which is why the script paths in it start with ../).
constexpr const char* kConfigPath = "../assets/config/servers.json";

// How long a freshly spawned server reads as Starting before the process table
// alone is allowed to call it Running. This is not a readiness measurement --
// that only arrives over the announce socket.
constexpr double kWarmupSeconds = 2.5;

// Liveness polling period. Each tick walks /proc once for every server we do
// not own, so a second of latency is a fair trade for not doing it per frame.
constexpr float kMonitorInterval = 1.0f;

// The registry entities live under one parent, so a lookup by id is scoped
// rather than a scan, and so they read clearly in the flecs explorer.
constexpr const char* kScope = "Servers";

std::string g_conda_root = "~/miniconda3";

// Thornfield binds, the servers connect -- the direction design/networking.txt
// asks for. One PULL socket for every server, rather than a socket each.
std::unique_ptr<zmq::socket_t> g_announce;

std::string expandHome(const std::string& path)
{
    if (path.empty() || path[0] != '~') return path;
    const char* home = getenv("HOME");
    return home ? std::string(home) + path.substr(1) : path;
}

double worldTime(const flecs::world& world)
{
    return world.get_info()->world_time_total;
}

// ---- Process table ---------------------------------------------------------

// Every process' command line, keyed by pid. /proc stores argv NUL separated;
// joining with spaces is what makes a "python3 ../scripts/pos.py" needle match.
// Built at most once per monitor tick and shared by every server.
std::vector<std::pair<int, std::string>> snapshotProcesses()
{
    std::vector<std::pair<int, std::string>> table;

    DIR* proc = opendir("/proc");
    if (!proc) return table;

    const pid_t self = getpid();
    table.reserve(256);

    while (dirent* entry = readdir(proc)) {
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
        const int pid = atoi(entry->d_name);
        if (pid <= 0 || pid == self) continue;

        std::ifstream file(std::string("/proc/") + entry->d_name + "/cmdline", std::ios::binary);
        if (!file) continue;

        std::string cmd((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (cmd.empty()) continue;   // kernel thread
        for (char& c : cmd) {
            if (c == '\0') c = ' ';
        }
        table.emplace_back(pid, std::move(cmd));
    }

    closedir(proc);
    return table;
}

int matchProcess(const std::vector<std::pair<int, std::string>>& table, const std::string& needle)
{
    if (needle.empty()) return 0;
    for (const auto& [pid, cmd] : table) {
        if (cmd.find(needle) != std::string::npos) return pid;
    }
    return 0;
}

// ---- Config ----------------------------------------------------------------

ServerDef parseServer(const json& node)
{
    ServerDef def;
    def.id = node.value("id", std::string{});
    def.label = node.value("label", def.id);
    def.icon = node.value("icon", def.id);
    def.description = node.value("description", std::string{});
    def.script = node.value("script", std::string{});
    def.cwd = node.value("cwd", std::string{});
    def.conda_env = node.value("env", std::string{});
    def.launch = node.value("launch", std::string{"direct"});
    def.socket = node.value("socket", std::string{});
    def.match = node.value("match", def.script);
    def.autostart = node.value("autostart", false);

    if (node.contains("args") && node["args"].is_array()) {
        for (const json& arg : node["args"]) def.args.push_back(arg.get<std::string>());
    }
    return def;
}

// The child entity that carries the spawn. Separate from the registry entity
// because tradewinds' ProcessMonitor destructs whatever holds LinuxProcess when
// the process exits, and that must not take the registry entry with it.
flecs::entity processChild(flecs::entity server, bool create)
{
    flecs::entity child = server.lookup("process");
    if (!child && create) child = server.world().entity("process").child_of(server);
    return child;
}

} // namespace

// ---- Public ----------------------------------------------------------------

std::string command_line(const ServerDef& def, const std::string& conda_root)
{
    std::string cmd;
    if (def.conda_env.empty()) {
        cmd = "python3 " + def.script;
    } else if (def.launch == "conda-run") {
        cmd = expandHome(conda_root) + "/bin/conda run -n " + def.conda_env
            + " --no-capture-output python -u " + def.script;
    } else {
        cmd = expandHome(conda_root) + "/envs/" + def.conda_env + "/bin/python -u " + def.script;
    }
    for (const std::string& arg : def.args) cmd += " " + arg;
    return cmd;
}

flecs::entity find(flecs::world& world, const std::string& id)
{
    flecs::entity scope = world.lookup(kScope);
    return scope ? scope.lookup(id.c_str()) : flecs::entity();
}

void report_ready(flecs::world& world, const std::string& id, bool ready)
{
    flecs::entity server = find(world, id);
    if (!server) {
        std::cerr << "[servers] readiness report from unknown server '" << id << "'" << std::endl;
        return;
    }

    ServerState& state = server.ensure<ServerState>();
    if (state.ready != ready) {
        std::cout << "[servers] " << id << " reports " << (ready ? "READY" : "not ready") << std::endl;
    }
    state.ready = ready;
    state.last_report = worldTime(world);

    // Only an observed-running server can be Ready; a report that arrives
    // before the monitor has seen the process is kept and applied next tick.
    if (ready && state.status == ServerStatus::Running) state.status = ServerStatus::Ready;
    if (!ready && state.status == ServerStatus::Ready) state.status = ServerStatus::Running;
}

void start(flecs::entity server)
{
    if (!server.is_valid() || !server.is_alive()) return;

    const ServerDef* def = server.try_get<ServerDef>();
    ServerState* state = server.try_get_mut<ServerState>();
    if (!def || !state) return;

    if (state->status != ServerStatus::Offline) {
        std::cout << "[servers] " << def->id << " already running (pid " << state->pid << ")" << std::endl;
        return;
    }

    std::string program;
    std::vector<std::string> args;
    if (def->conda_env.empty()) {
        program = "python3";
        args = {def->script};
    } else if (def->launch == "conda-run") {
        program = expandHome(g_conda_root) + "/bin/conda";
        args = {"run", "-n", def->conda_env, "--no-capture-output", "python", "-u", def->script};
    } else {
        program = expandHome(g_conda_root) + "/envs/" + def->conda_env + "/bin/python";
        args = {"-u", def->script};
    }
    args.insert(args.end(), def->args.begin(), def->args.end());

    std::cout << "[servers] starting " << def->id << ": "
              << command_line(*def, g_conda_root) << std::endl;

    // The pid is deliberately not read back here. A click arrives inside the
    // deferred window around glfwPollEvents(), so tradewinds' OnSet observer
    // has not forked yet; the monitor picks the pid up from the child.
    processChild(server, true).set<SpawnRequest>({program, args, def->cwd});

    state->owned = true;
    state->pid = 0;
    state->ready = false;
    state->status = ServerStatus::Starting;
    state->started_at = worldTime(server.world());
}

void stop(flecs::entity server)
{
    if (!server.is_valid() || !server.is_alive()) return;

    const ServerDef* def = server.try_get<ServerDef>();
    ServerState* state = server.try_get_mut<ServerState>();
    if (!def || !state) return;

    if (!state->owned) {
        if (state->status != ServerStatus::Offline) {
            std::cout << "[servers] " << def->id << " (pid " << state->pid
                      << ") was not started by Thornfield; leaving it alone" << std::endl;
        }
        return;
    }

    std::cout << "[servers] stopping " << def->id << " (pid " << state->pid << ")" << std::endl;

    // Destructing the child fires tradewinds' OnRemove hook, which SIGTERMs the
    // whole process group -- a conda-run wrapper and the python beneath it.
    if (flecs::entity child = processChild(server, false)) child.destruct();

    *state = {};
}

void shutdown()
{
    if (!g_announce) return;
    // linger 0: nothing is waiting on an inbound-only socket, and a non-zero
    // linger is the other way zmq_ctx_term() stalls a shutdown.
    g_announce->set(zmq::sockopt::linger, 0);
    g_announce->close();
    g_announce.reset();
}

void toggle(flecs::entity server)
{
    if (!server.is_valid() || !server.is_alive()) return;
    const ServerState* state = server.try_get<ServerState>();
    if (!state) return;
    if (state->status == ServerStatus::Offline) start(server); else stop(server);
}

module::module(flecs::world& world)
{
    world.module<module>();
    world.component<ServerDef>();
    world.component<ServerState>();
    world.component<ServerOf>();

    std::ifstream file(kConfigPath);
    if (!file) {
        std::cerr << "[servers] no config at " << kConfigPath
                  << "; the Peach Core panel will be empty" << std::endl;
        return;
    }

    json config;
    try {
        // ignore_comments: the config documents itself inline. See servers.json.
        config = json::parse(file, nullptr, true, true);
    } catch (const json::parse_error& e) {
        std::cerr << "[servers] " << kConfigPath << " is not valid JSON: " << e.what() << std::endl;
        return;
    }

    g_conda_root = config.value("conda_root", g_conda_root);

    // A module constructor runs with the scope set to the module entity, so
    // creating the registry here unqualified would bury it at
    // ::servers::module::Servers and every lookup from root would miss it.
    // Same dance flecs' own do_import does around a module's construction.
    const flecs::entity_t prev_scope = ecs_set_scope(world, 0);

    flecs::entity scope = world.entity(kScope).add(flecs::OrderedChildren);

    // Build the entries inside the registry scope, so each id names a child
    // directly rather than being created at root and reparented.
    ecs_set_scope(world, scope);
    int count = 0;
    for (const json& node : config.value("servers", json::array())) {
        ServerDef def = parseServer(node);
        if (def.id.empty()) {
            std::cerr << "[servers] skipping an entry with no id" << std::endl;
            continue;
        }
        world.entity(def.id.c_str())
             .set<ServerDef>(def)
             .set<ServerState>({});
        ++count;
    }
    ecs_set_scope(world, prev_scope);

    std::cout << "[servers] loaded " << count << " server definitions at "
              << scope.path().c_str() << std::endl;

    // ---- Readiness inbox ---------------------------------------------------
    // Bound once, here, so a server can announce whenever it finishes loading
    // -- including before Thornfield has noticed its process. Servers push a
    // msgpack map: {"id": "<config id>", "status": "READY"}.
    if (const ZMQContext* ctx = world.try_get<ZMQContext>()) {
        try {
            g_announce = std::make_unique<zmq::socket_t>(*ctx->ctx, zmq::socket_type::pull);
            g_announce->bind(kAnnounceSocket);
            std::cout << "[servers] readiness inbox on " << kAnnounceSocket << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "[servers] could not bind " << kAnnounceSocket << ": " << e.what() << std::endl;
            g_announce.reset();
        }
    } else {
        std::cerr << "[servers] no ZMQContext yet; readiness reports disabled. "
                     "Import servers::module after world.set<ZMQContext>()." << std::endl;
    }

    world.system("ServerReadinessInbox")
         .run([](flecs::iter& it) {
             if (!g_announce) return;
             flecs::world world = it.world();

             // Drain: several servers can finish loading in the same frame.
             for (;;) {
                 zmq::message_t msg;
                 zmq::recv_result_t got;
                 try {
                     got = g_announce->recv(msg, zmq::recv_flags::dontwait);
                 } catch (const zmq::error_t&) {
                     return;
                 }
                 if (!got) return;

                 try {
                     msgpack::object_handle oh = msgpack::unpack((const char*)msg.data(), msg.size());
                     auto map = oh.get().as<std::map<std::string, std::string>>();
                     const std::string id = map.count("id") ? map["id"] : "";
                     const std::string status = map.count("status") ? map["status"] : "READY";
                     if (!id.empty()) report_ready(world, id, status == "READY");
                 } catch (const std::exception& e) {
                     std::cerr << "[servers] unreadable readiness report: " << e.what() << std::endl;
                 }
             }
         });

    // ---- Liveness ----------------------------------------------------------
    world.system<ServerDef, ServerState>("ServerMonitor")
         .interval(kMonitorInterval)
         .run([](flecs::iter& it) {
             const double now = worldTime(it.world());

             // At most one /proc walk per tick, built lazily so a session where
             // Thornfield owns every server never walks it at all.
             std::vector<std::pair<int, std::string>> table;
             bool scanned = false;

             while (it.next()) {
                 auto defs = it.field<ServerDef>(0);
                 auto states = it.field<ServerState>(1);

                 for (size_t i = 0; i < it.count(); ++i) {
                     const ServerDef& def = defs[i];
                     ServerState& state = states[i];

                     if (state.owned) {
                         // tradewinds destructs the process child when the
                         // process exits, so its absence is the death signal.
                         flecs::entity server = it.entity(i);
                         flecs::entity child = server.lookup("process");
                         const LinuxProcess* proc = child ? child.try_get<LinuxProcess>() : nullptr;

                         if (!proc || proc->pid <= 0) {
                             std::cout << "[servers] " << def.id << " exited" << std::endl;
                             state = {};
                             continue;
                         }

                         state.pid = proc->pid;
                         if (state.ready) {
                             state.status = ServerStatus::Ready;
                         } else if (now - state.started_at < kWarmupSeconds) {
                             state.status = ServerStatus::Starting;
                         } else {
                             state.status = ServerStatus::Running;
                         }
                         continue;
                     }

                     // Not ours, so the process table is the only witness.
                     if (!scanned) { table = snapshotProcesses(); scanned = true; }

                     const int foreign = matchProcess(table, def.match);
                     if (foreign) {
                         if (state.pid != foreign) {
                             std::cout << "[servers] " << def.id
                                       << " running externally (pid " << foreign << ")" << std::endl;
                         }
                         state.pid = foreign;
                         state.status = state.ready ? ServerStatus::Ready : ServerStatus::Running;
                     } else if (state.status != ServerStatus::Offline) {
                         state = {};
                     }
                 }
             }
         });

    // Autostart last: a server the user already had running is adopted rather
    // than started a second time.
    const auto table = snapshotProcesses();
    scope.children([&table](flecs::entity server) {
        const ServerDef* def = server.try_get<ServerDef>();
        if (!def || !def->autostart) return;

        if (const int existing = matchProcess(table, def->match)) {
            std::cout << "[servers] " << def->id << " already running externally (pid "
                      << existing << "); adopting rather than starting" << std::endl;
            ServerState& state = server.ensure<ServerState>();
            state.pid = existing;
            state.owned = false;
            state.status = ServerStatus::Running;
            return;
        }
        start(server);
    });
}

} // namespace servers
