#pragma once

// The supporting processes behind the gynoid -- the ones the Peach Core panel
// shows as icons -- loaded from assets/config/servers.json.
//
// A registry entity per server holds its ServerDef and ServerState and lives
// for the whole session, independent of whether a panel is open. Spawning goes
// through tradewinds, on a *child* of the registry entity: tradewinds'
// ProcessMonitor destructs the entity holding LinuxProcess when the process
// exits, and that must not take the registry entry with it.
//
// Stopping is a process-group SIGTERM (tradewinds' OnRemove hook), which is why
// a conda-run wrapper tears its python child down properly.

#include <string>

#include <flecs.h>

// Forward-declared rather than including components.h, which silently depends
// on the GL/GLFW prelude being included ahead of it. Same reasoning as
// panels/registry.h -- a header should be includable on its own.
struct ServerDef;

namespace servers {

// Where a server announces that it has finished loading. Thornfield binds this
// PULL socket -- one inbox for every server, in the direction
// design/networking.txt asks for -- and a server connects a PUSH socket to it
// and sends a msgpack map when it is ready to answer:
//
//     import zmq, msgpack
//     announce = zmq.Context.instance().socket(zmq.PUSH)
//     announce.connect("ipc:///tmp/thornfield_server_status")
//     announce.send(msgpack.packb({"id": "pos", "status": "READY"}))
//
// "id" is the server's id in servers.json. Any status other than "READY" drops
// it back to Running. Sending periodically rather than once is what would let a
// wedged server decay out of Ready on its own.
constexpr const char* kAnnounceSocket = "ipc:///tmp/thornfield_server_status";

// Reads the config, creates the registry entities, binds the readiness inbox,
// registers the liveness monitor, and spawns whatever is marked autostart.
// Import after tradewinds::module and after world.set<ZMQContext>().
struct module
{
    explicit module(flecs::world& world);
};

// Record a readiness report. Called by the announce inbox, and usable directly
// by any call site that learns a server's state some other way -- the existing
// REQ clients already get a "status" field back in every reply.
void report_ready(flecs::world& world, const std::string& id, bool ready);

// Close the announce socket. Must run before the world is destroyed, because
// the world owns the ZMQContext and zmq_ctx_term() blocks until every socket on
// that context is closed -- the announce socket is not a component, so nothing
// else would close it, and Thornfield would hang on exit.
void shutdown();

// Registry entity for a config id, or an invalid entity when there is none.
flecs::entity find(flecs::world& world, const std::string& id);

// Spawn the server. No-op when it is already running, or when Thornfield can
// see a foreign process already serving it.
void start(flecs::entity server);

// SIGTERM the process group Thornfield started. No-op for a foreign process --
// we did not start it, so we do not kill it.
void stop(flecs::entity server);

// Start when offline, stop when we own it. What the icon click does.
void toggle(flecs::entity server);

// The command line `start` would run, for display and for logging.
std::string command_line(const ServerDef& def, const std::string& conda_root);

} // namespace servers
