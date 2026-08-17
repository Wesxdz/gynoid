// Tradewinds is a module for flecs that supports process spawning and interactivity with ZeroMQ
// through data oriented component observers and callbacks

// It is mainly designed for integrating many IPC Python clients and servers with a monolithic flecs application

#ifndef TRADEWINDS_H
#define TRADEWINDS_H

#include <flecs.h>
#include <zmq.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <functional>
#include <map>

#include <msgpack.hpp>

#ifdef __cplusplus
extern "C" {
#endif

namespace tradewinds {

struct ZMQContext { 
    std::shared_ptr<zmq::context_t> ctx; 
};

struct ZMQClient {
    std::string addr;
    zmq::socket_type type;
    std::unique_ptr<zmq::socket_t> socket;
};
struct ZMQServer {
    std::string addr;
    zmq::socket_type type;
    std::unique_ptr<zmq::socket_t> socket;
};

struct AwaitResponse
{
    std::function<void(std::map<std::string, msgpack::object>&)> response_function;
};

struct SendMapRequest 
{
    std::map<std::string, std::string> data;
};

// A stream a client listens to rather than asks for. A subscription has no
// reply and no reservation: the publisher talks while it works, messages arrive
// whenever they arrive, and every one is delivered to the handler as its topic
// and its payload. Used for a solver reporting progress it would be useless to
// have to ask for.
struct AwaitPublished
{
    std::function<void(const std::string& topic, const std::string& payload)> on_message;
};

struct SpawnRequest {
    std::string command;
    std::vector<std::string> args;
    std::string directory;
};

struct LinuxProcess
{
    pid_t pid;
    int stdout_fd = -1;
};

struct module {
    module(flecs::world& world);
};

// In-flight reservation for a REQ client, visible IMMEDIATELY -- unlike the
// has<AwaitResponse>() guard, which is blind within a frame because component
// sets are deferred until the merge. Two systems that both pass the deferred
// guard in one frame each fire the send observer, and the second send on a
// REQ socket is a fatal zmq EFSM. Every sender sharing a client must guard
// with try_reserve instead of has<AwaitResponse>(); the receive system
// releases the reservation when the reply lands, *before* invoking the
// handler, so a handler that chains a follow-up request can re-reserve.
bool try_reserve(flecs::entity client);
void release(flecs::entity client);

}

#ifdef __cplusplus
}
#endif

#endif

