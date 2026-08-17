#include <tradewinds.h>
#include <sys/prctl.h>

#include <unordered_set>

namespace tradewinds {

// Clients with a request in flight. A plain set rather than a component so
// membership changes are visible the instant they happen -- see the header.
static std::unordered_set<ecs_entity_t> g_inflight;

bool try_reserve(flecs::entity client)
{
    if (!client.is_valid()) return false;
    return g_inflight.insert(client.id()).second;
}

void release(flecs::entity client)
{
    g_inflight.erase(client.id());
}

module::module(flecs::world& ecs) 
{
    ecs.module<module>();

    ecs.component<ZMQContext>();
    ecs.component<ZMQServer>();
    ecs.component<SpawnRequest>();
    ecs.component<LinuxProcess>();

    ecs.observer<LinuxProcess>()
        .event(flecs::OnRemove)
        .each([](flecs::entity e, LinuxProcess& p) {
            if (p.pid > 0) {
                kill(-p.pid, SIGTERM);
                waitpid(p.pid, nullptr, WNOHANG); // Non-blocking cleanup
                std::cout << "Process group " << p.pid << " cleaned up." << std::endl;
            }
        });

    ecs.observer<SpawnRequest>()
        .event(flecs::OnSet)
        .each([](flecs::entity e, SpawnRequest& req) {
            pid_t pid = fork();

            if (pid == -1) {
                // e.set<ProcessError>({errno});
                e.remove<SpawnRequest>();
                return;
            }

            if (pid == 0) {

                prctl(PR_SET_PDEATHSIG, SIGTERM);
                setpgid(0, 0); // Use process group so conda properly destroys python child
                
                if (!req.directory.empty()) {
                    if (chdir(req.directory.c_str()) == -1) {
                        perror("chdir failed");
                        _exit(1);
                    }
                }

                std::vector<char*> c_args;
                c_args.push_back(const_cast<char*>(req.command.c_str()));
                for (auto& a : req.args) c_args.push_back(const_cast<char*>(a.c_str()));
                c_args.push_back(nullptr);

                execvp(c_args[0], c_args.data());
                _exit(1); 
            } 

            // Parent: Successfully spawned
            e.set<LinuxProcess>({pid});
            std::cout << "Spawned " << req.command << " with PID " << pid << std::endl;
        });

    ecs.observer<ZMQServer>()
        .event(flecs::OnSet)
        .each([](flecs::entity e, ZMQServer& server) {
            // Prevent re-initialization if the socket already exists
            if (server.socket) return;

            auto ctx_comp = e.world().ensure<ZMQContext>();

            server.socket = std::make_unique<zmq::socket_t>(*(ctx_comp.ctx), server.type);
            server.socket->bind(server.addr);
        });

    ecs.observer<ZMQClient>()
        .event(flecs::OnSet)
        .each([](flecs::entity e, ZMQClient& server) {
            // Prevent re-initialization if the socket already exists
            if (server.socket) return;

            auto ctx_comp = e.world().ensure<ZMQContext>();

            server.socket = std::make_unique<zmq::socket_t>(*(ctx_comp.ctx), server.type);

            // A SUB socket that has not subscribed receives nothing at all --
            // the filter defaults to matching no topic, so the socket connects,
            // reports healthy, and stays silent forever. An empty prefix takes
            // everything and lets the handler sort topics out.
            if (server.type == zmq::socket_type::sub)
                server.socket->set(zmq::sockopt::subscribe, "");

            server.socket->connect(server.addr);
        });

    // Drains everything a subscription has waiting, rather than one message per
    // frame: a solver emits progress far faster than the editor draws, and a
    // one-per-frame drain would fall further behind the longer it ran.
    ecs.system<AwaitPublished, ZMQClient>()
        .each([](flecs::entity e, AwaitPublished& awaiting, ZMQClient& client)
        {
            if (!client.socket || !awaiting.on_message) return;
            for (int drained = 0; drained < 256; drained++) {
                zmq::message_t topic;
                zmq::recv_result_t got;
                try {
                    got = client.socket->recv(topic, zmq::recv_flags::dontwait);
                } catch (const zmq::error_t& exc) {
                    std::cerr << "[tradewinds] recv on " << e.name()
                              << " failed: " << exc.what() << std::endl;
                    return;
                }
                if (!got) return;

                // Multipart by convention: [topic, payload]. A publisher that
                // sends only one frame still reports, with an empty payload.
                std::string payload;
                if (client.socket->get(zmq::sockopt::rcvmore)) {
                    zmq::message_t body;
                    try {
                        if (client.socket->recv(body, zmq::recv_flags::none))
                            payload.assign((const char*)body.data(), body.size());
                    } catch (const zmq::error_t&) { return; }
                }

                try {
                    awaiting.on_message(
                        std::string((const char*)topic.data(), topic.size()), payload);
                } catch (const std::exception& exc) {
                    std::cerr << "[tradewinds] published handler threw: "
                              << exc.what() << std::endl;
                }
            }
        });

    ecs.observer<SendMapRequest, ZMQClient>()
        .event(flecs::OnSet)
        .each([](flecs::entity e, SendMapRequest& req, ZMQClient& client) {
            msgpack::sbuffer sbuf;
            msgpack::packer<msgpack::sbuffer> pk(&sbuf);

            pk.pack_map(req.data.size());
            for (const auto& [key, value] : req.data) {
                pk.pack(key);
                pk.pack(value);
            }

            zmq::message_t request(sbuf.data(), sbuf.size());
            // A REQ socket in the wrong state throws EFSM, and an exception
            // escaping an observer terminates the process -- a dropped
            // request must never take Thornfield down with it.
            try {
                client.socket->send(request, zmq::send_flags::dontwait);
            } catch (const zmq::error_t& exc) {
                std::cerr << "[tradewinds] send on " << e.name()
                          << " failed: " << exc.what() << std::endl;
            }
            e.remove<SendMapRequest>();
            // TODO: AwaitRequestResult
            // if (result) {
            //     std::cout << "Request sent successfully." << std::endl;
            // }
        });

    ecs.system<AwaitResponse, ZMQClient>()
        .without<SendMapRequest>()
        .each([](flecs::entity e, AwaitResponse& onResponse, ZMQClient& client)
        {
            zmq::message_t reply;
            zmq::recv_result_t res;
            try {
                res = client.socket->recv(reply, zmq::recv_flags::dontwait);
            } catch (const zmq::error_t& exc) {
                std::cerr << "[tradewinds] recv on " << e.name()
                          << " failed: " << exc.what() << std::endl;
                release(e);
                e.remove<AwaitResponse>();
                return;
            }
            if (res) {
                // The wire is free again the moment the reply is in hand --
                // released before the handler so a chained request can
                // re-reserve the client.
                release(e);
                // Queue the removal before invoking the handler: inside a
                // system these ops are deferred, so at the merge they apply in
                // order. Remove-then-handler means a handler that chains a new
                // request (setting a fresh AwaitResponse) is not wiped by the
                // removal of the one it is replacing. Removed on any reply, so
                // a bad one does not wedge the client either.
                e.remove<AwaitResponse>();
                // A malformed reply, or a handler that throws, must not abort
                // Thornfield: an exception escaping here unwinds out of a system
                // callback into flecs and terminates with no message at all,
                // which is a miserable thing to debug.
                try {
                    msgpack::object_handle oh = msgpack::unpack((const char*)reply.data(), reply.size());
                    msgpack::object obj = oh.get();
                    auto res_map = obj.as<std::map<std::string, msgpack::object>>();
                    onResponse.response_function(res_map);
                } catch (const std::exception& exc) {
                    std::cerr << "[tradewinds] response handler for " << e.name()
                              << " failed: " << exc.what() << std::endl;
                }
            }
        });

    ecs.system<LinuxProcess>("ProcessMonitor")
        .each([](flecs::entity e, LinuxProcess& p) {
            int status;
            pid_t result = waitpid(p.pid, &status, WNOHANG);
            
            if (result > 0) {
                std::cout << "Process " << p.pid << " exited. Removing entity." << std::endl;
                e.destruct(); 
            }
        });

    ecs.system<ZMQServer>("ZMQListenSystem")
        .each([](flecs::entity e, ZMQServer& server) {
            // Only attempt to receive if it's a REP (Reply) socket for this example
            if (server.type == zmq::socket_type::rep) {
                zmq::message_t request;
                
                // Use dontwait to ensure the ECS loop remains high-performance
                auto result = server.socket->recv(request, zmq::recv_flags::dontwait);

                if (result) {
                    std::cout << "[" << e.name() << "] Received: " << request.to_string() << std::endl;

                    // Simple Echo/Reply logic
                    std::string reply_str = "World";
                    zmq::message_t reply(reply_str.size());
                    memcpy(reply.data(), reply_str.data(), reply_str.size());
                    
                    server.socket->send(reply, zmq::send_flags::none);
                }
            }
        });

}

}