#include <tradewinds.h>
#include <sys/prctl.h>

namespace tradewinds {

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
            server.socket->connect(server.addr);
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
            auto result = client.socket->send(request, zmq::send_flags::dontwait);
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
            auto res = client.socket->recv(reply, zmq::recv_flags::dontwait);
            if (res) {
                msgpack::object_handle oh = msgpack::unpack((const char*)reply.data(), reply.size());
                msgpack::object obj = oh.get();
                auto res_map = obj.as<std::map<std::string, msgpack::object>>();
                onResponse.response_function(res_map);
                e.remove<AwaitResponse>();
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