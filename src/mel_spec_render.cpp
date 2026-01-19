#include "mel_spec_render.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <variant>
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>

// Forward declare GL2 function from nanovg
extern "C" {
int nvglCreateImageFromHandleGL2(NVGcontext* ctx, GLuint textureId, int w, int h, int flags);
}

// Forward declarations for components defined in main.cpp
struct Graphics {
    NVGcontext* vg;
};

struct Position {
    float x, y;
};

struct Local { };  // Tag for local position

struct ImageRenderable {
    int imageHandle;
    float width, height;
    float alpha;
    int spriteX = 0;
    int spriteY = 0;
    int spriteW = 0;
    int spriteH = 0;
    uint32_t tintColor = 0;
    float tintStrength = 0.0f;
};

struct RectRenderable {
    float width, height;
    uint32_t color;
};

struct TextRenderable {
    std::string text;
    std::string fontFace;
    float fontSize;
    uint32_t color;
    int alignment;
};

enum class RenderType {
    Rect,
    Text,
    Image
};

struct RenderCommand {
    Position pos;
    std::variant<RectRenderable, TextRenderable, ImageRenderable> renderData;
    RenderType type;
    int zIndex;
};

struct RenderQueue {
    std::vector<RenderCommand> commands;

    void clear() {
        commands.clear();
    }

    void sort() {
        std::sort(commands.begin(), commands.end(),
            [](const RenderCommand& a, const RenderCommand& b) {
                return a.zIndex < b.zIndex;
            });
    }

    void addRectCommand(const Position& pos, const RectRenderable& rect, int zIndex) {
        commands.push_back({pos, rect, RenderType::Rect, zIndex});
    }

    void addTextCommand(const Position& pos, const TextRenderable& text, int zIndex) {
        commands.push_back({pos, text, RenderType::Text, zIndex});
    }

    void addImageCommand(const Position& pos, const ImageRenderable& img, int zIndex) {
        commands.push_back({pos, img, RenderType::Image, zIndex});
    }
};

// IPC audio stream state
struct IPCAudioState {
    int socket_fd;
    int listen_fd;
    pid_t child_pid;
    std::string socket_path;

    NVGcontext* vg;
    GLuint glTexture;
    int texWidth;
    int texHeight;
    int nvgImageHandle;

    unsigned char* imageBuffer;
    int imageBufferSize;

    // Receive buffer for handling partial reads
    unsigned char* recvBuffer;
    int recvBufferSize;
    int recvBufferUsed;

    // Batching: track if imageBuffer was modified this frame
    bool needsUpload;

    // Fill tracking: how many columns have data (for positioning)
    int columnsFilled;    // 0 to texWidth
    bool isFullyFilled;   // true once we start receiving scroll commands

    // Time-based scroll synchronization
    double scrollStartTime;       // Wall clock time when scroll mode started
    size_t totalScrollCommands;   // Number of scroll commands since start
};

static IPCAudioState g_micState = {};
static IPCAudioState g_sysAudioState = {};

// Create listening socket
static int create_listen_socket(const char* socket_path) {
    unlink(socket_path);  // Remove old socket if exists

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(fd);
        return -1;
    }

    if (listen(fd, 1) < 0) {
        perror("listen");
        close(fd);
        return -1;
    }

    // Make non-blocking
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    return fd;
}

// Spawn audio client child process
static pid_t spawn_audio_client(const char* socket_path, int use_system_audio) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        // Child process
        char use_sys_str[2];
        snprintf(use_sys_str, sizeof(use_sys_str), "%d", use_system_audio);

        execl("./audio_stream_client", "audio_stream_client",
              socket_path, use_sys_str, (char*)NULL);

        // If exec fails
        perror("execl");
        exit(1);
    }

    return pid;
}

// Try to accept connection (non-blocking)
static void try_accept_connection(IPCAudioState* state) {
    if (state->socket_fd >= 0) return;  // Already connected
    if (state->listen_fd < 0) return;

    int client_fd = accept(state->listen_fd, NULL, NULL);
    if (client_fd < 0) {
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            perror("accept");
        }
        return;
    }

    // Make client socket non-blocking
    int flags = fcntl(client_fd, F_GETFL, 0);
    fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

    // Increase socket receive buffer to reduce lag during slow frames
    // 256KB buffer can hold ~650 scroll messages (128 height * 3 bytes + 16 header = 400 bytes each)
    int rcvbuf = 256 * 1024;
    if (setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt SO_RCVBUF");
    }

    state->socket_fd = client_fd;
    printf("IPC: Client connected to %s\n", state->socket_path.c_str());
}

// Try to read data from non-blocking socket
static ssize_t try_read(int fd, void* buf, size_t count) {
    ssize_t n = read(fd, buf, count);
    if (n < 0) {
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            return 0;  // Would block, no data available
        }
        return -1;  // Error
    }
    return n;  // Bytes read
}

// Receive and update texture from IPC (delta updates)
static void update_from_ipc(IPCAudioState* state) {
    if (state->socket_fd < 0) return;

    // Ensure image buffer is allocated (for scroll handling)
    int fullImageSize = state->texWidth * state->texHeight * 3;
    if (state->imageBufferSize < fullImageSize) {
        state->imageBuffer = (unsigned char*)realloc(state->imageBuffer, fullImageSize);
        memset(state->imageBuffer, 0, fullImageSize);
        state->imageBufferSize = fullImageSize;
    }

    // Ensure receive buffer is allocated (header + max region data)
    int maxMessageSize = 16 + fullImageSize;
    if (state->recvBufferSize < maxMessageSize * 2) {
        state->recvBuffer = (unsigned char*)realloc(state->recvBuffer, maxMessageSize * 2);
        state->recvBufferSize = maxMessageSize * 2;
    }

    // Drain all pending data from socket to keep mel spec in sync with real-time
    // This ensures scroll rate matches filmstrip even when frame rate drops
    while (true) {
        ssize_t n = try_read(state->socket_fd,
                             state->recvBuffer + state->recvBufferUsed,
                             state->recvBufferSize - state->recvBufferUsed);
        if (n < 0) {
            fprintf(stderr, "IPC: Socket error on %s\n", state->socket_path.c_str());
            close(state->socket_fd);
            state->socket_fd = -1;
            return;
        }
        if (n == 0) break;  // No more data available
        state->recvBufferUsed += n;
    }

    // Process complete messages from buffer
    while (state->recvBufferUsed >= 16) {
        // Read header: [x, y, w, h]
        int* header = (int*)state->recvBuffer;
        int x = header[0];
        int y = header[1];
        int width = header[2];
        int height = header[3];

        // Check for scroll command (x = -1) - legacy, skip it
        if (x == -1) {
            memmove(state->recvBuffer, state->recvBuffer + 16, state->recvBufferUsed - 16);
            state->recvBufferUsed -= 16;
            continue;
        }

        // Handle scroll + new column command (x = -2)
        // Shift imageBuffer left by 1 column, add new column at right
        if (x == -2) {
            int colHeight = height;  // height from header
            int dataSize = colHeight * 3;  // 1 column of RGB data
            int messageSize = 16 + dataSize;

            if (state->recvBufferUsed < messageSize) {
                return;  // Need more data
            }

            unsigned char* colData = state->recvBuffer + 16;

            // Shift imageBuffer left by 1 column
            for (int row = 0; row < state->texHeight; row++) {
                unsigned char* rowStart = state->imageBuffer + row * state->texWidth * 3;
                memmove(rowStart, rowStart + 3, (state->texWidth - 1) * 3);
            }

            // Add new column at rightmost position
            int rightX = state->texWidth - 1;
            for (int row = 0; row < colHeight && row < state->texHeight; row++) {
                int dstIdx = (row * state->texWidth + rightX) * 3;
                int srcIdx = row * 3;
                state->imageBuffer[dstIdx + 0] = colData[srcIdx + 0];
                state->imageBuffer[dstIdx + 1] = colData[srcIdx + 1];
                state->imageBuffer[dstIdx + 2] = colData[srcIdx + 2];
            }

            // Track scroll timing for synchronization
            if (!state->isFullyFilled) {
                // First scroll command - record start time
                state->scrollStartTime = glfwGetTime();
                state->totalScrollCommands = 0;
            }
            state->totalScrollCommands++;

            // Mark as fully filled (receiving scroll commands means texture is full)
            state->isFullyFilled = true;
            state->columnsFilled = state->texWidth;

            // Mark for upload (batched, will upload once per frame)
            state->needsUpload = true;

            memmove(state->recvBuffer, state->recvBuffer + messageSize,
                    state->recvBufferUsed - messageSize);
            state->recvBufferUsed -= messageSize;
            continue;
        }

        // Validate header for normal updates
        if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
            x + width > state->texWidth || y + height > state->texHeight) {
            fprintf(stderr, "IPC: Invalid header x=%d y=%d w=%d h=%d, resetting buffer\n",
                    x, y, width, height);
            state->recvBufferUsed = 0;
            return;
        }

        // Calculate message size
        int dataSize = width * height * 3;
        int messageSize = 16 + dataSize;

        // Check if we have the complete message
        if (state->recvBufferUsed < messageSize) {
            return;  // Need more data
        }

        // Process the complete message
        unsigned char* imageData = state->recvBuffer + 16;

        // Update imageBuffer with the received region
        for (int row = 0; row < height; row++) {
            int dstRowStart = ((y + row) * state->texWidth + x) * 3;
            int srcRowStart = row * width * 3;
            memcpy(state->imageBuffer + dstRowStart, imageData + srcRowStart, width * 3);
        }

        // Track fill progress (rightmost column written + 1)
        if (!state->isFullyFilled) {
            int rightmostCol = x + width;
            if (rightmostCol > state->columnsFilled) {
                state->columnsFilled = rightmostCol;
            }
        }

        // Upload region to GL texture
        glBindTexture(GL_TEXTURE_2D, state->glTexture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height,
                        GL_RGB, GL_UNSIGNED_BYTE, imageData);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "GL error in texture upload: 0x%x\n", err);
        }

        // Remove processed message from buffer
        memmove(state->recvBuffer, state->recvBuffer + messageSize,
                state->recvBufferUsed - messageSize);
        state->recvBufferUsed -= messageSize;
    }
}

// Upload batched texture changes (called once per frame after processing all IPC messages)
static void upload_if_needed(IPCAudioState* state) {
    if (!state->needsUpload || !state->imageBuffer) return;

    glBindTexture(GL_TEXTURE_2D, state->glTexture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, state->texWidth, state->texHeight,
                    GL_RGB, GL_UNSIGNED_BYTE, state->imageBuffer);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "GL error in batched texture upload: 0x%x\n", err);
    }

    state->needsUpload = false;
}

// System to update textures from IPC
static void melSpecUpdateSystem(flecs::entity e, MelSpecRender& melSpec) {
    static int update_count = 0;
    update_count++;

    // Try to accept connections if not yet connected
    try_accept_connection(&g_micState);
    try_accept_connection(&g_sysAudioState);

    // Process all pending IPC messages (may batch multiple scroll updates)
    update_from_ipc(&g_micState);
    update_from_ipc(&g_sysAudioState);

    // Upload batched changes once per frame
    upload_if_needed(&g_micState);
    upload_if_needed(&g_sysAudioState);

    // Update fill progress on the component
    // Determine which IPC state corresponds to this entity's texture
    IPCAudioState* state = nullptr;
    if (melSpec.nvgTextureHandle == g_micState.nvgImageHandle) {
        state = &g_micState;
    } else if (melSpec.nvgTextureHandle == g_sysAudioState.nvgImageHandle) {
        state = &g_sysAudioState;
    }

    if (state && state->texWidth > 0) {
        melSpec.fillProgress = (float)state->columnsFilled / (float)state->texWidth;

        // Calculate time-based render offset for synchronization with filmstrip
        if (state->isFullyFilled && state->scrollStartTime > 0) {
            double now = glfwGetTime();
            double elapsedTime = now - state->scrollStartTime;

            // Expected columns based on wall clock time
            double expectedColumns = elapsedTime * MelSpecRender::COLUMNS_PER_SECOND;

            // Actual columns received
            double actualColumns = (double)state->totalScrollCommands;

            // Lag in columns (positive = mel spec is behind)
            double lagColumns = expectedColumns - actualColumns;

            // Convert to normalized offset (0-1 representing fraction of 24-second window)
            double totalColumnsInWindow = MelSpecRender::COLUMNS_PER_SECOND * MelSpecRender::SCROLL_DURATION;
            float targetOffset = (float)(lagColumns / totalColumnsInWindow);

            // Clamp to reasonable range
            targetOffset = std::max(-0.5f, std::min(0.5f, targetOffset));

            // Smooth interpolation toward target offset (no sudden jumps)
            float smoothingFactor = 0.05f;  // Adjust for smoother/faster response
            melSpec.renderOffset += (targetOffset - melSpec.renderOffset) * smoothingFactor;

            // Copy timing info to component
            melSpec.scrollStartTime = state->scrollStartTime;
            melSpec.totalScrollCommands = state->totalScrollCommands;
        } else {
            melSpec.renderOffset = 0.0f;
        }
    }
}

void MelSpecRenderModule(flecs::world& world) {
    printf("MelSpecRenderModule: Starting initialization...\n");

    // Register components
    world.component<MelSpecRender>();
    world.component<MelSpecConfig>();

    // Get graphics context
    auto graphicsEntity = world.lookup("Graphics");
    if (!graphicsEntity) {
        fprintf(stderr, "MelSpecRender: Graphics entity not found!\n");
        return;
    }

    auto graphics = graphicsEntity.try_get<Graphics>();
    if (!graphics || !graphics->vg) {
        fprintf(stderr, "MelSpecRender: NanoVG context not available!\n");
        return;
    }

    printf("MelSpecRenderModule: Graphics context found, proceeding...\n");

    // Calculate texture dimensions
    // int rollingFrames = (int)((5.0f * 22050) / 256);  // ~431 frames
    int rollingFrames = (int)((24.0f * 22050) / 256);  // ~431 frames
    int texWidth = rollingFrames;
    int texHeight = 128;  // N_MELS

    // Initialize microphone IPC state
    g_micState.socket_path = "/tmp/mel_spec_mic.sock";
    g_micState.vg = graphics->vg;
    g_micState.texWidth = texWidth;
    g_micState.texHeight = texHeight;
    g_micState.socket_fd = -1;
    g_micState.imageBuffer = nullptr;
    g_micState.imageBufferSize = 0;
    g_micState.recvBuffer = nullptr;
    g_micState.recvBufferSize = 0;
    g_micState.recvBufferUsed = 0;
    g_micState.needsUpload = false;

    // Create GL texture for mic
    glGenTextures(1, &g_micState.glTexture);
    glBindTexture(GL_TEXTURE_2D, g_micState.glTexture);
    unsigned char* blackData = (unsigned char*)calloc(texWidth * texHeight * 3, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, blackData);
    free(blackData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    g_micState.nvgImageHandle = nvglCreateImageFromHandleGL2(g_micState.vg, g_micState.glTexture, texWidth, texHeight, 0);
    printf("MelSpecRender: Created mic GL texture %dx%d (ID=%u, NVG=%d)\n", texWidth, texHeight, g_micState.glTexture, g_micState.nvgImageHandle);

    // Initialize system audio IPC state
    g_sysAudioState.socket_path = "/tmp/mel_spec_sys.sock";
    g_sysAudioState.vg = graphics->vg;
    g_sysAudioState.texWidth = texWidth;
    g_sysAudioState.texHeight = texHeight;
    g_sysAudioState.socket_fd = -1;
    g_sysAudioState.imageBuffer = nullptr;
    g_sysAudioState.imageBufferSize = 0;
    g_sysAudioState.recvBuffer = nullptr;
    g_sysAudioState.recvBufferSize = 0;
    g_sysAudioState.recvBufferUsed = 0;
    g_sysAudioState.needsUpload = false;

    // Create GL texture for system audio
    glGenTextures(1, &g_sysAudioState.glTexture);
    glBindTexture(GL_TEXTURE_2D, g_sysAudioState.glTexture);
    blackData = (unsigned char*)calloc(texWidth * texHeight * 3, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, blackData);
    free(blackData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    g_sysAudioState.nvgImageHandle = nvglCreateImageFromHandleGL2(g_sysAudioState.vg, g_sysAudioState.glTexture, texWidth, texHeight, 0);
    printf("MelSpecRender: Created sys audio GL texture %dx%d (ID=%u, NVG=%d)\n", texWidth, texHeight, g_sysAudioState.glTexture, g_sysAudioState.nvgImageHandle);

    // Create listening sockets
    g_micState.listen_fd = create_listen_socket(g_micState.socket_path.c_str());
    if (g_micState.listen_fd < 0) {
        fprintf(stderr, "MelSpecRender: Failed to create mic listen socket\n");
    }

    g_sysAudioState.listen_fd = create_listen_socket(g_sysAudioState.socket_path.c_str());
    if (g_sysAudioState.listen_fd < 0) {
        fprintf(stderr, "MelSpecRender: Failed to create sys audio listen socket\n");
    }

    // Spawn child processes
    printf("MelSpecRender: Spawning audio client processes...\n");
    g_micState.child_pid = spawn_audio_client(g_micState.socket_path.c_str(), 0);
    g_sysAudioState.child_pid = spawn_audio_client(g_sysAudioState.socket_path.c_str(), 1);

    if (g_micState.child_pid < 0 || g_sysAudioState.child_pid < 0) {
        fprintf(stderr, "MelSpecRender: Failed to spawn child processes\n");
    } else {
        printf("MelSpecRender: Spawned mic client (PID %d) and sys audio client (PID %d)\n",
               g_micState.child_pid, g_sysAudioState.child_pid);
    }

    // Create config entity
    auto configEntity = world.entity("MelSpecConfig")
        .set<MelSpecConfig>({
            .enabled = true,
            .alpha = 1.0f,
            .scale = 1.0f,
            .zIndex = 310
        });

    // Create mel_spec render entity (microphone)
    // Note: .enabled = false because we render these in the Hearing editor panel, not above the avatar
    auto melSpecEntity = world.entity("MelSpecRenderer")
        .set<MelSpecRender>({
            .nvgTextureHandle = g_micState.nvgImageHandle,
            .width = texWidth,
            .height = texHeight,
            .imageData = nullptr,
            .hasUpdate = false,
            .enabled = false,  // Disabled - rendered in editor panel instead
            .xOffset = 0.0f,
            .yOffset = -10.0f,
            .zIndex = 310,
            .fillProgress = 0.0f,
            .scrollStartTime = 0.0,
            .totalScrollCommands = 0,
            .renderOffset = 0.0f
        });

    printf("MelSpecRender: Created MelSpecRenderer entity (handle=%d, size=%dx%d)\n",
           g_micState.nvgImageHandle, texWidth, texHeight);

    // Create system audio render entity
    // Note: .enabled = false because we render these in the Hearing editor panel, not above the avatar
    auto systemAudioEntity = world.entity("SystemAudioRenderer")
        .set<MelSpecRender>({
            .nvgTextureHandle = g_sysAudioState.nvgImageHandle,
            .width = texWidth,
            .height = texHeight,
            .imageData = nullptr,
            .hasUpdate = false,
            .enabled = false,  // Disabled - rendered in editor panel instead
            .xOffset = 0.0f,
            .yOffset = -150.0f,
            .zIndex = 311,
            .fillProgress = 0.0f,
            .scrollStartTime = 0.0,
            .totalScrollCommands = 0,
            .renderOffset = 0.0f
        });

    printf("MelSpecRender: Created SystemAudioRenderer entity (handle=%d, size=%dx%d)\n",
           g_sysAudioState.nvgImageHandle, texWidth, texHeight);

    // Register update system (runs in OnUpdate phase before rendering)
    world.system<MelSpecRender>("MelSpecUpdateSystem")
        .kind(flecs::OnUpdate)
        .each(melSpecUpdateSystem);

    // Register render system (runs in PostUpdate to queue render commands)
    world.system<MelSpecRender>("MelSpecRenderSystem")
        .kind(flecs::PostUpdate)
        .each([&](flecs::iter& it, size_t i, MelSpecRender& melSpec) {

            RenderQueue& queue = world.ensure<RenderQueue>();
            
            if (melSpec.enabled && melSpec.nvgTextureHandle != -1)
            {
                // Get avatar head position to position above it
                auto avatarHead = world.lookup("AriaHead");
                if (!avatarHead) {
                    return;  // Avatar head doesn't exist, skip rendering
                }

                // In Flecs 4, access Position with Local tag as a pair
                Position avatarPos = avatarHead.get<Position, Local>();
    
                // Calculate position (above avatar in upper left)
                float x = avatarPos.x + melSpec.xOffset;
                float y = avatarPos.y + melSpec.yOffset - melSpec.height;  // Above avatar
    
                // Create image renderable
                ImageRenderable img;
                img.imageHandle = melSpec.nvgTextureHandle;
                img.width = melSpec.width;
                img.height = melSpec.height;
                img.alpha = 1.0f;
                img.spriteX = 0;
                img.spriteY = 0;
                img.spriteW = 0;
                img.spriteH = 0;
                img.tintColor = 0;
                img.tintStrength = 0.0f;
    
                // Queue render command (use zIndex from component)
                queue.addImageCommand({x, y}, img, melSpec.zIndex);

            }
        });
}

void CleanupMelSpec() {
    // Kill child processes
    if (g_micState.child_pid > 0) {
        kill(g_micState.child_pid, SIGTERM);
        waitpid(g_micState.child_pid, NULL, 0);
    }

    if (g_sysAudioState.child_pid > 0) {
        kill(g_sysAudioState.child_pid, SIGTERM);
        waitpid(g_sysAudioState.child_pid, NULL, 0);
    }

    // Close sockets
    if (g_micState.socket_fd >= 0) close(g_micState.socket_fd);
    if (g_micState.listen_fd >= 0) close(g_micState.listen_fd);
    if (g_sysAudioState.socket_fd >= 0) close(g_sysAudioState.socket_fd);
    if (g_sysAudioState.listen_fd >= 0) close(g_sysAudioState.listen_fd);

    // Remove socket files
    unlink(g_micState.socket_path.c_str());
    unlink(g_sysAudioState.socket_path.c_str());

    // Free buffers
    free(g_micState.imageBuffer);
    free(g_micState.recvBuffer);
    free(g_sysAudioState.imageBuffer);
    free(g_sysAudioState.recvBuffer);

    // Delete GL textures
    if (g_micState.glTexture) glDeleteTextures(1, &g_micState.glTexture);
    if (g_sysAudioState.glTexture) glDeleteTextures(1, &g_sysAudioState.glTexture);
}
