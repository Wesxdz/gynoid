#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "../deps/mel_spec/src/audio_stream.h"
#include <sys/prctl.h>
#include <signal.h>

static int g_socket_fd = -1;
static int g_use_system_audio = 0;
static int g_frame_count = 0;
static unsigned char* g_prev_image = NULL;
static int g_prev_width = 0;
static int g_prev_height = 0;

// Callback when mel spectrogram frame is ready
void frame_callback(unsigned char* image_data, int width, int height, void* user_data) {
    g_frame_count++;

    if (g_socket_fd < 0 || !image_data) {
        if (g_frame_count % 100 == 0) {
            fprintf(stderr, "[CLIENT] Frame %d: Invalid socket or data (fd=%d)\n", g_frame_count, g_socket_fd);
        }
        return;
    }

    // First frame: send full image
    if (g_prev_image == NULL || g_prev_width != width || g_prev_height != height) {
        // Header: [x, y, w, h] - position and size of update region
        int header[4] = {0, 0, width, height};
        ssize_t sent = send(g_socket_fd, header, sizeof(header), MSG_NOSIGNAL);
        if (sent != sizeof(header)) {
            fprintf(stderr, "[CLIENT] Frame %d: Failed to send header\n", g_frame_count);
            return;
        }

        // Send full image
        int size = width * height * 3;
        sent = send(g_socket_fd, image_data, size, MSG_NOSIGNAL);
        if (sent != size) {
            fprintf(stderr, "[CLIENT] Frame %d: Failed to send image (sent %zd/%d)\n",
                    g_frame_count, sent, size);
            return;
        }

        // Allocate previous image buffer
        g_prev_image = (unsigned char*)realloc(g_prev_image, size);
        memcpy(g_prev_image, image_data, size);
        g_prev_width = width;
        g_prev_height = height;

        if (g_frame_count % 100 == 0) {
            printf("[CLIENT] Frame %d: Sent full %dx%d image\n", g_frame_count, width, height);
        }
        return;
    }

    // Check if the entire buffer shifted (rolling window scrolling)
    // Compare leftmost columns - if different from previous rightmost, we scrolled
    int scrolled = 0;
    for (int x = 0; x < width - 1; x++) {
        for (int y = 0; y < height; y++) {
            int curr_idx = (y * width + x) * 3;
            int prev_idx = (y * width + (x + 1)) * 3;
            if (memcmp(&image_data[curr_idx], &g_prev_image[prev_idx], 3) != 0) {
                goto check_rightmost;
            }
        }
    }
    scrolled = 1;

check_rightmost:
    if (scrolled) {
        // When scrolled, send the entire width but only rows that changed
        // This avoids complex GPU-side scrolling in the parent
        // Send full width update at x=0
        int header[4] = {0, 0, width, height};
        ssize_t sent = send(g_socket_fd, header, sizeof(header), MSG_NOSIGNAL);
        if (sent != sizeof(header)) {
            fprintf(stderr, "[CLIENT] Frame %d: Failed to send scroll header\n", g_frame_count);
            return;
        }

        // Send full image (shifted)
        int size = width * height * 3;
        sent = send(g_socket_fd, image_data, size, MSG_NOSIGNAL);
        if (sent != size) {
            fprintf(stderr, "[CLIENT] Frame %d: Failed to send scrolled image\n", g_frame_count);
            return;
        }

        memcpy(g_prev_image, image_data, width * height * 3);

        if (g_frame_count % 100 == 0) {
            printf("[CLIENT] Frame %d: Sent scrolled full image\n", g_frame_count);
        }
        return;
    }

    // Not scrolling - find rightmost changed column
    int last_changed_col = -1;
    for (int x = width - 1; x >= 0; x--) {
        for (int y = 0; y < height; y++) {
            int idx = (y * width + x) * 3;
            if (memcmp(&image_data[idx], &g_prev_image[idx], 3) != 0) {
                last_changed_col = x;
                goto found_change;
            }
        }
    }
found_change:

    if (last_changed_col == -1) {
        // No changes
        return;
    }

    // Send only the rightmost changed column(s)
    int update_x = last_changed_col;
    int update_width = width - last_changed_col;

    // Header: [x, y, w, h]
    int header[4] = {update_x, 0, update_width, height};
    ssize_t sent = send(g_socket_fd, header, sizeof(header), MSG_NOSIGNAL);
    if (sent != sizeof(header)) {
        fprintf(stderr, "[CLIENT] Frame %d: Failed to send header\n", g_frame_count);
        return;
    }

    // Send only the changed columns
    for (int y = 0; y < height; y++) {
        int idx = (y * width + update_x) * 3;
        int row_size = update_width * 3;
        sent = send(g_socket_fd, &image_data[idx], row_size, MSG_NOSIGNAL);
        if (sent != row_size) {
            fprintf(stderr, "[CLIENT] Frame %d: Failed to send row %d\n", g_frame_count, y);
            return;
        }
    }

    // Update previous image
    memcpy(g_prev_image, image_data, width * height * 3);

    if (g_frame_count % 100 == 0) {
        printf("[CLIENT] Frame %d: Sent delta update at x=%d, w=%d\n",
               g_frame_count, update_x, update_width);
    }
}

int main(int argc, char** argv) {
    prctl(PR_SET_PDEATHSIG, SIGTERM);

    signal(SIGTERM, _exit);
    signal(SIGHUP, _exit);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <socket_path> <0=mic|1=system_audio>\n", argv[0]);
        return 1;
    }

    const char* socket_path = argv[1];
    g_use_system_audio = atoi(argv[2]);

    printf("[CLIENT] Starting audio stream client (source=%s, socket=%s)\n",
           g_use_system_audio ? "system_audio" : "microphone", socket_path);

    // Create Unix domain socket
    g_socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_socket_fd < 0) {
        perror("[CLIENT] socket");
        return 1;
    }

    // Connect to parent process
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    printf("[CLIENT] Connecting to socket...\n");
    int retries = 10;
    while (retries-- > 0) {
        if (connect(g_socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            break;
        }
        usleep(100000); // 100ms
    }

    if (retries <= 0) {
        perror("[CLIENT] connect");
        close(g_socket_fd);
        return 1;
    }

    printf("[CLIENT] Connected successfully\n");

    // Initialize audio stream
    AudioStreamContext* ctx = audio_stream_init(g_use_system_audio, frame_callback, NULL);
    if (!ctx) {
        fprintf(stderr, "[CLIENT] Failed to initialize audio stream\n");
        close(g_socket_fd);
        return 1;
    }

    // Start audio stream
    if (audio_stream_start(ctx) != 0) {
        fprintf(stderr, "[CLIENT] Failed to start audio stream\n");
        audio_stream_free(ctx);
        close(g_socket_fd);
        return 1;
    }

    printf("[CLIENT] Audio stream started, streaming data...\n");

    // Run indefinitely
    while (1) {
        sleep(1);
    }

    // Cleanup (unreachable but for completeness)
    audio_stream_stop(ctx);
    audio_stream_free(ctx);
    close(g_socket_fd);
    return 0;
}
