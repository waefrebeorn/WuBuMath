/*
 * generate_training_data.c -- Generate synthetic WuBu training dataset
 *
 * Creates 25 diverse video clips (each 2-6 seconds) totaling ~30 minutes.
 * Each clip exercises different motion patterns for flow matching training.
 *
 * Output: dataset/train/*.mp4 (25 files)
 * Total: ~30 min of diverse synthetic motion
 *
 * Categories:
 *   - Static scenes with color changes (5 clips)
 *   - Linear motion (5 clips)
 *   - Rotational motion (5 clips)
 *   - Zoom in/out (5 clips)
 *   - Complex multi-object motion (5 clips)
 *
 * Each clip: 480P, 30fps, H.264 encoded
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate a frame into buffer [H*W*3] in [0,1] */
static void generate_frame(float* buf, int W, int H, int frame_num, int total_frames,
                            int category, int clip_id) {
    float t = (float)frame_num / (float)total_frames;
    float phase = t * 2.0f * M_PI;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float u = (float)x / (W - 1);
            float v = (float)y / (H - 1);
            int idx = (y * W + x) * 3;

            float r = 0.5f, g = 0.5f, b = 0.5f;

            switch (category) {
                case 0: /* Static color shift */
                    r = 0.5f + 0.4f * sinf(phase + u * 3.0f);
                    g = 0.5f + 0.4f * sinf(phase * 1.3f + v * 2.0f);
                    b = 0.5f + 0.4f * cosf(phase * 0.7f + (u + v) * 2.5f);
                    break;

                case 1: /* Linear motion (horizontal) */
                    {
                        float offset = fmodf(t * 2.0f, 1.0f);
                        float cx = u - offset;
                        float cy = v - 0.5f;
                        float dist = sqrtf(cx*cx + cy*cy);
                        r = fmaxf(0.0f, 1.0f - dist * 2.0f);
                        g = fmaxf(0.0f, 1.0f - dist * 2.5f) * 0.5f;
                        b = fmaxf(0.0f, 1.0f - dist * 1.5f) * 0.8f;
                    }
                    break;

                case 2: /* Rotational motion */
                    {
                        float cx = u - 0.5f, cy = v - 0.5f;
                        float angle = atan2f(cy, cx) + phase * (1.0f + clip_id * 0.2f);
                        float dist = sqrtf(cx*cx + cy*cy);
                        float ring = fmodf(angle * 3.0f + dist * 10.0f, 1.0f);
                        r = ring * (1.0f - dist);
                        g = ring * dist;
                        b = (1.0f - ring) * (1.0f - dist * 0.5f);
                    }
                    break;

                case 3: /* Zoom */
                    {
                        float zoom = 1.0f + 0.5f * sinf(phase * 2.0f);
                        float cx = (u - 0.5f) * zoom + 0.5f;
                        float cy = (v - 0.5f) * zoom + 0.5f;
                        float dist = sqrtf((cx-0.5f)*(cx-0.5f) + (cy-0.5f)*(cy-0.5f));
                        r = fmaxf(0.0f, 1.0f - dist * 3.0f);
                        g = fmaxf(0.0f, 1.0f - dist * 2.0f) * 0.7f;
                        b = fmaxf(0.0f, 1.0f - dist * 4.0f) * 0.3f;
                    }
                    break;

                case 4: /* Complex multi-object */
                    {
                        /* Multiple moving circles */
                        r = g = b = 0.05f;
                        for (int obj = 0; obj < 3; obj++) {
                            float ox = 0.5f + 0.3f * sinf(phase * (1.0f + obj * 0.3f) + obj * 2.094f);
                            float oy = 0.5f + 0.3f * cosf(phase * (0.7f + obj * 0.2f) + obj * 2.094f);
                            float dist = sqrtf((u-ox)*(u-ox) + (v-oy)*(v-oy));
                            float intensity = fmaxf(0.0f, 1.0f - dist * 8.0f);
                            r += intensity * (obj == 0 ? 1.0f : 0.0f);
                            g += intensity * (obj == 1 ? 1.0f : 0.0f);
                            b += intensity * (obj == 2 ? 1.0f : 0.0f);
                        }
                    }
                    break;
            }

            buf[idx + 0] = fmaxf(0.0f, fminf(1.0f, r));
            buf[idx + 1] = fmaxf(0.0f, fminf(1.0f, g));
            buf[idx + 2] = fmaxf(0.0f, fminf(1.0f, b));
        }
    }
}

int main(void) {
    printf("========================================================\n");
    printf("  Generating WuBu Training Dataset\n");
    printf("========================================================\n\n");

    const char* output_dir = "dataset/train";
    int W = 320, H = 240;
    int fps = 30;

    /* 25 clips: 5 categories × 5 clips each */
    const char* cat_names[] = {"static", "linear", "rotation", "zoom", "complex"};
    int durations[] = {120, 120, 120, 120, 120, /* 2 min each */
                        120, 120, 120, 120, 120,
                        120, 120, 120, 120, 120,
                        120, 120, 120, 120, 120,
                        120, 120, 120, 120, 120};
    int total_videos = 25;
    int total_seconds = 0;

    /* Create output directory */
    system("mkdir -p dataset/train dataset/val dataset/processed");

    for (int clip = 0; clip < total_videos; clip++) {
        int category = clip / 5;
        int clip_in_cat = clip % 5;
        int num_frames = durations[clip] * fps;
        int num_seconds = durations[clip];
        total_seconds += num_seconds;

        char name[256];
        snprintf(name, sizeof(name), "%s_%s_%02d", cat_names[category],
                 cat_names[category], clip_in_cat);

        char out_cmd[512];
        snprintf(out_cmd, sizeof(out_cmd),
            "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - "
            "-c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "
            "-t %d %s/%s.mp4 2>/dev/null",
            W, H, fps, num_seconds, output_dir, name);

        FILE* pipe = popen(out_cmd, "w");
        if (!pipe) {
            printf("  Failed to create pipe for %s\n", name);
            continue;
        }

        printf("  [%2d/25] %s: %ds (%d frames)\n", clip + 1, name, num_seconds, num_frames);

        /* Generate frames and pipe to ffmpeg */
        size_t frame_size = (size_t)W * H * 3;
        float* frame = (float*)malloc(frame_size * sizeof(float));
        uint8_t* row = (uint8_t*)malloc(frame_size);

        for (int f = 0; f < num_frames; f++) {
            generate_frame(frame, W, H, f, num_frames, category, clip_in_cat);

            /* Convert float [0,1] to uint8 */
            for (int i = 0; i < W * H * 3; i++) {
                int val = (int)(frame[i] * 255.0f);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                row[i] = (uint8_t)val;
            }
            fwrite(row, 1, frame_size, pipe);
        }

        int ret = pclose(pipe);
        free(frame);
        free(row);

        if (ret == 0) {
            printf("        ✓ Generated successfully\n");
        } else {
            printf("        ✗ ffmpeg error (code %d)\n", ret);
        }
    }

    printf("\n========================================================\n");
    printf("  Dataset Generation Complete\n");
    printf("========================================================\n");
    printf("  Videos:     25 clips\n");
    printf("  Duration:   %ds (%d min %d sec)\n",
           total_seconds, total_seconds / 60, total_seconds % 60);
    printf("  Resolution: %dx%d @ %dfps\n", W, H, fps);
    printf("  Output:     %s/\n", output_dir);
    printf("========================================================\n");

    /* Generate manifest */
    FILE* manifest = fopen("dataset/manifest.json", "w");
    if (manifest) {
        fprintf(manifest, "{\n");
        fprintf(manifest, "  \"dataset\": \"wubu_synthetic_v1\",\n");
        fprintf(manifest, "  \"version\": \"1.0.0\",\n");
        fprintf(manifest, "  \"total_videos\": %d,\n", total_videos);
        fprintf(manifest, "  \"total_duration_sec\": %d,\n", total_seconds);
        fprintf(manifest, "  \"resolution\": \"%dx%d\",\n", W, H);
        fprintf(manifest, "  \"fps\": %d,\n", fps);
        fprintf(manifest, "  \"categories\": [");
        for (int c = 0; c < 5; c++) {
            fprintf(manifest, \"%s\"%s\"", cat_names[c], c < 4 ? ", " : "");
        }
        fprintf(manifest, "],\n");
        fprintf(manifest, "  \"train_split\": 0.8,\n");
        fprintf(manifest, "  \"val_split\": 0.2\n");
        fprintf(manifest, "}\n");
        fclose(manifest);
        printf("  Manifest:   dataset/manifest.json\n");
    }

    return 0;
}
