/* gen_train_ppm.c -- Generate synthetic PPM training frames for GAAD encoder
 *
 * Generates diverse synthetic video frames procedurally:
 * - Static color shifts
 * - Linear motion
 * - Rotational motion
 * - Zoom in/out
 * - Complex multi-object motion
 *
 * Usage: ./bin/gen_train_ppm <output_dir> <num_frames> <W> <H>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void generate_frame(float* buf, int W, int H, int frame_num, int total_frames,
                            int category, int seed) {
    float t = (float)frame_num / (float)total_frames;
    float phase = t * 2.0f * M_PI;
    float u, v, r, g, b;
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            u = (float)x / (W - 1);
            v = (float)y / (H - 1);
            int idx = (y * W + x) * 3;
            r = 0.5f; g = 0.5f; b = 0.5f;
            
            switch (category) {
                case 0: /* Static color shift */
                    r = 0.5f + 0.4f * sinf(phase + u * 3.0f + seed);
                    g = 0.5f + 0.4f * sinf(phase * 1.3f + v * 2.0f + seed * 1.1f);
                    b = 0.5f + 0.4f * cosf(phase * 0.7f + (u + v) * 2.5f + seed * 0.9f);
                    break;
                    
                case 1: { /* Linear motion */
                    float offset = fmodf(t * 2.0f + seed * 0.1f, 1.0f);
                    float cx = u - offset;
                    float cy = v - 0.5f;
                    float dist = sqrtf(cx*cx + cy*cy);
                    r = fmaxf(0.0f, 1.0f - dist * 2.0f);
                    g = fmaxf(0.0f, 1.0f - dist * 2.5f) * 0.5f;
                    b = fmaxf(0.0f, 1.0f - dist * 1.5f) * 0.8f;
                    break;
                }
                
                case 2: { /* Rotational motion */
                    float cx = u - 0.5f, cy = v - 0.5f;
                    float angle = atan2f(cy, cx) + phase * (1.0f + seed * 0.2f);
                    float dist = sqrtf(cx*cx + cy*cy);
                    float ring = fmodf(angle * 3.0f + dist * 10.0f, 1.0f);
                    r = ring * (1.0f - dist);
                    g = ring * dist;
                    b = (1.0f - ring) * (1.0f - dist * 0.5f);
                    break;
                }
                
                case 3: { /* Zoom */
                    float zoom = 1.0f + 0.5f * sinf(phase * 2.0f + seed);
                    float cx = (u - 0.5f) * zoom + 0.5f;
                    float cy = (v - 0.5f) * zoom + 0.5f;
                    float dist = sqrtf((cx-0.5f)*(cx-0.5f) + (cy-0.5f)*(cy-0.5f));
                    r = fmaxf(0.0f, 1.0f - dist * 3.0f);
                    g = fmaxf(0.0f, 1.0f - dist * 2.0f) * 0.7f;
                    b = fmaxf(0.0f, 1.0f - dist * 4.0f) * 0.3f;
                    break;
                }
                
                case 4: { /* Complex multi-object */
                    r = g = b = 0.05f;
                    for (int obj = 0; obj < 3; obj++) {
                        float ox = 0.5f + 0.3f * sinf(phase * (1.0f + obj * 0.3f) + obj * 2.094f + seed);
                        float oy = 0.5f + 0.3f * cosf(phase * (0.7f + obj * 0.2f) + obj * 2.094f + seed);
                        float dist = sqrtf((u-ox)*(u-ox) + (v-oy)*(v-oy));
                        float intensity = fmaxf(0.0f, 1.0f - dist * 8.0f);
                        r += intensity * (obj == 0 ? 1.0f : 0.0f);
                        g += intensity * (obj == 1 ? 1.0f : 0.0f);
                        b += intensity * (obj == 2 ? 1.0f : 0.0f);
                    }
                    break;
                }
            }
            
            buf[idx + 0] = fmaxf(0.0f, fminf(1.0f, r));
            buf[idx + 1] = fmaxf(0.0f, fminf(1.0f, g));
            buf[idx + 2] = fmaxf(0.0f, fminf(1.0f, b));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <output_dir> <num_frames> <W> <H>\n", argv[0]);
        return 1;
    }
    
    const char* output_dir = argv[1];
    int num_frames = atoi(argv[2]);
    int W = atoi(argv[3]);
    int H = atoi(argv[4]);
    
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir);
    system(cmd);
    
    float* frame = (float*)malloc(W * H * 3 * sizeof(float));
    
    for (int f = 0; f < num_frames; f++) {
        int category = f % 5;
        int seed = f / 5;
        
        generate_frame(frame, W, H, f, num_frames, category, seed);
        
        char path[256];
        snprintf(path, sizeof(path), "%s/frame_%04d.ppm", output_dir, f);
        
        FILE* fp = fopen(path, "wb");
        if (!fp) continue;
        
        fprintf(fp, "P6\n%d %d\n255\n", W, H);
        
        for (int i = 0; i < W * H * 3; i++) {
            int val = (int)(frame[i] * 255.0f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            fputc((uint8_t)val, fp);
        }
        
        fclose(fp);
        
        if (f % 50 == 0) {
            printf("  Generated %d/%d frames\n", f, num_frames);
        }
    }
    
    free(frame);
    printf("Done. Generated %d frames in %s/\n", num_frames, output_dir);
    return 0;
}
