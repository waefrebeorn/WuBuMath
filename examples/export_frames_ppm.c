/*
 * export_frames_ppm.c -- Export raw float frames to viewable PPM images
 * Reads the output/raw_frames.bin from media_creator and saves PPM files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char** argv) {
    const char* input = "output/raw_frames.bin";
    const char* output_dir = "output/frames";
    int target_frame = -1; /* -1 = export all */

    if (argc > 1) target_frame = atoi(argv[1]);

    /* Create output directory */
    system("mkdir -p output/frames");

    FILE* f = fopen(input, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", input);
        return 1;
    }

    /* Read header */
    int num_frames, img_size, channels;
    fread(&num_frames, sizeof(int), 1, f);
    fread(&img_size, sizeof(int), 1, f);
    fread(&channels, sizeof(int), 1, f);

    printf("Frames: %d, Size: %dx%d, Channels: %d\n", num_frames, img_size, img_size, channels);

    size_t frame_size = (size_t)img_size * img_size * channels;
    float* frame = (float*)malloc(frame_size * sizeof(float));

    int start = (target_frame >= 0) ? target_frame : 0;
    int end = (target_frame >= 0) ? target_frame + 1 : num_frames;

    for (int i = start; i < end; i++) {
        fread(frame, sizeof(float), frame_size, f);

        char path[256];
        snprintf(path, sizeof(path), "%s/frame_%03d.ppm", output_dir, i);

        FILE* out = fopen(path, "wb");
        if (!out) { fprintf(stderr, "Cannot create %s\n", path); continue; }

        /* PPM header */
        fprintf(out, "P6\n%d %d\n255\n", img_size, img_size);

        /* Convert float [-1,1] to uint8 [0,255] */
        for (int p = 0; p < img_size * img_size; p++) {
            unsigned char rgb[3];
            for (int c = 0; c < 3; c++) {
                float val = frame[p * channels + c];
                if (val < 0.0f) val = 0.0f;
                if (val > 1.0f) val = 1.0f;
                rgb[c] = (unsigned char)(val * 255.0f);
            }
            fwrite(rgb, 1, 3, out);
        }
        fclose(out);
        printf("Saved %s\n", path);
    }

    free(frame);
    fclose(f);
    printf("Done. Export frames with: ffplay output/frames/frame_000.ppm\n");
    return 0;
}
