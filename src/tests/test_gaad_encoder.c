/*
 * test_gaad_encoder.c -- Test the GAAD WuBu encoder
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_gaad_encoder.h"

static int pass = 0, fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fail++; } \
    else { pass++; } \
} while(0)

static void test_init_free(void) {
    printf("  Init/free...\n");
    WubuGaadEncoder enc;
    wubu_gaad_init(&enc, 32);
    ASSERT(enc.num_spiral > 0, "spiral points generated");
    ASSERT(enc.num_regions > 0, "golden regions generated");
    ASSERT(enc.levels[0].c > 0, "level 0 curvature positive");
    ASSERT(enc.levels[3].c > enc.levels[0].c, "curvature increases with level (PHI scaling)");
    wubu_gaad_free(&enc);
}

static void test_encode_finite(void) {
    printf("  Encode produces finite values...\n");
    WubuGaadEncoder enc;
    wubu_gaad_init(&enc, 32);
    
    float image[64*64*3];
    for (int i = 0; i < 64*64*3; i++)
        image[i] = (float)(i % 256) / 255.0f;
    
    float latent[32];
    wubu_gaad_encode(&enc, image, latent);
    
    int finite = 1;
    for (int i = 0; i < 32; i++)
        if (!isfinite(latent[i])) { finite = 0; break; }
    ASSERT(finite, "all latent values are finite (no NaN/Inf)");
    
    wubu_gaad_free(&enc);
}

static void test_full_encode(void) {
    printf("  Full encode (256 patches)...\n");
    WubuGaadEncoder enc;
    wubu_gaad_init(&enc, 32);
    
    float image[64*64*3];
    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) {
        int i = (y*64+x)*3;
        image[i] = (float)x/63.0f;
        image[i+1] = (float)y/63.0f;
        image[i+2] = 0.5f;
    }
    
    float latents[256*32];
    int num;
    wubu_gaad_encode_full(&enc, image, latents, &num);
    
    ASSERT(num == 256, "256 patches encoded");
    
    int finite = 1;
    for (int i = 0; i < 256*32; i++)
        if (!isfinite(latents[i])) { finite = 0; break; }
    ASSERT(finite, "all 256 latent vectors are finite");
    
    /* Check that different patches produce different latents */
    int different = 0;
    for (int i = 0; i < 32; i++) {
        if (latents[i] != latents[32+i]) { different = 1; break; }
    }
    ASSERT(different, "different patches produce different latents");
    
    wubu_gaad_free(&enc);
}

static void test_golden_ratio_spiral(void) {
    printf("  Golden ratio spiral...\n");
    SpiralPoint pts[100];
    int n = generate_phi_spiral(pts, 100, 64, 64);
    ASSERT(n == 100, "100 spiral points generated");
    
    /* First point should be center */
    ASSERT(fabsf(pts[0].x - 32.0f) < 0.01f, "first point x at center");
    ASSERT(fabsf(pts[0].y - 32.0f) < 0.01f, "first point y at center");
    
    /* Points should be within image bounds */
    int in_bounds = 1;
    for (int i = 0; i < n; i++) {
        if (pts[i].x < 0 || pts[i].x >= 64 || pts[i].y < 0 || pts[i].y >= 64) {
            in_bounds = 0; break;
        }
    }
    ASSERT(in_bounds, "all spiral points within image bounds");
}

static void test_golden_subdivide(void) {
    printf("  Golden ratio subdivision...\n");
    Rect rects[256];
    int n = golden_subdivide(rects, 64, 64, 16);
    ASSERT(n >= 16, "at least 16 regions generated");
    
    /* All regions should be valid (x1 < x2, y1 < y2) */
    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (rects[i].x1 >= rects[i].x2 || rects[i].y1 >= rects[i].y2) {
            valid = 0; break;
        }
    }
    ASSERT(valid, "all regions have valid coordinates");
}

static void test_curvature_scaling(void) {
    printf("  Curvature PHI scaling...\n");
    WubuGaadEncoder enc;
    wubu_gaad_init(&enc, 32);
    
    /* c = c_base * PHI^(level % 4 - 1.5) */
    /* Level 0: PHI^(-1.5) = 0.4859 */
    /* Level 1: PHI^(-0.5) = 0.7862 */
    /* Level 2: PHI^(0.5) = 1.2720 */
    /* Level 3: PHI^(1.5) = 2.0582 */
    float expected[4] = {0.4859f, 0.7862f, 1.2720f, 2.0582f};
    for (int l = 0; l < 4; l++) {
        float diff = fabsf(enc.levels[l].c - expected[l]);
        ASSERT(diff < 0.01f, "curvature matches PHI^(level%4-1.5) formula");
    }
    
    wubu_gaad_free(&enc);
}

int main(void) {
    printf("=== GAAD Encoder Tests ===\n\n");
    test_init_free();
    test_encode_finite();
    test_full_encode();
    test_golden_ratio_spiral();
    test_golden_subdivide();
    test_curvature_scaling();
    printf("\n=== Results: %d pass, %d fail ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
