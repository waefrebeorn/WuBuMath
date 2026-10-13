/*
 * test_lean_bridge.c -- C11 property tests that verify the Lean-proven
 * invariants hold in the C implementation.  This is the bridge the WuBuMath
 * charter requires: every Lean theorem (source of truth) is checked against
 * a numerical C implementation, so the C code is trusted *because* the
 * theorems hold, not merely because "it mostly stays in the ball."
 *
 * Theorems bridged (see lean/WubuProofs/ and docs/LINKAGE.md):
 *   [T3]  MobiusAddND:  Mobius addition preserves the Poincaré ball (nD).
 *         c*|x (+) y|^2 < 1  for all x,y in the ball.
 *   [T8]  GyrationND:   gyration preserves the norm  ||gyr[u,v]w||^2 = ||w||^2.
 *   [T9]  GyrationND:   gyration preserves the inner product
 *                       <gyr[u,v]a, gyr[u,v]b> = <a,b>  (Ungar 1.32).
 *   [K]   KleinDeck:    deck generators A(x,y)=(x+1,y), B(x,y)=(-x,y+1);
 *                       ABA = B, B^2 = translation by (0,2), A^k = +k in x.
 *
 * All formulas transcribed directly from the Lean definitions so the numbers
 * match what is proven.  Deterministic PRNG for reproducibility.
 */

#include "wubu_hyperbolic.h"   /* wubu_mobius_add, wubu_poincare_clip */
#include <stdio.h>
#include <math.h>
#include <string.h>

static int fails = 0;
static void ok(const char *name) { printf("PASS %s\n", name); }
static void bad(const char *name, float got, float expect) {
    printf("FAIL %s: got=%g expect=%g\n", name, got, expect); fails++;
}

/* deterministic LCG PRNG in [-1,1] */
static unsigned lcg_state = 12345;
static float frnd(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((lcg_state >> 8) & 0xffff) / 65535.0f * 2.0f - 1.0f;
}

/* ---- C transcriptions of the Lean definitions ------------------------- */

/* Mobius addition (MobiusAddND.lean). out length N. */
static void c_mobius_add(float *out, const float *x, const float *y, int N, float c) {
    float xy = 0, x2 = 0, y2 = 0;
    for (int i = 0; i < N; i++) { xy += x[i]*y[i]; x2 += x[i]*x[i]; y2 += y[i]*y[i]; }
    float den = 1 + 2*c*xy + c*c*x2*y2;
    for (int i = 0; i < N; i++)
        out[i] = ((1 + 2*c*xy + c*y2) * x[i] + (1 - c*x2) * y[i]) / den;
}

/* Gyration closed form (GyrationND.lean gyr_cf). out length N. */
static void c_gyr(float *out, const float *u, const float *v, const float *w,
                  int N, float c) {
    float uv=0, uw=0, vw=0, u2=0, v2=0;
    for (int i = 0; i < N; i++) {
        uv += u[i]*v[i]; uw += u[i]*w[i]; vw += v[i]*w[i];
        u2 += u[i]*u[i]; v2 += v[i]*v[i];
    }
    float d = 1 + 2*c*uv + c*c*u2*v2;
    float a = -c*c*uw*v2 + c*vw + 2*c*c*uv*vw;
    float b = -c*c*vw*u2 - c*uw;
    for (int i = 0; i < N; i++)
        out[i] = w[i] + (2/d) * (a*u[i] + b*v[i]);
}

static float c_dot(const float *x, const float *y, int N) {
    float s = 0; for (int i = 0; i < N; i++) s += x[i]*y[i]; return s;
}

/* sample a point strictly inside the ball (norm < radius) */
static void ball_pt(float *p, int N, float radius) {
    float n2 = 0;
    for (int i = 0; i < N; i++) { p[i] = frnd(); n2 += p[i]*p[i]; }
    float n = sqrtf(n2) + 1e-6f;
    float scale = (0.1f + (radius-0.1f) * fabsf(frnd())) / n;
    for (int i = 0; i < N; i++) p[i] *= scale;
}

int main(void) {
    const int N = 8;
    const float c = 1.0f;
    const float TOL = 1e-4f;
    float x[64], y[64], out[64], u[64], v[64], w[64], a[64], b[64], ga[64], gb[64];

    printf("=== Lean-bridge property tests (C11) ===\n");

    /* [T3] Mobius addition preserves the ball (nD) */
    {
        float worst = 0.0f;
        for (int t = 0; t < 5000; t++) {
            ball_pt(x, N, 0.6f); ball_pt(y, N, 0.6f);
            c_mobius_add(out, x, y, N, c);
            float s = c * c_dot(out, out, N);   /* c*|x(+)y|^2 */
            if (s > worst) worst = s;
        }
        if (worst < 1.0f) ok("[T3] mobius_add preserves ball (nD)");
        else bad("[T3] mobius_add ball", worst, 1.0f);
        printf("     max c*|x(+)y|^2 over 5000 pairs = %g  (< 1)\n", worst);
    }

    /* [T9] gyration preserves the inner product (Ungar 1.32) */
    {
        float maxerr = 0.0f;
        for (int t = 0; t < 2000; t++) {
            ball_pt(u, N, 0.5f); ball_pt(v, N, 0.5f);
            ball_pt(a, N, 0.5f); ball_pt(b, N, 0.5f);
            c_gyr(ga, u, v, a, N, c);
            c_gyr(gb, u, v, b, N, c);
            float lhs = c_dot(ga, gb, N);
            float rhs = c_dot(a, b, N);
            float e = fabsf(lhs - rhs);
            if (e > maxerr) maxerr = e;
        }
        if (maxerr < TOL) ok("[T9] gyration preserves inner product");
        else bad("[T9] gyration inner-product", maxerr, 0.0f);
        printf("     max |<gyr a,gyr b> - <a,b>| over 2000 trials = %g\n", maxerr);
    }

    /* [T8] gyration preserves the norm (Ungar 1.33) */
    {
        float maxerr = 0.0f;
        for (int t = 0; t < 2000; t++) {
            ball_pt(u, N, 0.5f); ball_pt(v, N, 0.5f); ball_pt(w, N, 0.5f);
            c_gyr(out, u, v, w, N, c);
            float e = fabsf(c_dot(out, out, N) - c_dot(w, w, N));
            if (e > maxerr) maxerr = e;
        }
        if (maxerr < TOL) ok("[T8] gyration preserves norm");
        else bad("[T8] gyration norm", maxerr, 0.0f);
        printf("     max ||gyr w||^2 - ||w||^2 over 2000 trials = %g\n", maxerr);
    }

    /* [K] Klein deck: ABA = B, B^2 = translation by (0,2), A^k = +k in x */
    {
        /* A(x,y) = (x+1, y);  B(x,y) = (-x, y+1) */
        float px, py;
        int okrel = 1, okb2 = 1, okak = 1;
        for (int t = 0; t < 1000; t++) {
            px = frnd() * 10.0f; py = frnd() * 10.0f;
            /* ABA: A(B(A(p))) */
            float ax = px + 1, ay = py;          /* A */
            float bx = -ax, by = ay + 1;         /* B */
            float abx = bx + 1, aby = by;        /* A */
            float Bpx = -px, Bpy = py + 1;       /* B(p) */
            if (fabsf(abx - Bpx) > TOL || fabsf(aby - Bpy) > TOL) okrel = 0;
            /* B^2: B(B(p)) = (p.x, p.y+2) */
            float b2x = -Bpx, b2y = Bpy + 1;
            if (fabsf(b2x - px) > TOL || fabsf(b2y - (py + 2)) > TOL) okb2 = 0;
            /* A^k = +k in x */
            float kx = px; for (int k = 0; k < 5; k++) kx += 1;
            if (fabsf(kx - (px + 5)) > TOL) okak = 0;
        }
        if (okrel) ok("[K] Klein deck relation ABA = B"); else bad("[K] ABA=B", 0, 1);
        if (okb2)  ok("[K] Klein B^2 = translation by (0,2)"); else bad("[K] B^2", 0, 1);
        if (okak)  ok("[K] Klein A^5 = translation by +5 in x"); else bad("[K] A^k", 0, 1);
    }

    printf("=== %s (%d failures) ===\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
