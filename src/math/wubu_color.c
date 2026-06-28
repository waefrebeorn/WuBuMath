/*
 * wubu_color.c -- RGB/HSL color manifold (slermed from vhf_audio.py)
 */

#include "wubumath.h"

WubuHSL wubu_rgb_to_hsl(WubuRGB rgb) {
    float r = rgb.r, g = rgb.g, b = rgb.b;
    float cmax = fmaxf(fmaxf(r, g), b);
    float cmin = fminf(fminf(r, g), b);
    float delta = cmax - cmin;
    float l = (cmax + cmin) / 2.0f;
    float s = 0.0f;
    float h = 0.0f;

    if (delta > 1e-8f) {
        s = delta / (1.0f - fabsf(2.0f * l - 1.0f) + 1e-8f);
        if (cmax == r) {
            h = fmodf((g - b) / (delta + 1e-8f), 6.0f);
        } else if (cmax == g) {
            h = ((b - r) / (delta + 1e-8f)) + 2.0f;
        } else {
            h = ((r - g) / (delta + 1e-8f)) + 4.0f;
        }
        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }

    WubuHSL hsl = { .h = h, .s = s, .l = l };
    return hsl;
}

WubuRGB wubu_hsl_to_rgb(WubuHSL hsl) {
    float h = hsl.h, s = hsl.s, l = hsl.l;
    WubuRGB rgb;

    if (s < 1e-8f) {
        rgb.r = rgb.g = rgb.b = l;
        return rgb;
    }

    float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float hh = h * 6.0f;
    float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float m = l - c / 2.0f;

    float r1, g1, b1;
    if (hh < 1)      { r1 = c; g1 = x; b1 = 0; }
    else if (hh < 2) { r1 = x; g1 = c; b1 = 0; }
    else if (hh < 3) { r1 = 0; g1 = c; b1 = x; }
    else if (hh < 4) { r1 = 0; g1 = x; b1 = c; }
    else if (hh < 5) { r1 = x; g1 = 0; b1 = c; }
    else             { r1 = c; g1 = 0; b1 = x; }

    rgb.r = r1 + m;
    rgb.g = g1 + m;
    rgb.b = b1 + m;
    return rgb;
}

float wubu_rgb_to_grayscale(WubuRGB rgb) {
    return 0.2989f * rgb.r + 0.5870f * rgb.g + 0.1140f * rgb.b;
}

float wubu_circular_l1_loss(float pred, float target) {
    float diff = fabsf(pred - target);
    return fminf(diff, 1.0f - diff);
}
