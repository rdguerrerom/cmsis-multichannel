/* Multi-channel FIR: CMSIS-DSP per channel, versus channel-innermost.
 *
 * FIR is the most used filter in DSP and, unlike a biquad, it has no
 * loop-carried dependency -- each output is an independent dot product over
 * numTaps. So CMSIS's within-channel NEON has something real to work with
 * here, and the batching argument is correspondingly weaker. That is exactly
 * why it is worth measuring rather than assuming: the biquad result does not
 * transfer to a kernel whose parallelism is already reachable.
 */
#include "arm_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9; }

/* Channel-innermost FIR over a padded window.
 *
 * Two things this gets right that the first attempt did not. CMSIS stores the
 * coefficients TIME-REVERSED -- pCoeffs = {b[N-1], ..., b[0]} -- so the same
 * array has to be read backwards here to be the same filter. And the history
 * is a window into a zero-padded copy of the input rather than a buffer that
 * is shifted every sample: shifting costs O(taps*C) moves per sample, the same
 * order as the arithmetic, which would have doubled the work and measured a
 * data-movement bug as a kernel result.
 */
static void batched_fir(const float32_t *coef, int taps, const float32_t *pad,
                        float32_t *out, int n, int C)
{
    for (int t = 0; t < n; t++) {
        float32_t *op = out + (size_t)t * C;
        for (int c = 0; c < C; c++) op[c] = 0.0f;
        for (int k = 0; k < taps; k++) {
            /* coef is CMSIS-ordered, so b[k] lives at coef[taps-1-k] */
            const float32_t h = coef[taps - 1 - k];
            const float32_t *sp = pad + (size_t)(t + taps - 1 - k) * C;
            for (int c = 0; c < C; c++) op[c] += h * sp[c];   /* lanes */
        }
    }
}

int main(int argc, char**argv)
{
    const int C = argc>1?atoi(argv[1]):32;
    const int taps = argc>2?atoi(argv[2]):32;
    const int n = argc>3?atoi(argv[3]):4096;
    float32_t *coef = malloc((size_t)taps*4);
    for (int k=0;k<taps;k++) coef[k] = 0.5f/(float32_t)taps*(float32_t)((k%7)+1);
    float32_t *xi = malloc((size_t)n*C*4), *o_b = malloc((size_t)n*C*4);
    float32_t *o_c = malloc((size_t)n*C*4);
    float32_t *chan = malloc((size_t)n*4), *ochan = malloc((size_t)n*4);
    for (int i=0;i<n*C;i++) xi[i]=(float32_t)((i%97)-48)/48.0f;
    /* zero-padded history so the window needs no shifting */
    float32_t *pad = calloc((size_t)(n + taps) * C, 4);
    for (int i = 0; i < n*C; i++) pad[(size_t)(taps-1)*C + i] = xi[i];
    float32_t *stc = calloc((size_t)(taps+n), 4);

    double bc=1e9, bb=1e9;
    for (int r=0;r<5;r++){
        double t=now();
        for (int c=0;c<C;c++){
            for (int i=0;i<n;i++) chan[i]=xi[(size_t)i*C+c];
            arm_fir_instance_f32 S;
            memset(stc,0,(size_t)(taps+n)*4);
            arm_fir_init_f32(&S,(uint16_t)taps,coef,stc,(uint32_t)n);
            arm_fir_f32(&S,chan,ochan,(uint32_t)n);
            for (int i=0;i<n;i++) o_c[(size_t)i*C+c]=ochan[i];
        }
        double d=now()-t; if(d<bc)bc=d;
        t=now(); batched_fir(coef,taps,pad,o_b,n,C); d=now()-t; if(d<bb)bb=d;
    }
    double worst=0;
    for (size_t i=0;i<(size_t)n*C;i++){ double e=fabs(o_b[i]-o_c[i]);
        if (e>worst) worst=e; }
    if (!(worst < 1e-4)) { printf("MISMATCH %.3e\n", worst); return 1; }
    printf("{\"channels\": %d, \"taps\": %d, \"n\": %d, "
           "\"cmsis_ns_per_sample\": %.3f, \"batched_ns_per_sample\": %.3f, "
           "\"speedup\": %.2f, \"max_abs_diff\": %.1e}\n",
           C, taps, n, bc/((double)n*C)*1e9, bb/((double)n*C)*1e9, bc/bb, worst);
    return 0;
}
