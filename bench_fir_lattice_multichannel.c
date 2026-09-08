/* Multi-channel FIR lattice: CMSIS-DSP per channel, versus channel-innermost.
 *
 * A harder case than the IIR lattice, and a better test of the rule. It also
 * has no NEON path -- but CMSIS software-pipelines FOUR SAMPLES at a time
 * through the stage chain, so it already extracts instruction-level
 * parallelism the IIR lattice version does not. The question is whether true
 * SIMD across channels still beats scalar ILP across samples.
 */
#include "arm_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9; }

/* The FIR lattice, channel-innermost.
 *
 *     f_0(n) = g_0(n) = x(n)
 *     f_{i+1}(n) = f_i(n) + k_i * g_i(n-1)
 *     g_{i+1}(n) = k_i * f_i(n) + g_i(n-1)
 *
 * The delay is on g, and stage i CONSUMES g_i(n-1) while PRODUCING g_{i+1}(n)
 * -- so the output goes to index i+1, not i. Writing it back to i is a
 * different filter that agrees at sample 0 and drifts after, which is exactly
 * what the first version did.
 */
static void batched_fir_lattice(const float32_t *k, int stages,
                                float32_t *prev, float32_t *cur,
                                const float32_t *in, float32_t *out,
                                float32_t *fbuf, int n, int C)
{
    for (int t = 0; t < n; t++) {
        const float32_t *xp = in + (size_t)t * C;
        for (int c = 0; c < C; c++) { fbuf[c] = xp[c]; cur[c] = xp[c]; }
        for (int i = 0; i < stages; i++) {
            const float32_t ki = k[i];
            const float32_t *gp = prev + (size_t)i * C;
            float32_t *gn = cur + (size_t)(i + 1) * C;
            for (int c = 0; c < C; c++) {           /* independent lanes */
                const float32_t g = gp[c], f = fbuf[c];
                gn[c]   = ki * f + g;
                fbuf[c] = f + ki * g;
            }
        }
        float32_t *op = out + (size_t)t * C;
        for (int c = 0; c < C; c++) op[c] = fbuf[c];
        { float32_t *tmp = prev; prev = cur; cur = tmp; }
    }
}

int main(int argc, char**argv)
{
    const int C = argc>1?atoi(argv[1]):32;
    const int stages = argc>2?atoi(argv[2]):8;
    const int n = argc>3?atoi(argv[3]):4096;
    float32_t *kc = malloc((size_t)stages*4);
    for (int i=0;i<stages;i++) kc[i] = 0.25f*(float32_t)((i%5)+1)/5.0f;
    float32_t *xi=malloc((size_t)n*C*4), *o_b=malloc((size_t)n*C*4);
    float32_t *o_c=malloc((size_t)n*C*4);
    float32_t *chan=malloc((size_t)n*4), *ochan=malloc((size_t)n*4);
    for (int i=0;i<n*C;i++) xi[i]=(float32_t)((i%97)-48)/48.0f;
    float32_t *g1=calloc((size_t)(stages+1)*C,4);
    float32_t *g2=calloc((size_t)(stages+1)*C,4);
    float32_t *fb=calloc((size_t)C,4);
    float32_t *sc=calloc((size_t)stages,4);

    double bc=1e9,bb=1e9;
    for (int r=0;r<5;r++){
        double t=now();
        for (int c=0;c<C;c++){
            for (int i=0;i<n;i++) chan[i]=xi[(size_t)i*C+c];
            arm_fir_lattice_instance_f32 S;
            memset(sc,0,(size_t)stages*4);
            arm_fir_lattice_init_f32(&S,(uint16_t)stages,kc,sc);
            arm_fir_lattice_f32(&S,chan,ochan,(uint32_t)n);
            for (int i=0;i<n;i++) o_c[(size_t)i*C+c]=ochan[i];
        }
        double d=now()-t; if(d<bc)bc=d;
        memset(g1,0,(size_t)(stages+1)*C*4); memset(g2,0,(size_t)(stages+1)*C*4);
        t=now(); batched_fir_lattice(kc,stages,g1,g2,xi,o_b,fb,n,C);
        d=now()-t; if(d<bb)bb=d;
    }
    double worst=0,scale=0;
    for (size_t i=0;i<(size_t)n*C;i++){ double e=fabs(o_b[i]-o_c[i]);
        if(e>worst)worst=e; if(fabs(o_c[i])>scale)scale=fabs(o_c[i]); }
    if (!(worst < 1e-4*(scale+1.0))) {
        printf("MISMATCH abs=%.3e scale=%.3e\n", worst, scale);
        for(int i=0;i<3;i++) printf("  t=%d cmsis=%.7f ours=%.7f\n",
            i,o_c[(size_t)i*C],o_b[(size_t)i*C]);
        return 1; }
    printf("{\"channels\": %d, \"stages\": %d, \"n\": %d, "
           "\"cmsis_ns_per_sample\": %.3f, \"batched_ns_per_sample\": %.3f, "
           "\"speedup\": %.2f, \"max_abs_diff\": %.1e}\n",
           C,stages,n,bc/((double)n*C)*1e9,bb/((double)n*C)*1e9,bc/bb,worst);
    return 0;
}
