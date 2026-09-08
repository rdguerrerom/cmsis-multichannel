/* Multi-channel IIR lattice: CMSIS-DSP per channel, versus channel-innermost.
 *
 * arm_iir_lattice_f32 is the strongest structural case in the library. It has
 * NO NEON and NO Helium path at all -- unlike the biquad, where at least an
 * attempt exists -- and its inner recurrence is a serial chain: each lattice
 * stage consumes the f value the previous stage produced, so there is nothing
 * to vectorise within a sample even in principle. Samples are chained through
 * the g state, so time is serial too.
 *
 * Channels are independent. That is the only parallelism this kernel has, and
 * the one-signal-per-call API cannot reach it.
 */
#include "arm_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9; }

/* The lattice delay line, channel-innermost, in ONE buffer.
 *
 * Stage i reads g[i+1] and writes g[i]. Processing i in increasing order, the
 * write at i always lands below the read at i+1 that already happened, so a
 * value is never clobbered before use and the ping-pong pair this first used
 * is unnecessary. That matters more than it sounds: on a Cortex-M the state
 * is the constraint, and a second (numStages+1) x C buffer would have doubled
 * the only part of the footprint that scales with channel count.
 */
static void batched_lattice(const float32_t *k, const float32_t *v, int stages,
                            float32_t *g, float32_t *fbuf, float32_t *abuf,
                            const float32_t *in, float32_t *out, int n, int C)
{
    for (int t = 0; t < n; t++) {
        const float32_t *xp = in + (size_t)t * C;
        float32_t *op = out + (size_t)t * C;
        for (int c = 0; c < C; c++) { fbuf[c] = xp[c]; abuf[c] = 0.0f; }
        for (int i = 0; i < stages; i++) {
            const float32_t ki = k[i], vi = v[i];
            const float32_t *gp = g + (size_t)(i + 1) * C;   /* read i+1 */
            float32_t *gw = g + (size_t)i * C;               /* write i   */
            for (int c = 0; c < C; c++) {
                const float32_t gc = gp[c];
                const float32_t f  = fbuf[c] - ki * gc;
                const float32_t gn = gc + ki * f;
                abuf[c] += gn * vi;
                gw[c]    = gn;
                fbuf[c]  = f;
            }
        }
        const float32_t vlast = v[stages];
        float32_t *gtail = g + (size_t)stages * C;
        for (int c = 0; c < C; c++) {
            op[c] = abuf[c] + fbuf[c] * vlast;
            gtail[c] = fbuf[c];
        }
    }
}

int main(int argc, char**argv)
{
    const int C = argc>1?atoi(argv[1]):32;
    const int stages = argc>2?atoi(argv[2]):8;
    const int n = argc>3?atoi(argv[3]):4096;
    float32_t *kc = malloc((size_t)stages*4), *vc = malloc((size_t)(stages+1)*4);
    for (int i=0;i<stages;i++)   kc[i] = 0.3f * (float32_t)((i%5)+1) / 5.0f;
    for (int i=0;i<=stages;i++)  vc[i] = 0.2f * (float32_t)((i%3)+1);

    float32_t *xi = malloc((size_t)n*C*4), *o_b = malloc((size_t)n*C*4);
    float32_t *o_c = malloc((size_t)n*C*4);
    float32_t *chan = malloc((size_t)n*4), *ochan = malloc((size_t)n*4);
    for (int i=0;i<n*C;i++) xi[i]=(float32_t)((i%97)-48)/48.0f;
    float32_t *gp1 = calloc((size_t)(stages+1)*C, 4);
    float32_t *fbuf = calloc((size_t)C, 4), *abuf = calloc((size_t)C, 4);
    float32_t *sc = calloc((size_t)(stages+n), 4);

    double bc=1e9, bb=1e9;
    for (int r=0;r<5;r++){
        double t=now();
        for (int c=0;c<C;c++){
            for (int i=0;i<n;i++) chan[i]=xi[(size_t)i*C+c];
            arm_iir_lattice_instance_f32 S;
            memset(sc,0,(size_t)(stages+n)*4);
            arm_iir_lattice_init_f32(&S,(uint16_t)stages,kc,vc,sc,(uint32_t)n);
            arm_iir_lattice_f32(&S,chan,ochan,(uint32_t)n);
            for (int i=0;i<n;i++) o_c[(size_t)i*C+c]=ochan[i];
        }
        double d=now()-t; if(d<bc)bc=d;
        memset(gp1,0,(size_t)(stages+1)*C*4);
        t=now(); batched_lattice(kc,vc,stages,gp1,fbuf,abuf,xi,o_b,n,C);
        d=now()-t; if(d<bb)bb=d;
    }
    double worst=0, scale=0;
    for (size_t i=0;i<(size_t)n*C;i++){ double e=fabs(o_b[i]-o_c[i]);
        if (e>worst) worst=e; if (fabs(o_c[i])>scale) scale=fabs(o_c[i]); }
    if (!(worst < 1e-4*(scale+1.0))) {
        printf("MISMATCH abs=%.3e (scale %.3e)\n", worst, scale);
        for (int i=0;i<4;i++) printf("  t=%d cmsis=%.7f batched=%.7f\n",
                                     i, o_c[(size_t)i*C], o_b[(size_t)i*C]);
        return 1; }
    /* FOOTPRINT, for C channels streaming continuously, in bytes.
     *
     * Three numbers, because one of them flatters us and saying so is the
     * only way the other two get believed.
     *
     * cmsis_api    what arm_iir_lattice_init_f32 actually requires: pState of
     *              numStages+blockSize per instance, and a streaming
     *              multi-channel system needs one instance per channel, so the
     *              blockSize part is paid C times. Plus a de/interleave pair
     *              if acquisition is interleaved, which it usually is.
     *
     * cmsis_best   what a determined integrator could get to: only numStages
     *              per channel is genuinely persistent (CMSIS copies it back
     *              to the head of the buffer at the end of each call), so with
     *              enough copying in and out one blockSize scratch could be
     *              shared across channels. The API does not offer this and the
     *              copies are not free, but it is the fair lower bound and the
     *              honest comparison.
     *
     * ours         one (numStages+1) x C state buffer and two C-length
     *              temporaries. No de/interleave: it consumes the interleaved
     *              block directly.
     */
    const double cmsis_api  = (double)C * (stages + n) * 4.0 + 2.0 * n * 4.0;
    const double cmsis_best = (double)C * stages * 4.0 + (double)n * 4.0
                            + 2.0 * n * 4.0;
    const double ours_bytes = (double)(stages + 1) * C * 4.0 + 2.0 * C * 4.0;
    printf("{\"cmsis_api_bytes\": %.0f, \"cmsis_best_bytes\": %.0f, "
           "\"ours_bytes\": %.0f, \"vs_api\": %.3f, \"vs_best\": %.3f, ",
           cmsis_api, cmsis_best, ours_bytes,
           ours_bytes / cmsis_api, ours_bytes / cmsis_best);
    printf("\"channels\": %d, \"stages\": %d, \"n\": %d, "
           "\"cmsis_ns_per_sample\": %.3f, \"batched_ns_per_sample\": %.3f, "
           "\"speedup\": %.2f, \"max_abs_diff\": %.1e}\n",
           C, stages, n, bc/((double)n*C)*1e9, bb/((double)n*C)*1e9, bc/bb, worst);
    return 0;
}
