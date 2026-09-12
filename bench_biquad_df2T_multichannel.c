/* Multi-channel biquad, transposed Direct Form II: CMSIS-DSP per channel
 * versus channel-innermost.
 *
 * df2T is the form CMSIS provides for f32 alongside df1, and in this kernel the
 * NEON path is disabled in source -- arm_biquad_cascade_df2T_f32.c:187 reads
 * `#if 0 //defined(ARM_MATH_NEON)` -- while the Helium path sits behind
 * ARM_MATH_HELIUM_EXPERIMENTAL. So on a NEON target the scalar loop is what
 * runs.
 *
 * The recurrence (from the reference implementation) is serial in time:
 *     acc = b0*x[n] + d1
 *     d1  = b1*x[n] + a1*acc + d2
 *     d2  = b2*x[n] + a2*acc
 * d1 and d2 carry to the next sample, so there is nothing to vectorise within a
 * channel. Channels are independent.
 */
#include "arm_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){ return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9; }

/* Channel index innermost: x[t*C + c]. The inner loop is pure vector work. */
static void batched_df2T(const float32_t *coef, float32_t *st, int stages,
                         const float32_t *in, float32_t *out, int n, int C)
{
    for (int s = 0; s < stages; s++) {
        const float32_t b0=coef[5*s+0], b1=coef[5*s+1], b2=coef[5*s+2];
        const float32_t a1=coef[5*s+3], a2=coef[5*s+4];
        float32_t *d1 = st + (size_t)(2*s+0)*C;
        float32_t *d2 = st + (size_t)(2*s+1)*C;
        const float32_t *src = (s==0)? in : out;
        for (int t = 0; t < n; t++) {
            const float32_t *sp = src + (size_t)t*C;
            float32_t *op = out + (size_t)t*C;
            for (int c = 0; c < C; c++) {            /* independent lanes */
                const float32_t x = sp[c];
                const float32_t acc = b0*x + d1[c];
                d1[c] = b1*x + a1*acc + d2[c];
                d2[c] = b2*x + a2*acc;
                op[c] = acc;
            }
        }
    }
}

int main(int argc, char**argv)
{
    const int C = argc>1?atoi(argv[1]):32;
    const int stages = argc>2?atoi(argv[2]):4;
    const int n = argc>3?atoi(argv[3]):128;
    float32_t *coef = malloc((size_t)5*stages*4);
    for (int s=0;s<stages;s++){ coef[5*s+0]=0.0201f; coef[5*s+1]=0.0402f;
        coef[5*s+2]=0.0201f; coef[5*s+3]=1.5610f; coef[5*s+4]=-0.6414f; }

    float32_t *xi=malloc((size_t)n*C*4), *o_b=malloc((size_t)n*C*4);
    float32_t *o_c=malloc((size_t)n*C*4);
    float32_t *cmaj=malloc((size_t)n*C*4), *cmajo=malloc((size_t)n*C*4);
    float32_t *chan=malloc((size_t)n*4), *ochan=malloc((size_t)n*4);
    for (int i=0;i<n*C;i++) xi[i]=(float32_t)((i%97)-48)/48.0f;
    for (int c=0;c<C;c++) for (int i=0;i<n;i++) cmaj[(size_t)c*n+i]=xi[(size_t)i*C+c];
    float32_t *stb=calloc((size_t)2*stages*C,4), *stc=calloc((size_t)2*stages,4);

    double bi=1e9, bm=1e9, bb=1e9;
    for (int r=0;r<9;r++){
        /* CMSIS, interleaved source: pays de/interleave */
        double t=now();
        for (int c=0;c<C;c++){
            for (int i=0;i<n;i++) chan[i]=xi[(size_t)i*C+c];
            arm_biquad_cascade_df2T_instance_f32 S;
            memset(stc,0,(size_t)2*stages*4);
            arm_biquad_cascade_df2T_init_f32(&S,(uint8_t)stages,coef,stc);
            arm_biquad_cascade_df2T_f32(&S,chan,ochan,(uint32_t)n);
            for (int i=0;i<n;i++) o_c[(size_t)i*C+c]=ochan[i];
        }
        double d=now()-t; if(d<bi)bi=d;

        /* CMSIS at its best: channel-major already, no transpose charged */
        t=now();
        for (int c=0;c<C;c++){
            arm_biquad_cascade_df2T_instance_f32 S2;
            memset(stc,0,(size_t)2*stages*4);
            arm_biquad_cascade_df2T_init_f32(&S2,(uint8_t)stages,coef,stc);
            arm_biquad_cascade_df2T_f32(&S2,cmaj+(size_t)c*n,cmajo+(size_t)c*n,(uint32_t)n);
        }
        d=now()-t; if(d<bm)bm=d;

        memset(stb,0,(size_t)2*stages*C*4);
        t=now(); batched_df2T(coef,stb,stages,xi,o_b,n,C); d=now()-t; if(d<bb)bb=d;
    }
    double worst=0;
    for (size_t i=0;i<(size_t)n*C;i++){ double e=fabs(o_b[i]-o_c[i]);
        if(e>worst)worst=e; }
    if (!(worst < 1e-4)) { printf("MISMATCH %.3e\n", worst); return 1; }

    /* Footprint, C channels streaming. Both carry 2 floats per stage per
     * channel; the only difference is CMSIS's de/interleave scratch. */
    const double cm_api = 2.0*stages*C*4.0 + 2.0*n*4.0;
    const double cm_best= 2.0*stages*C*4.0;
    const double ours   = 2.0*stages*C*4.0;
    printf("{\"channels\": %d, \"stages\": %d, \"n\": %d, "
           "\"cmsis_interleaved\": %.3f, \"cmsis_channel_major\": %.3f, "
           "\"ours\": %.3f, \"vs_interleaved\": %.2f, \"vs_channel_major\": %.2f, "
           "\"cmsis_api_bytes\": %.0f, \"cmsis_best_bytes\": %.0f, "
           "\"ours_bytes\": %.0f, \"mem_vs_best\": %.3f, \"max_abs_diff\": %.1e}\n",
           C,stages,n, bi/((double)n*C)*1e9, bm/((double)n*C)*1e9,
           bb/((double)n*C)*1e9, bi/bb, bm/bb,
           cm_api, cm_best, ours, ours/cm_best, worst);
    return 0;
}
