/* Reproduce CMSIS-DSP issue #231: the NEON arm_biquad_cascade_df2T_f32
 * disagrees with the scalar one. The NEON path is disabled in the shipped
 * library; this build re-enables it and compares the two on the same filter.
 *
 * Per the docs and the maintainer's comment on #231, the NEON path does not
 * take the coefficients directly -- they must be passed through
 * arm_biquad_cascade_df2T_compute_coefs_f32 first. That is done here, so a
 * disagreement is not simply the documented usage mistake.
 */
#include "arm_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void arm_biquad_cascade_df2T_f32_neon(const arm_biquad_cascade_df2T_instance_f32*,
                                      const float32_t*, float32_t*, uint32_t);
int main(int argc,char**argv){
    const int stages = argc>1?atoi(argv[1]):4;
    const int n = argc>2?atoi(argv[2]):32;
    float coef[5*16], comp[8*16*4]={0};
    /* DISTINCT coefficients per stage. Identical stages would mask an index
     * error in the cascade-gain products the NEON layout precomputes
     * (b0[1]*b0[2], b0[2]*b0[3], ...), which is exactly where a bug would hide. */
    for(int s=0;s<stages;s++){
        float k=1.0f+0.37f*(float)s;
        coef[5*s+0]=0.0201f*k; coef[5*s+1]=0.0402f/k; coef[5*s+2]=0.0151f*k;
        coef[5*s+3]=1.5610f-0.03f*s; coef[5*s+4]=-0.6414f+0.02f*s; }
    float *in=malloc(n*4),*o_s=malloc(n*4),*o_n=malloc(n*4);
    for(int i=0;i<n;i++) in[i]=(float)((i%17)-8)/8.0f;

    /* scalar reference: the shipped behaviour */
    float st1[8*16]={0};
    arm_biquad_cascade_df2T_instance_f32 S1={(uint8_t)stages,st1,coef};
    arm_biquad_cascade_df2T_f32(&S1,in,o_s,(uint32_t)n);

    /* NEON: coefficients through compute_coefs, as documented */
    arm_biquad_cascade_df2T_compute_coefs_f32((uint8_t)stages,coef,comp);
    float st2[8*16]={0};
    arm_biquad_cascade_df2T_instance_f32 S2={(uint8_t)stages,st2,comp};
    arm_biquad_cascade_df2T_f32_neon(&S2,in,o_n,(uint32_t)n);

    double worst=0,scale=0;
    for(int i=0;i<n;i++){ double d=fabs(o_s[i]-o_n[i]); if(d>worst)worst=d;
                          if(fabs(o_s[i])>scale)scale=fabs(o_s[i]); }
    printf("stages=%d n=%d  max|scalar-neon| = %.4e  (signal scale %.3e)\n",
           stages,n,worst,scale);
    if (worst > 1e-4*(scale+1.0)) {
        printf("  DIVERGES\n");
        for(int i=0;i<4&&i<n;i++) printf("    %2d  scalar %10.6f   neon %10.6f\n",
                                         i,o_s[i],o_n[i]);
    }
    return 0;
}
