/* What is the disabled NEON df2T path actually worth, single channel?
 * The maintainer's stated reason for leaving it off (#186) was that "the
 * performance improvement is also not great". This measures it. */
#include "arm_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void arm_biquad_cascade_df2T_f32_neon(const arm_biquad_cascade_df2T_instance_f32*,
                                      const float32_t*, float32_t*, uint32_t);
static double now(void){return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW)*1e-9;}
static int cmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;
    return x<y?-1:x>y?1:0;}
int main(int argc,char**argv){
    const int stages=argc>1?atoi(argv[1]):4, n=argc>2?atoi(argv[2]):256,
              reps=argc>3?atoi(argv[3]):400;
    float coef[5*16],comp[8*16*4]={0};
    for(int s=0;s<stages;s++){ float k=1.0f+0.37f*s;
        coef[5*s+0]=0.0201f*k; coef[5*s+1]=0.0402f/k; coef[5*s+2]=0.0151f*k;
        coef[5*s+3]=1.5610f-0.03f*s; coef[5*s+4]=-0.6414f+0.02f*s; }
    arm_biquad_cascade_df2T_compute_coefs_f32((uint8_t)stages,coef,comp);
    float *in=malloc((size_t)n*4),*o=malloc((size_t)n*4);
    for(int i=0;i<n;i++) in[i]=(float)((i%17)-8)/8.0f;
    float st1[8*16],st2[8*16];
    double *ts=malloc((size_t)reps*8),*tn=malloc((size_t)reps*8);
    for(int r=0;r<reps;r++){
        memset(st1,0,sizeof st1);
        arm_biquad_cascade_df2T_instance_f32 S1={(uint8_t)stages,st1,coef};
        double t=now(); arm_biquad_cascade_df2T_f32(&S1,in,o,(uint32_t)n); ts[r]=now()-t;
        memset(st2,0,sizeof st2);
        arm_biquad_cascade_df2T_instance_f32 S2={(uint8_t)stages,st2,comp};
        t=now(); arm_biquad_cascade_df2T_f32_neon(&S2,in,o,(uint32_t)n); tn[r]=now()-t;
    }
    qsort(ts,(size_t)reps,8,cmp); qsort(tn,(size_t)reps,8,cmp);
    printf("stages=%-2d n=%-4d  scalar %7.3f ns/sample   NEON %7.3f ns/sample   NEON is %.2fx\n",
           stages,n, ts[reps/2]*1e9/n, tn[reps/2]*1e9/n, ts[reps/2]/tn[reps/2]);
    return 0;
}
