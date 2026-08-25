/* GAP-C004: exp/log-map flow integration */
#ifndef WUBU_FLOW_EXP_H
#define WUBU_FLOW_EXP_H
#ifdef __cplusplus
extern "C" {
#endif
int wubu_fe_step(const float* x,const float* velocity,int D,
                  float c,float dt,float* out);
int wubu_fe_trajectory(const float* start,const float* end,int D,
                        float c,int n_steps,float lr,float* out_path);
#ifdef __cplusplus
}
#endif
#endif
