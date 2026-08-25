/* GAP-C063: Screw Linear Interpolation for dual quaternions */
#ifndef WUBU_SCLERP_H
#define WUBU_SCLERP_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_sclerp(const float* qa,const float* qb,float t,float* out);
#ifdef __cplusplus
}
#endif
#endif
