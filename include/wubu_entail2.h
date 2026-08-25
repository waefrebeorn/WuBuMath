/* GAP-D030: entailment cone (Ganea 2018 explicit formula) */
#ifndef WUBU_ENTAIL2_H
#define WUBU_ENTAIL2_H
#ifdef __cplusplus
extern "C" {
#endif
/* optimal cone aperture (full angle) at parent point p */
float wubu_ec_aperture(const float* p,int D,float c);
/* 1 if child is inside parent's entailment cone */
int wubu_ec_entailed(const float* parent,const float* child,
                      int D,float c);
#ifdef __cplusplus
}
#endif
#endif
