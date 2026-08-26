/*
 * wubu_hgnn.c -- GROUP 19: Hyperbolic Graph Neural Network (HGCN-style)
 *
 * Lorentz-model graph convolution with trainable curvature.
 * Aggregation via Lorentzian centroid, feature transform via
 * hyperbolic linear layers, all operations stay on the hyperboloid.
 *
 * Research source: Chami et al. 2019 (HGCN), Liu et al. 2019 (HGNN)
 */
#define M_PI 3.14159265358979f
#include "wubu_hgnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Lorentz model helpers ===== */

/* Minkowski inner product: <x,y> = -x0*y0 + x1*y1 + ... + xn*yn */
double wubu_hgnn_lorentz_dot(const double* a,const double* b,int dim){
    double sum=-a[0]*b[0];
    for(int i=1;i<dim;i++)sum+=a[i]*b[i];
    return sum;
}

/* distance on hyperboloid: d(x,y) = arcosh(-<x,y>) */
double wubu_hgnn_distance(const double* x,const double* y,int dim){
    double ip=wubu_hgnn_lorentz_dot(x,y,dim);
    if(ip>-1)ip=-1; /* clamp for numerical safety */
    return acosh(-ip);
}

/* project to hyperboloid: ensure L(p,p) = -1/c² */
void wubu_hgnn_project(double* p,int dim,double c){
    /* rescale so that -p[0]²+Σp[i]² = -1/c */
    double spatial_norm=0;
    for(int i=1;i<dim;i++)spatial_norm+=p[i]*p[i];
    spatial_norm=sqrt(spatial_norm);
    
    if(spatial_norm<1e-10){
        p[0]=1.0/sqrt(c);
        for(int i=1;i<dim;i++)p[i]=1e-6;
        return;
    }
    
    double time_coord=sqrt(1.0/c+spatial_norm*spatial_norm);
    p[0]=time_coord;
}

/* exp map at origin (Lorentz model) */
void wubu_hgnn_exp_origin(const double* v,double* out,int dim,double c){
    /* v is in tangent space at origin o=(1/sqrt(c), 0,...,0) */
    double norm=0;
    for(int i=1;i<dim;i++)norm+=v[i]*v[i];
    norm=sqrt(norm);
    
    if(norm<1e-10){
        out[0]=1.0/sqrt(c);
        for(int i=1;i<dim;i++)out[i]=0;
        return;
    }
    
    double sqrt_c=sqrt(c);
    out[0]=cosh(sqrt_c*norm)/sqrt_c;
    for(int i=1;i<dim;i++)
        out[i]=sinh(sqrt_c*norm)*v[i]/(sqrt_c*norm);
}

/* log map at origin */
void wubu_hgnn_log_origin(const double* p,double* out,int dim,double c){
    double sqrt_c=sqrt(c);
    double alpha=c*sqrt_c*p[0]; /* = c^(3/2) * p_0 */
    
    double spatial_norm=0;
    for(int i=1;i<dim;i++)spatial_norm+=p[i]*p[i];
    spatial_norm=sqrt(spatial_norm);
    
    if(spatial_norm<1e-10){
        for(int i=0;i<dim;i++)out[i]=0;
        return;
    }
    
    out[0]=0;
    for(int i=1;i<dim;i++)
        out[i]=acosh(sqrt_c*p[0])*p[i]/(sqrt_c*spatial_norm);
}

/* ===== Lorentzian Centroid (Fréchet mean approximation) ===== */

int wubu_hgnn_centroid(const double* points,int n_points,int dim,
                         double c,double* centroid){
    if(n_points==0)return -1;
    
    /* simple approach: average in tangent space then map back */
    double* tangent=calloc((size_t)dim,sizeof(double));
    
    /* project each point to tangent space at origin and average */
    for(int i=0;i<n_points;i++){
        double tmp[64];
        wubu_hgnn_log_origin(points+(size_t)i*dim,tmp,dim,c);
        for(int d=0;d<dim;d++)tangent[d]+=tmp[d]/n_points;
    }
    
    wubu_hgnn_exp_origin(tangent,centroid,dim,c);
    free(tangent);
    return 0;
}

/* ===== Hyperbolic Graph Convolution Layer ===== */

/*
 * HGCN layer:
 * 1. Aggregate neighbor features via Lorentzian centroid
 * 2. Combine with self features (weighted)
 * 3. Apply hyperbolic linear layer
 * 4. Project back to hyperboloid
 */
int wubu_hgnn_layer(const double* input_features,const int* adj_matrix,
                      int n_nodes,int n_edges,int feature_dim,
                      const double* weight, double bias,
                      double c,double* output_features){
    /* for each node:
     *   1. aggregate neighbors → centroid
     *   2. combine self + aggregated
     *   3. apply linear transform in tangent space
     *   4. project back to hyperboloid
     */
    double* agg=calloc((size_t)n_nodes*(size_t)feature_dim,sizeof(double));
    
    for(int node=0;node<n_nodes;node++){
        /* collect neighbor indices from adjacency matrix */
        int* neighbors=malloc(sizeof(int)*(size_t)n_nodes);
        int n_neighbors=0;
        
        for(int e=0;e<n_edges;e++){
            if(adj_matrix[e*2]==node){
                neighbors[n_neighbors++]=adj_matrix[e*2+1];
            }
            if(adj_matrix[e*2+1]==node){
                neighbors[n_neighbors++]=adj_matrix[e*2];
            }
        }
        
        if(n_neighbors==0){
            memcpy(agg+(size_t)node*feature_dim,
                   input_features+(size_t)node*feature_dim,
                   sizeof(double)*(size_t)feature_dim);
            free(neighbors);continue;
        }
        
        /* gather neighbor features */
        double* neigh_feats=malloc(sizeof(double)*(size_t)n_neighbors*feature_dim);
        for(int i=0;i<n_neighbors;i++)
            memcpy(neigh_feats+(size_t)i*feature_dim,
                   input_features+(size_t)neighbors[i]*feature_dim,
                   sizeof(double)*(size_t)feature_dim);
        
        /* Lorentzian centroid of neighbors */
        wubu_hgnn_centroid(neigh_feats,n_neighbors,feature_dim,c,
                            agg+(size_t)node*feature_dim);
        free(neigh_feats);free(neighbors);
    }
    
    /* combine self + aggregate, apply linear, re-project */
    for(int node=0;node<n_nodes;node++){
        double combined[64];
        
        /* average self and aggregate in tangent space */
        double t_self[64],t_agg[64];
        wubu_hgnn_log_origin(input_features+(size_t)node*feature_dim,t_self,feature_dim,c);
        wubu_hgnn_log_origin(agg+(size_t)node*feature_dim,t_agg,feature_dim,c);
        
        for(int d=0;d<feature_dim;d++)
            combined[d]=(t_self[d]+t_agg[d])/2+bias;
        
        /* linear transform in tangent space */
        double transformed[64];
        for(int d=0;d<feature_dim;d++){
            double sum=0;
            for(int k=0;k<feature_dim;k++)
                sum+=combined[k]*weight[k*feature_dim+d];
            transformed[d]=sum;
        }
        
        /* project back to hyperboloid */
        wubu_hgnn_project(output_features+(size_t)node*feature_dim,feature_dim,c);
        wubu_hgnn_exp_origin(transformed,output_features+(size_t)node*feature_dim,feature_dim,c);
    }
    
    free(agg);
    return 0;
}
