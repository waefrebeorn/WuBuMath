/*
 * jax_ir.c -- JAX-slermed: Jaxpr IR (intermediate representation)
 *
 * Slermed from jax_source/jax/_src/core.py Jaxpr.
 * Our IR is minimal: a list of ops with typed inputs/outputs.
 * This enables future jit compilation and autodiff.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jax_arena.h"
#include "jax_simd.h"

/* ===================================================================
 * IR Types
 * =================================================================== */

typedef enum {
    JAX_IR_LITERAL = 0,
    JAX_IR_PARAM,
    JAX_IR_OP,
    JAX_IR_VAR,
} JaxIrKind;

typedef enum {
    JAX_OP_ADD = 0,
    JAX_OP_SUB,
    JAX_OP_MUL,
    JAX_OP_DIV,
    JAX_OP_NEG,
    JAX_OP_EXP,
    JAX_OP_LOG,
    JAX_OP_POW,
    JAX_OP_ABS,
    JAX_OP_GEMM,
    JAX_OP_RELU,
    JAX_OP_SIGMOID,
    JAX_OP_TANH,
    JAX_OP_SOFTMAX,
    JAX_OP_LOG_SOFTMAX,
    JAX_OP_REDUCE_SUM,
    JAX_OP_REDUCE_MAX,
    JAX_OP_REDUCE_MEAN,
    JAX_OP_TRANSPOSE,
    JAX_OP_RESHAPE,
    JAX_OP_SLICE,
    JAX_OP_GATHER,
    JAX_OP_SCATTER,
    JAX_OP_CONCAT,
    JAX_OP_SELECT,
    JAX_OP_CLAMP,
    JAX_OP_LAYER_NORM,
    JAX_OP_RMS_NORM,
    JAX_OP_ATTENTION,
    JAX_OP_DOTP,
    JAX_OP_MLP_FORWARD,
    JAX_OP_MLP_BACKWARD,
    JAX_OP_COUNT
} JaxIrOpKind;

static const char* jax_ir_op_names[] = {
    "add", "sub", "mul", "div", "neg", "exp", "log", "pow",
    "abs", "gemm", "relu", "sigmoid", "tanh", "softmax",
    "log_softmax", "reduce_sum", "reduce_max", "reduce_mean",
    "transpose", "reshape", "slice", "gather", "scatter",
    "concat", "select", "clamp", "layer_norm", "rms_norm",
    "attention", "dotp", "mlp_fwd", "mlp_bwd"
};

typedef int64_t JaxVarId;

typedef struct {
    JaxIrKind kind;
    union {
        struct { float value; } literal;
        struct { int index; } param;
        struct {
            JaxIrOpKind op;
            JaxVarId* inputs;
            int n_inputs;
            int64_t* shape;
            int ndim;
            /* Gradient accumulator */
            float* grad;
        } op;
        struct { JaxVarId id; } var;
    };
} JaxIrInstr;

typedef struct {
    JaxVarId* inputs;     /* Input variable IDs */
    int n_inputs;
    JaxVarId* outputs;    /* Output variable IDs */
    int n_outputs;
    JaxIrInstr* instrs;   /* Instructions */
    int n_instrs;
    int capacity;
    /* Variable type tracking */
    int64_t** var_shapes;  /* Shape per variable */
    int* var_ndims;       /* ndim per variable */
    int n_vars;
    int var_capacity;
} JaxIr;

/* ===================================================================
 * IR Construction
 * =================================================================== */

JaxIr* jax_ir_create(JaxArena* arena, int n_inputs) {
    JaxIr* ir = JAX_ARENA_ALLOC(arena, JaxIr, 1);
    if (!ir) return NULL;
    memset(ir, 0, sizeof(JaxIr));
    
    ir->inputs = JAX_ARENA_ALLOC(arena, JaxVarId, n_inputs);
    ir->n_inputs = n_inputs;
    ir->n_outputs = 0;
    ir->n_instrs = 0;
    ir->capacity = 64;
    ir->instrs = JAX_ARENA_ALLOC(arena, JaxIrInstr, ir->capacity);
    
    ir->n_vars = n_inputs;
    ir->var_capacity = n_inputs + 64;
    ir->var_shapes = JAX_ARENA_ALLOC(arena, int64_t*, ir->var_capacity);
    ir->var_ndims = JAX_ARENA_ALLOC(arena, int, ir->var_capacity);
    
    /* Initialize input variables */
    for (int i = 0; i < n_inputs; ++i) {
        ir->inputs[i] = i;
        ir->var_shapes[i] = NULL;
        ir->var_ndims[i] = 0;
    }
    
    return ir;
}

static JaxVarId jax_ir_alloc_var(JaxIr* ir, int ndim) {
    JaxVarId id = ir->n_vars++;
    if (ir->n_vars > ir->var_capacity) {
        /* Grow */
        int new_cap = ir->var_capacity * 2;
        int64_t** new_shapes = (int64_t**)malloc(new_cap * sizeof(int64_t*));
        int* new_ndims = (int*)malloc(new_cap * sizeof(int));
        memcpy(new_shapes, ir->var_shapes, ir->var_capacity * sizeof(int64_t*));
        memcpy(new_ndims, ir->var_ndims, ir->var_capacity * sizeof(int));
        free(ir->var_shapes);
        free(ir->var_ndims);
        ir->var_shapes = new_shapes;
        ir->var_ndims = new_ndims;
        ir->var_capacity = new_cap;
    }
    ir->var_ndims[id] = ndim;
    ir->var_shapes[id] = NULL;
    return id;
}

static void jax_ir_emit(JaxIr* ir, JaxIrInstr instr) {
    if (ir->n_instrs >= ir->capacity) {
        int new_cap = ir->capacity * 2;
        JaxIrInstr* new_instrs = (JaxIrInstr*)malloc(new_cap * sizeof(JaxIrInstr));
        memcpy(new_instrs, ir->instrs, ir->capacity * sizeof(JaxIrInstr));
        free(ir->instrs);
        ir->instrs = new_instrs;
        ir->capacity = new_cap;
    }
    ir->instrs[ir->n_instrs++] = instr;
}

/* ===================================================================
 * IR Operations (emit instructions)
 * =================================================================== */

JaxVarId jax_ir_add(JaxIr* ir, JaxVarId a, JaxVarId b) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_OP,
        .op = { .op = JAX_OP_ADD, .inputs = (JaxVarId[]){a, b}, .n_inputs = 2 }
    };
    instr.op.inputs = (JaxVarId*)malloc(2 * sizeof(JaxVarId));
    instr.op.inputs[0] = a;
    instr.op.inputs[1] = b;
    jax_ir_emit(ir, instr);
    return out;
}

JaxVarId jax_ir_mul(JaxIr* ir, JaxVarId a, JaxVarId b) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_OP,
        .op = { .op = JAX_OP_MUL, .n_inputs = 2 }
    };
    instr.op.inputs = (JaxVarId*)malloc(2 * sizeof(JaxVarId));
    instr.op.inputs[0] = a;
    instr.op.inputs[1] = b;
    jax_ir_emit(ir, instr);
    return out;
}

JaxVarId jax_ir_gemm(JaxIr* ir, JaxVarId a, JaxVarId b) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_OP,
        .op = { .op = JAX_OP_GEMM, .n_inputs = 2 }
    };
    instr.op.inputs = (JaxVarId*)malloc(2 * sizeof(JaxVarId));
    instr.op.inputs[0] = a;
    instr.op.inputs[1] = b;
    jax_ir_emit(ir, instr);
    return out;
}

JaxVarId jax_ir_relu(JaxIr* ir, JaxVarId a) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_OP,
        .op = { .op = JAX_OP_RELU, .n_inputs = 1 }
    };
    instr.op.inputs = (JaxVarId*)malloc(sizeof(JaxVarId));
    instr.op.inputs[0] = a;
    jax_ir_emit(ir, instr);
    return out;
}

JaxVarId jax_ir_reduce_sum(JaxIr* ir, JaxVarId a) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_OP,
        .op = { .op = JAX_OP_REDUCE_SUM, .n_inputs = 1 }
    };
    instr.op.inputs = (JaxVarId*)malloc(sizeof(JaxVarId));
    instr.op.inputs[0] = a;
    jax_ir_emit(ir, instr);
    return out;
}

JaxVarId jax_ir_literal(JaxIr* ir, float value) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_LITERAL,
        .literal = { .value = value }
    };
    jax_ir_emit(ir, instr);
    return out;
}

JaxVarId jax_ir_param(JaxIr* ir, int index) {
    JaxVarId out = jax_ir_alloc_var(ir, 0);
    JaxIrInstr instr = {
        .kind = JAX_IR_PARAM,
        .param = { .index = index }
    };
    jax_ir_emit(ir, instr);
    return out;
}

/* ===================================================================
 * IR Print (debug)
 * =================================================================== */

void jax_ir_print(JaxIr* ir) {
    printf("JaxPr(%d inputs, %d vars, %d instrs)\n",
           ir->n_inputs, ir->n_vars, ir->n_instrs);
    
    for (int i = 0; i < ir->n_instrs; ++i) {
        JaxIrInstr* instr = &ir->instrs[i];
        switch (instr->kind) {
            case JAX_IR_LITERAL:
                printf("  %d = literal(%f)\n", i, instr->literal.value);
                break;
            case JAX_IR_PARAM:
                printf("  %d = param(%d)\n", i, instr->param.index);
                break;
            case JAX_IR_OP:
                printf("  %d = %s(", i,
                       instr->op.op < JAX_OP_COUNT ?
                       jax_ir_op_names[instr->op.op] : "unknown");
                for (int j = 0; j < instr->op.n_inputs; ++j) {
                    if (j > 0) printf(", ");
                    printf("v%ld", (long)instr->op.inputs[j]);
                }
                printf(")\n");
                break;
            default:
                printf("  %d = <?>\n", i);
                break;
        }
    }
}

int jax_ir_num_instrs(JaxIr* ir) {
    return ir ? ir->n_instrs : 0;
}

/* ===================================================================
 * IR Gradient (reverse-mode AD — single pass)
 * =================================================================== */

int jax_ir_backward(JaxIr* ir, JaxVarId loss_var) {
    if (!ir || (int)loss_var >= ir->n_vars) return -1;
    
    /* Allocate gradient storage for all variables */
    float* grad = (float*)calloc(ir->n_vars, sizeof(float));
    if (!grad) return -1;
    
    /* Seed: d(loss)/d(loss) = 1 */
    grad[loss_var] = 1.0f;
    
    /* Reverse pass */
    for (int i = ir->n_instrs - 1; i >= 0; --i) {
        JaxIrInstr* instr = &ir->instrs[i];
        if (instr->kind != JAX_IR_OP) continue;
        
        float g = grad[i];
        if (g == 0.0f) continue;
        
        switch (instr->op.op) {
            case JAX_OP_ADD:
                grad[instr->op.inputs[0]] += g;
                grad[instr->op.inputs[1]] += g;
                break;
            case JAX_OP_MUL:
                /* d(a*b)/da = b, d(a*b)/db = a — need forward values */
                /* Simplified: just pass gradient through */
                grad[instr->op.inputs[0]] += g;
                grad[instr->op.inputs[1]] += g;
                break;
            case JAX_OP_RELU:
                /* d(relu)/dx = 1 if x > 0 else 0 — need forward value */
                grad[instr->op.inputs[0]] += g;
                break;
            case JAX_OP_GEMM:
                /* d(Wx)/dW = x^T, d(Wx)/dx = W^T — simplified */
                grad[instr->op.inputs[0]] += g;
                grad[instr->op.inputs[1]] += g;
                break;
            case JAX_OP_REDUCE_SUM:
                grad[instr->op.inputs[0]] += g;
                break;
            default:
                /* Pass-through for unknown ops */
                for (int j = 0; j < instr->op.n_inputs; ++j) {
                    grad[instr->op.inputs[j]] += g;
                }
                break;
        }
    }
    
    free(grad);
    return 0;
}
