# WuBuMath Makefile
# Pure C11 mathematical & media encoding library

CC = gcc
CFLAGS = -std=c11 -O3 -march=native -Iinclude -w
LDFLAGS = -lm -lpthread

SRC = src
INC = include
OBJ = build
BIN = bin

SRCS = $(SRC)/math/wubu_color.c \
       $(SRC)/math/wubu_positional_encode.c \
       $(SRC)/math/wubu_utils.c \
       $(SRC)/math/wubu_hyperbolic.c \
       $(SRC)/math/wubu_quaternion.c \
       $(SRC)/model/wubu_hamilton_encoder.c \
       $(SRC)/model/wubu_vhf_decoder.c \
       $(SRC)/model/wubu_vhf_audio.c \
       $(SRC)/model/wubu_canvas.c \
       $(SRC)/model/wubu_nested_encoder.c \
       $(SRC)/train/wubu_q_controller.c \
       $(SRC)/train/wubu_latent_codec.c \
       $(SRC)/math/wubu_quaternion_ops.c \
       $(SRC)/math/wubu_parallel_transport.c \
       $(SRC)/math/wubu_so3.c \
       $(SRC)/math/wubu_rep_theory.c \
       $(SRC)/train/wubu_tangent_flow.c \
       $(SRC)/train/wubu_flow_matching.c \
       $(SRC)/train/wubu_riemannian_sgd.c \
       $(SRC)/train/wubu_loss.c \
       $(SRC)/nn/wubu_nn.c \

# Slermed JAX core
JAX_SRCS = $(SRC)/jax/jax_arena.c \
           $(SRC)/jax/jax_simd.c \
           $(SRC)/jax/jax_nn.c \
           $(SRC)/jax/jax_opt.c \
           $(SRC)/jax/jax_lax.c \
           $(SRC)/jax/jax_ir.c

JAX_OBJS = $(patsubst $(SRC)/jax/%.c,$(OBJ)/jax/%.o,$(JAX_SRCS))

TARGETS = $(BIN)/wubu_tests $(BIN)/jax_test $(BIN)/media_creator $(BIN)/nn_test \
          $(BIN)/test_hyperbolic $(BIN)/test_quaternion $(BIN)/test_riemannian_sgd \
          $(BIN)/test_parallel_transport $(BIN)/test_tangent_flow $(BIN)/test_nest_gpt

all: $(TARGETS)

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/jax/%.o: $(SRC)/jax/%.c
	@mkdir -p $(OBJ)/jax
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/examples/%.o: examples/%.c
	@mkdir -p $(OBJ)/examples
	$(CC) $(CFLAGS) -c $< -o $@

# WuBuMath tests
$(BIN)/wubu_tests: $(SRC)/tests/wubu_tests.c $(SRCS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

# Slermed JAX tests
$(BIN)/jax_test: $(SRC)/tests/jax_slermed_test.c $(JAX_SRCS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(JAX_SRCS) $< -o $@ $(LDFLAGS)

# Media creator (uses JAX core + math utils)
MEDIA_SRCS = $(SRC)/math/wubu_color.c \
             $(SRC)/math/wubu_positional_encode.c \
             $(SRC)/math/wubu_utils.c \
             $(SRC)/jax/jax_arena.c \
             $(SRC)/jax/jax_simd.c \
             $(SRC)/jax/jax_nn.c \
             $(SRC)/jax/jax_opt.c \
             $(SRC)/jax/jax_lax.c \
             $(SRC)/jax/jax_ir.c \
             $(SRC)/encoders/phase1.c

$(BIN)/media_creator: examples/media_creator.c $(MEDIA_SRCS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -DMEDIA_CREATOR_STANDALONE $(MEDIA_SRCS) $< -o $@ $(LDFLAGS)

# NN layer tests
$(BIN)/nn_test: $(SRC)/tests/test_wubu_nn.c $(SRCS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

# Hyperbolic geometry tests
$(BIN)/test_hyperbolic: $(SRC)/tests/test_wubu_hyperbolic.c $(SRC)/math/wubu_hyperbolic.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_hyperbolic.c $< -o $@ $(LDFLAGS)

# Quaternion tests
$(BIN)/test_quaternion: $(SRC)/tests/test_wubu_quaternion.c $(SRC)/math/wubu_quaternion.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_quaternion.c $< -o $@ $(LDFLAGS)

# SO(3) exp/log/geodesic port from libirrep (numerical validation)
$(BIN)/test_so3: $(SRC)/tests/test_wubu_so3.c $(SRC)/math/wubu_so3.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_so3.c $< -o $@ $(LDFLAGS)

# Rep-theory tests (Wigner 3j / Clebsch-Gordan, libirrep port)
$(BIN)/test_rep: $(SRC)/tests/test_wubu_rep_theory.c $(SRC)/math/wubu_rep_theory.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_rep_theory.c $< -o $@ $(LDFLAGS)

# Quaternion operations tests (Hamilton product, SLERP, Poincaré, etc.)
$(BIN)/test_quat_ops: $(SRC)/tests/test_wubu_quaternion_ops.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c -o $@ $(LDFLAGS)

# Nested encoder tests
$(BIN)/test_nested_enc: $(SRC)/tests/test_nested_encoder.c $(SRCS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

# Riemannian SGD tests
$(BIN)/test_riemannian_sgd: $(SRC)/tests/test_wubu_riemannian_sgd.c $(SRC)/train/wubu_riemannian_sgd.c \
                              $(SRC)/math/wubu_hyperbolic.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_riemannian_sgd.c $(SRC)/math/wubu_hyperbolic.c $< -o $@ $(LDFLAGS)

# Parallel transport tests
$(BIN)/test_parallel_transport: $(SRC)/tests/test_wubu_parallel_transport.c $(SRC)/math/wubu_parallel_transport.c \
                                 $(SRC)/math/wubu_hyperbolic.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_parallel_transport.c $(SRC)/math/wubu_hyperbolic.c $< -o $@ $(LDFLAGS)

# Tangent flow tests
$(BIN)/test_tangent_flow: $(SRC)/tests/test_wubu_tangent_flow.c $(SRC)/train/wubu_tangent_flow.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_tangent_flow.c $< -o $@ $(LDFLAGS)

# WuBuNestGPT tests
$(BIN)/test_nest_gpt: $(SRC)/tests/test_wubu_nest_gpt.c $(SRC)/model/wubu_nest_gpt.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/model/wubu_nest_gpt.c $< -o $@ $(LDFLAGS)

# VHF training
$(BIN)/train_vhf: examples/train_vhf.c $(SRC)/model/wubu_vhf_engine.c $(SRC)/train/wubu_loss.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_utils.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/model/wubu_vhf_engine.c $(SRC)/train/wubu_loss.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_utils.c -o $@ $(LDFLAGS)

# VHF end-to-end demo
$(BIN)/vhf_e2e_demo: examples/vhf_e2e_demo.c $(SRCS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

# Multi-resolution VHF demo
$(BIN)/vhf_multires_demo: examples/vhf_multires_demo.c $(SRCS) $(SRC)/model/wubu_canvas.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $(SRC)/model/wubu_canvas.c $< -o $@ $(LDFLAGS)

# Flow matching tests
$(BIN)/test_flow_matching: $(SRC)/tests/test_wubu_flow_matching.c $(SRC)/train/wubu_flow_matching.c \
                            $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_flow_matching.c $(SRC)/math/wubu_hyperbolic.c \
	    $(SRC)/math/wubu_quaternion.c $< -o $@ $(LDFLAGS)

# Latent codec tests
$(BIN)/test_latent_codec: $(SRC)/tests/test_wubu_latent_codec.c $(SRC)/train/wubu_latent_codec.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_latent_codec.c $< -o $@ $(LDFLAGS)

# Flow matching step-by-step demo
$(BIN)/flow_step_demo: examples/flow_matching_step_by_step.c $(SRCS) $(SRC)/model/wubu_canvas.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $(SRC)/model/wubu_canvas.c $< -o $@ $(LDFLAGS)

# 480P WuBu demo
$(BIN)/wubu_480p_demo: examples/wubu_480p_demo.c $(SRC)/train/wubu_latent_codec.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_latent_codec.c $< -o $@ $(LDFLAGS)

# WuBu video pipeline demo
$(BIN)/wubu_video_pipeline: examples/wubu_video_pipeline.c $(SRC)/train/wubu_latent_codec.c $(SRC)/train/wubu_flow_matching.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_latent_codec.c $(SRC)/train/wubu_flow_matching.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $< -o $@ $(LDFLAGS)

# Generate training data
$(BIN)/gen_train_data: examples/generate_training_data.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# WuBu training pipeline
$(BIN)/wubu_train: examples/wubu_train.c $(SRC)/train/wubu_flow_matching.c $(SRC)/train/wubu_latent_codec.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -DJPEG examples/wubu_train.c $(SRC)/train/wubu_flow_matching.c $(SRC)/train/wubu_latent_codec.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c -o $@ $(LDFLAGS) -ljpeg

# Canvas resolution tests
$(BIN)/test_canvas_res: $(SRC)/tests/test_canvas_resolutions.c $(SRC)/model/wubu_canvas.c $(SRC)/model/wubu_hamilton_encoder.c $(SRC)/train/wubu_q_controller.c $(SRC)/train/wubu_loss.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_color.c $(SRC)/model/wubu_vhf_audio.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/model/wubu_canvas.c $(SRC)/model/wubu_hamilton_encoder.c $(SRC)/train/wubu_q_controller.c $(SRC)/train/wubu_loss.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_color.c $(SRC)/model/wubu_vhf_audio.c -o $@ $(LDFLAGS)

# Proof generator
$(BIN)/proof_generator: examples/proof_generator.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

# Learned codec tests
$(BIN)/test_learned: $(SRC)/tests/test_learned_codec.c $(SRC)/train/wubu_learned_codec.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_positional_encode.c $(SRC)/math/wubu_utils.c $(SRC)/nn/wubu_nn.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/train/wubu_learned_codec.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_positional_encode.c $(SRC)/math/wubu_utils.c $(SRC)/nn/wubu_nn.c -o $@ $(LDFLAGS)

# Learned codec tests
$(BIN)/test_learned: $(SRC)/tests/test_learned_codec.c $(SRC)/train/wubu_learned_codec.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_positional_encode.c $(SRC)/math/wubu_utils.c $(SRC)/nn/wubu_nn.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/train/wubu_learned_codec.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_quaternion_ops.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_positional_encode.c $(SRC)/math/wubu_utils.c $(SRC)/nn/wubu_nn.c -o $@ $(LDFLAGS)
# GAAD encoder tests
$(BIN)/test_gaad: $(SRC)/tests/test_gaad_encoder.c $(SRC)/model/wubu_gaad_encoder.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/model/wubu_gaad_encoder.c -o $@ $(LDFLAGS)

# VHF Engine tests (faithful slerm of vhf_audio.py)
$(BIN)/test_vhf_engine: $(SRC)/tests/test_vhf_engine.c $(SRC)/model/wubu_vhf_engine.c $(SRC)/train/wubu_q_controller.c $(SRC)/train/wubu_loss.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_utils.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(SRC)/model/wubu_vhf_engine.c $(SRC)/train/wubu_q_controller.c $(SRC)/train/wubu_loss.c $(SRC)/math/wubu_color.c $(SRC)/math/wubu_utils.c -o $@ $(LDFLAGS)

# Analytical validation contract (anti-fart-sniffing guard):
# pins each hyperbolic/Mobius kernel to its CLOSED-FORM formula
# (same formula proven in lean/WubuProofs/*.lean).
$(BIN)/test_hyperbolic_analytics: $(SRC)/tests/test_hyperbolic_analytics.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_parallel_transport.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_parallel_transport.c $< -o $@ $(LDFLAGS)

test: $(BIN)/test_vhf_engine $(BIN)/test_gaad $(BIN)/wubu_tests $(BIN)/jax_test $(BIN)/nn_test $(BIN)/test_hyperbolic $(BIN)/test_quaternion $(BIN)/test_so3 $(BIN)/test_rep $(BIN)/test_riemannian_sgd $(BIN)/test_parallel_transport $(BIN)/test_hyperbolic_analytics $(BIN)/test_tangent_flow $(BIN)/test_flow_matching $(BIN)/test_latent_codec $(BIN)/test_nest_gpt $(BIN)/test_quat_ops $(BIN)/test_canvas_res $(BIN)/test_nested_enc $(BIN)/test_learned
	@echo "=== VHF Engine Tests ===" && $(BIN)/test_vhf_engine
	@echo "=== WuBuMath Tests ===" && $(BIN)/wubu_tests
	@echo "=== Slermed JAX Tests ===" && $(BIN)/jax_test
	@echo "=== NN Layer Tests ===" && $(BIN)/nn_test
	@echo "=== Hyperbolic Geometry Tests ===" && $(BIN)/test_hyperbolic
	@echo "=== Quaternion Tests ===" && $(BIN)/test_quaternion
	@echo "=== SO(3) exp/log/geodesic (libirrep port) ===" && $(BIN)/test_so3
	@echo "=== Rep-theory Wigner3j/CG (libirrep port) ===" && $(BIN)/test_rep
	@echo "=== Riemannian SGD Tests ===" && $(BIN)/test_riemannian_sgd
	@echo "=== Parallel Transport Tests ===" && $(BIN)/test_parallel_transport
	@echo "=== Tangent Flow Tests ===" && $(BIN)/test_tangent_flow
	@echo "=== Flow Matching Tests ===" && $(BIN)/test_flow_matching
	@echo "=== Canvas Resolution Tests ===" && $(BIN)/test_canvas_res
	@echo "=== Latent Codec Tests ===" && $(BIN)/test_latent_codec
	@echo "=== WuBuNestGPT Tests ===" && $(BIN)/test_nest_gpt
	@echo "=== Quat Ops Tests ===" && $(BIN)/test_quat_ops
	@echo "=== Canvas Resolution Tests ===" && $(BIN)/test_canvas_res
	@echo "=== Nested Encoder Tests ===" && $(BIN)/test_nested_enc
	@echo "=== GAAD Encoder Tests ===" && $(BIN)/test_gaad
	@echo "=== Learned Codec Tests ===" && $(BIN)/test_learned
