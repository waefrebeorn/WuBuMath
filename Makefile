# WuBuMath Makefile
# Pure C11 mathematical & media encoding library

CC = gcc
CFLAGS = -std=c11 -O3 -march=native -Iinclude -w
LDFLAGS = -lm -lpthread

SRC = src
INC = include
BEARINC = /home/wubu/BearRL/bear
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
       $(SRC)/math/wubu_manifold.c \
       $(SRC)/math/wubu_anyon.c \
       $(SRC)/math/wubu_poincare_geom.c \
       $(SRC)/math/wubu_manifold_ad.c \
       $(SRC)/math/wubu_lorentz.c \
       $(SRC)/math/wubu_lorentz_poincare.c \
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

# Manifold geodesic tests (qgt RK4 geodesic equation port, S^2 validation)
$(BIN)/test_manifold: $(SRC)/tests/test_wubu_manifold.c $(SRC)/math/wubu_manifold.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_manifold.c $< -o $@ $(LDFLAGS)

# Anyon SU(2)_k tests (moonlab port: fusion, R-matrix, quantum 6j)
$(BIN)/test_anyon: $(SRC)/tests/test_wubu_anyon.c $(SRC)/math/wubu_anyon.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_anyon.c $< -o $@ $(LDFLAGS)

# Poincare/sphere geometry cross-validation (eshkol manifold.esk release)
$(BIN)/test_pgeom: $(SRC)/tests/test_wubu_poincare_geom.c $(SRC)/math/wubu_poincare_geom.c $(SRC)/math/wubu_hyperbolic.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_poincare_geom.c $(SRC)/math/wubu_hyperbolic.c $< -o $@ $(LDFLAGS)

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
                            $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c \
                            $(SRC)/math/wubu_parallel_transport.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/train/wubu_flow_matching.c $(SRC)/math/wubu_hyperbolic.c \
	    $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_parallel_transport.c $< -o $@ $(LDFLAGS)

$(BIN)/test_avfid: $(SRC)/tests/test_wubu_av_fidelity.c $(SRC)/train/wubu_av_fidelity.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_av_fidelity.c $< -o $@ $(LDFLAGS)

$(BIN)/test_kodak: $(SRC)/tests/test_wubu_kodak.c $(SRC)/audio/wubu_kodak.c $(SRC)/audio/wubu_stft.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/audio/wubu_kodak.c $(SRC)/audio/wubu_stft.c $< -o $@ $(LDFLAGS)

$(BIN)/test_uv: $(SRC)/tests/test_uv_band.c $(SRC)/model/wubu_beam.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/model/wubu_beam.c $< -o $@ $(LDFLAGS)

$(BIN)/test_pframe_ir: $(SRC)/tests/test_pframe_ir.c $(SRC)/train/wubu_flow_matching.c $(SRC)/model/wubu_beam.c $(SRC)/train/wubu_text_encoder.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_parallel_transport.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_flow_matching.c $(SRC)/model/wubu_beam.c $(SRC)/train/wubu_text_encoder.c $(SRC)/math/wubu_hyperbolic.c $(SRC)/math/wubu_quaternion.c $(SRC)/math/wubu_parallel_transport.c $< -o $@ $(LDFLAGS)

$(BIN)/test_pairs: $(SRC)/tests/test_pair_retrieval.c $(SRC)/train/wubu_pairdata.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_pairdata.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hlin: $(SRC)/tests/test_wubu_hlinear.c $(SRC)/math/wubu_hlinear.c $(SRC)/math/wubu_lorentz.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hlinear.c $(SRC)/math/wubu_lorentz.c $< -o $@ $(LDFLAGS)

$(BIN)/test_pa: $(SRC)/tests/test_wubu_psychoacoustic.c $(SRC)/audio/wubu_psychoacoustic.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/audio/wubu_psychoacoustic.c $< -o $@ $(LDFLAGS)

$(BIN)/test_eg: $(SRC)/tests/test_wubu_expgolomb.c $(SRC)/audio/wubu_expgolomb.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/audio/wubu_expgolomb.c $< -o $@ $(LDFLAGS)

$(BIN)/test_delta: $(SRC)/tests/test_wubu_delta.c $(SRC)/audio/wubu_delta.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/audio/wubu_delta.c $< -o $@ $(LDFLAGS)

$(BIN)/test_aroll: $(SRC)/tests/test_wubu_attn_rollout.c $(SRC)/math/wubu_attn_rollout.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_attn_rollout.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hier: $(SRC)/tests/test_wubu_hier.c $(SRC)/math/wubu_hier.c $(SRC)/math/wubu_tree_embed.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hier.c $(SRC)/math/wubu_tree_embed.c $< -o $@ $(LDFLAGS)

$(BIN)/test_gpath: $(SRC)/tests/test_wubu_geodesic_path.c $(SRC)/math/wubu_geodesic_path.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_geodesic_path.c $< -o $@ $(LDFLAGS)

$(BIN)/test_pm: $(SRC)/tests/test_wubu_prodman.c $(SRC)/math/wubu_prodman.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_prodman.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hkm: $(SRC)/tests/test_wubu_hkmedoids.c $(SRC)/math/wubu_hkmedoids.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hkmedoids.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hvakl: $(SRC)/tests/test_wubu_hvae_kl.c $(SRC)/train/wubu_hvae_kl.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_hvae_kl.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hae: $(SRC)/tests/test_wubu_hae.c $(SRC)/train/wubu_hae.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_hae.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hdt: $(SRC)/tests/test_wubu_hdt.c $(SRC)/math/wubu_hdt.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hdt.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hr: $(SRC)/tests/test_wubu_hretrieval.c $(SRC)/math/wubu_hretrieval.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hretrieval.c $< -o $@ $(LDFLAGS)

$(BIN)/test_g2b: $(SRC)/tests/test_wubu_graph2ball.c $(SRC)/train/wubu_graph2ball.c $(SRC)/train/wubu_poincare_emb.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_graph2ball.c $(SRC)/train/wubu_poincare_emb.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hdpc: $(SRC)/tests/test_wubu_hdpc.c $(SRC)/math/wubu_hdpc.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hdpc.c $< -o $@ $(LDFLAGS)

$(BIN)/test_ec2: $(SRC)/tests/test_wubu_entail2.c $(SRC)/math/wubu_entail2.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_entail2.c $< -o $@ $(LDFLAGS)

$(BIN)/test_lclip: $(SRC)/tests/test_wubu_lorentz_clip.c $(SRC)/math/wubu_lorentz_clip.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_lorentz_clip.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hm: $(SRC)/tests/test_wubu_hmix.c $(SRC)/math/wubu_hmix.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hmix.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hlr: $(SRC)/tests/test_wubu_hlogreg.c $(SRC)/train/wubu_hlogreg.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_hlogreg.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hmds: $(SRC)/tests/test_wubu_hmds.c $(SRC)/math/wubu_hmds.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hmds.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hkrum: $(SRC)/tests/test_wubu_hkrum.c $(SRC)/math/wubu_hkrum.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hkrum.c $< -o $@ $(LDFLAGS)

$(BIN)/test_pemb: $(SRC)/tests/test_wubu_poincare_emb.c $(SRC)/train/wubu_poincare_emb.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_poincare_emb.c $< -o $@ $(LDFLAGS)

$(BIN)/test_kvc: $(SRC)/tests/test_wubu_kvcache.c $(SRC)/math/wubu_kvcache.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_kvcache.c $< -o $@ $(LDFLAGS)

$(BIN)/test_causal: $(SRC)/tests/test_wubu_causal.c $(SRC)/math/wubu_causal.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_causal.c $< -o $@ $(LDFLAGS)

$(BIN)/test_lpos: $(SRC)/tests/test_wubu_learned_pos.c $(SRC)/math/wubu_learned_pos.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_learned_pos.c $< -o $@ $(LDFLAGS)

$(BIN)/test_dg: $(SRC)/tests/test_wubu_dasgupta.c $(SRC)/math/wubu_dasgupta.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_dasgupta.c $< -o $@ $(LDFLAGS)

$(BIN)/test_knng: $(SRC)/tests/test_wubu_knngraph.c $(SRC)/math/wubu_knngraph.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_knngraph.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hgat: $(SRC)/tests/test_wubu_hgat.c $(SRC)/math/wubu_hgat.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hgat.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hstack: $(SRC)/tests/test_wubu_hstack.c $(SRC)/math/wubu_hstack.c $(SRC)/math/wubu_hgnn.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hstack.c $(SRC)/math/wubu_hgnn.c $< -o $@ $(LDFLAGS)

$(BIN)/test_htsne: $(SRC)/tests/test_wubu_htsne.c $(SRC)/math/wubu_htsne.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_htsne.c $< -o $@ $(LDFLAGS)

$(BIN)/test_htf: $(SRC)/tests/test_wubu_htransformer.c $(SRC)/model/wubu_htransformer.c $(SRC)/math/wubu_learned_pos.c $(SRC)/math/wubu_hblock.c $(SRC)/math/wubu_hnorm.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/model/wubu_htransformer.c $(SRC)/math/wubu_learned_pos.c $(SRC)/math/wubu_hblock.c $(SRC)/math/wubu_hnorm.c $< -o $@ $(LDFLAGS)

$(BIN)/test_radam: $(SRC)/tests/test_wubu_radam.c $(SRC)/math/wubu_radam.c $(SRC)/math/wubu_rsgd.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_radam.c $(SRC)/math/wubu_rsgd.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hmha: $(SRC)/tests/test_wubu_hmha.c $(SRC)/math/wubu_hmha.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hmha.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hblock: $(SRC)/tests/test_wubu_hblock.c $(SRC)/math/wubu_hblock.c $(SRC)/math/wubu_hnorm.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hblock.c $(SRC)/math/wubu_hnorm.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hnorm: $(SRC)/tests/test_wubu_hnorm.c $(SRC)/math/wubu_hnorm.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hnorm.c $< -o $@ $(LDFLAGS)

$(BIN)/test_pflow: $(SRC)/tests/test_wubu_pflow.c $(SRC)/train/wubu_pflow.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_pflow.c $< -o $@ $(LDFLAGS)

$(BIN)/test_ec2: $(SRC)/tests/test_wubu_entail2.c $(SRC)/math/wubu_entail2.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_entail2.c $< -o $@ $(LDFLAGS)

$(BIN)/test_lclip: $(SRC)/tests/test_wubu_lorentz_clip.c $(SRC)/math/wubu_lorentz_clip.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_lorentz_clip.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hm: $(SRC)/tests/test_wubu_hiermerge.c $(SRC)/math/wubu_hiermerge.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hiermerge.c $< -o $@ $(LDFLAGS)

$(BIN)/test_rd: $(SRC)/tests/test_wubu_rdomode.c $(SRC)/train/wubu_rdomode.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_rdomode.c $< -o $@ $(LDFLAGS)

$(BIN)/test_slerp: $(SRC)/tests/test_wubu_slerp_path.c $(SRC)/math/wubu_slerp_path.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_slerp_path.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hconv: $(SRC)/tests/test_wubu_hconv.c $(SRC)/math/wubu_hconv.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hconv.c $< -o $@ $(LDFLAGS)

$(BIN)/test_ep: $(SRC)/tests/test_wubu_euclidparam.c $(SRC)/math/wubu_euclidparam.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_euclidparam.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hgnn: $(SRC)/tests/test_wubu_hgnn.c $(SRC)/math/wubu_hgnn.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hgnn.c $< -o $@ $(LDFLAGS)

$(BIN)/test_tree: $(SRC)/tests/test_wubu_tree_embed.c $(SRC)/math/wubu_tree_embed.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_tree_embed.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hvae: $(SRC)/tests/test_wubu_hvae.c $(SRC)/math/wubu_hvae.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hvae.c $< -o $@ $(LDFLAGS)

$(BIN)/test_rsgd: $(SRC)/tests/test_wubu_rsgd.c $(SRC)/math/wubu_rsgd.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_rsgd.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hknn: $(SRC)/tests/test_wubu_hknn.c $(SRC)/math/wubu_hknn.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hknn.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hope: $(SRC)/tests/test_wubu_hope.c $(SRC)/math/wubu_hope.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hope.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hact: $(SRC)/tests/test_wubu_hactivation.c $(SRC)/math/wubu_hlinear.c $(SRC)/math/wubu_lorentz.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hlinear.c $(SRC)/math/wubu_lorentz.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hkmeans: $(SRC)/tests/test_wubu_hkmeans.c $(SRC)/math/wubu_hkmeans.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hkmeans.c $< -o $@ $(LDFLAGS)

$(BIN)/test_ldirect: $(SRC)/tests/test_wubu_ldirect.c $(SRC)/math/wubu_ldirect.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_ldirect.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hmlr: $(SRC)/tests/test_wubu_hmlr.c $(SRC)/math/wubu_hmlr.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hmlr.c $< -o $@ $(LDFLAGS)

$(BIN)/test_hattn: $(SRC)/tests/test_wubu_hattention.c $(SRC)/math/wubu_hattention.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_hattention.c $< -o $@ $(LDFLAGS)

$(BIN)/test_lorclip: $(SRC)/tests/test_lorentz_clip.c $(SRC)/math/wubu_lorentz.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_lorentz.c $< -o $@ $(LDFLAGS)

$(BIN)/test_arena: $(SRC)/tests/test_arena_alignment.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(BEARINC) $< -o $@ $(LDFLAGS)

$(BIN)/test_bands: $(SRC)/tests/test_wubu_bands.c $(SRC)/audio/wubu_bands.c $(SRC)/audio/wubu_stft.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/audio/wubu_bands.c $(SRC)/audio/wubu_stft.c $< -o $@ $(LDFLAGS)

$(BIN)/test_lorflow: $(SRC)/tests/test_lorentz_flow.c $(SRC)/math/wubu_lorentz.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_lorentz.c $< -o $@ $(LDFLAGS)

$(BIN)/test_quat_prop: $(SRC)/tests/test_quat_properties.c $(SRC)/math/wubu_quaternion_ops.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/math/wubu_quaternion_ops.c $< -o $@ $(LDFLAGS)

$(BIN)/test_text_encoder: $(SRC)/tests/test_wubu_text_encoder.c $(SRC)/train/wubu_text_encoder.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/train/wubu_text_encoder.c $< -o $@ $(LDFLAGS)

$(BIN)/test_decode_quality: $(SRC)/tests/test_decode_at_quality.c $(SRC)/model/wubu_beam.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/model/wubu_beam.c $< -o $@ $(LDFLAGS)

$(BIN)/test_beam_8k: $(SRC)/tests/test_beam_8k16k.c $(SRC)/model/wubu_beam.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/model/wubu_beam.c $< -o $@ $(LDFLAGS)

# Beam canvas + STFT tests (GAP-A001/A005/B003/E001)
$(BIN)/test_beam: $(SRC)/tests/test_wubu_beam.c $(SRC)/model/wubu_beam.c $(SRC)/train/wubu_text_encoder.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/model/wubu_beam.c $(SRC)/train/wubu_text_encoder.c $< -o $@ $(LDFLAGS)

$(BIN)/test_stft: $(SRC)/tests/test_wubu_stft.c $(SRC)/audio/wubu_stft.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $(SRC)/audio/wubu_stft.c $< -o $@ $(LDFLAGS)

# Manifold CLIP tests (GAP-D002/D003)
$(BIN)/test_manifold_clip: $(SRC)/tests/test_wubu_manifold_clip.c $(SRC)/train/wubu_manifold_clip.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -I$(INC) $< -o $@ $(LDFLAGS)

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

# Manifold automatic differentiation (B4): Riemannian gradient vs finite diff.
$(BIN)/test_manifold_ad: $(SRC)/tests/test_wubu_manifold_ad.c $(SRC)/math/wubu_manifold_ad.c $(SRC)/math/wubu_poincare_geom.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_manifold_ad.c $(SRC)/math/wubu_poincare_geom.c $< -o $@ $(LDFLAGS)

# Lorentz (hyperboloid) model + nested-hyperboloid projection (B-nested):
# grounded in Fan et al. CVPR 2022 Eq. 9/10/13.
$(BIN)/test_lorentz: $(SRC)/tests/test_wubu_lorentz.c $(SRC)/math/wubu_lorentz.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_lorentz.c $< -o $@ $(LDFLAGS)

# Lorentz<->Poincare conversion bridge (cross-checks wubu_poincare_geom).
$(BIN)/test_lorentz_poincare: $(SRC)/tests/test_wubu_lorentz_poincare.c $(SRC)/math/wubu_lorentz.c $(SRC)/math/wubu_lorentz_poincare.c $(SRC)/math/wubu_poincare_geom.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(SRC)/math/wubu_lorentz.c $(SRC)/math/wubu_lorentz_poincare.c $(SRC)/math/wubu_poincare_geom.c $< -o $@ $(LDFLAGS)

test: $(BIN)/test_vhf_engine $(BIN)/test_gaad $(BIN)/wubu_tests $(BIN)/jax_test $(BIN)/nn_test $(BIN)/test_hyperbolic $(BIN)/test_quaternion $(BIN)/test_so3 $(BIN)/test_rep $(BIN)/test_manifold $(BIN)/test_anyon $(BIN)/test_pgeom $(BIN)/test_riemannian_sgd $(BIN)/test_parallel_transport $(BIN)/test_hyperbolic_analytics $(BIN)/test_manifold_ad $(BIN)/test_lorentz $(BIN)/test_lorentz_poincare $(BIN)/test_tangent_flow $(BIN)/test_flow_matching $(BIN)/test_manifold_clip $(BIN)/test_beam $(BIN)/test_beam_8k $(BIN)/test_decode_quality $(BIN)/test_text_encoder $(BIN)/test_bands $(BIN)/test_lorflow $(BIN)/test_quat_prop $(BIN)/test_arena $(BIN)/test_lorclip $(BIN)/test_hattn $(BIN)/test_hmlr $(BIN)/test_ldirect $(BIN)/test_hkmeans $(BIN)/test_hlin $(BIN)/test_hact $(BIN)/test_uv $(BIN)/test_kodak $(BIN)/test_pframe_ir $(BIN)/test_uv $(BIN)/test_pairs $(BIN)/test_avfid $(BIN)/test_stft $(BIN)/test_latent_codec $(BIN)/test_nest_gpt $(BIN)/test_quat_ops $(BIN)/test_canvas_res $(BIN)/test_nested_enc $(BIN)/test_learned
	@echo "=== VHF Engine Tests ===" && $(BIN)/test_vhf_engine
	@echo "=== WuBuMath Tests ===" && $(BIN)/wubu_tests
	@echo "=== Slermed JAX Tests ===" && $(BIN)/jax_test
	@echo "=== NN Layer Tests ===" && $(BIN)/nn_test
	@echo "=== Hyperbolic Geometry Tests ===" && $(BIN)/test_hyperbolic
	@echo "=== Quaternion Tests ===" && $(BIN)/test_quaternion
	@echo "=== SO(3) exp/log/geodesic (libirrep port) ===" && $(BIN)/test_so3
	@echo "=== Rep-theory Wigner3j/CG (libirrep port) ===" && $(BIN)/test_rep
	@echo "=== Manifold geodesic (qgt RK4 port, S^2) ===" && $(BIN)/test_manifold
	@echo "=== Anyon SU(2)_k (moonlab port) ===" && $(BIN)/test_anyon
	@echo "=== Poincare/sphere geometry cross-val (eshkol) ===" && $(BIN)/test_pgeom
	@echo "=== Riemannian SGD Tests ===" && $(BIN)/test_riemannian_sgd
	@echo "=== Parallel Transport Tests ===" && $(BIN)/test_parallel_transport
	@echo "=== Tangent Flow Tests ===" && $(BIN)/test_tangent_flow
	@echo "=== Flow Matching Tests ===" && $(BIN)/test_flow_matching
	@echo "=== Manifold CLIP Tests ===" && $(BIN)/test_manifold_clip
	@echo "=== Beam Canvas Tests ===" && $(BIN)/test_beam
	@echo "=== Beam 8K/16K Integration ===" && $(BIN)/test_beam_8k
	@echo "=== Decode-at-N Quality ===" && $(BIN)/test_decode_quality
	@echo "=== Text Encoder Tests ===" && $(BIN)/test_text_encoder
	@echo "=== Perceptual Bands Tests ===" && $(BIN)/test_bands
	@echo "=== Lorentz Flow Tests ===" && $(BIN)/test_lorflow
	@echo "=== Quat Property Invariants ===" && $(BIN)/test_quat_prop
	@echo "=== Arena Property Tests ===" && $(BIN)/test_arena
	@echo "=== Lorentz CLIP Tests ===" && $(BIN)/test_lorclip
	@echo "=== Hyperbolic Attention Tests ===" && $(BIN)/test_hattn
	@echo "=== Hyperbolic MLR ===" && $(BIN)/test_hmlr
	@echo "=== Tangent-Free Lorentz Linear ===" && $(BIN)/test_ldirect
	@echo "=== Hyperbolic K-Means ===" && $(BIN)/test_hkmeans
	@echo "=== Hyperboloid Linear Layer ===" && $(BIN)/test_hlin
	@echo "=== Hyperbolic Activation ===" && $(BIN)/test_hact
	@echo "=== HoPE Positional Encoding ===" && $(BIN)/test_hope
	@echo "=== Hyperbolic k-NN ===" && $(BIN)/test_hknn
	@echo "=== Riemannian SGD ===" && $(BIN)/test_rsgd
	@echo "=== Hyperbolic VAE Sampling ===" && $(BIN)/test_hvae
	@echo "=== Sarkar Tree Embedding ===" && $(BIN)/test_tree
	@echo "=== Hyperbolic GCN Layer ===" && $(BIN)/test_hgnn
	@echo "=== Euclidean Parametrization ===" && $(BIN)/test_ep
	@echo "=== Hyperbolic Conv (Poincaré MLP) ===" && $(BIN)/test_hconv
	@echo "=== Quaternion SLERP Path ===" && $(BIN)/test_slerp
	@echo "=== RD Mode Decision ===" && $(BIN)/test_rd
	@echo "=== Hierarchical Merge (tree recovery) ===" && $(BIN)/test_hm
	@echo "=== Poincaré Flow Matching ===" && $(BIN)/test_pflow
	@echo "=== Poincaré LayerNorm ===" && $(BIN)/test_hnorm
	@echo "=== Hyperbolic Transformer Block ===" && $(BIN)/test_hblock
	@echo "=== Hyperbolic Multi-Head Attention ===" && $(BIN)/test_hmha
	@echo "=== Riemannian Adam ===" && $(BIN)/test_radam
	@echo "=== Full Hyperbolic Transformer ===" && $(BIN)/test_htf
	@echo "=== Hyperbolic t-SNE ===" && $(BIN)/test_htsne
	@echo "=== Hyperbolic GCN Stack ===" && $(BIN)/test_hstack
	@echo "=== Hyperbolic GAT ===" && $(BIN)/test_hgat
	@echo "=== k-NN Graph ===" && $(BIN)/test_knng
	@echo "=== Dasgupta Cost ===" && $(BIN)/test_dg
	@echo "=== Learned Positional Embedding ===" && $(BIN)/test_lpos
	@echo "=== Causal Mask ===" && $(BIN)/test_causal
	@echo "=== KV Cache ===" && $(BIN)/test_kvc
	@echo "=== Poincaré Embeddings ===" && $(BIN)/test_pemb
	@echo "=== Hyperbolic Krum Outliers ===" && $(BIN)/test_hkrum
	@echo "=== Hyperbolic MDS ===" && $(BIN)/test_hmds
	@echo "=== Hyperbolic Logistic Regression ===" && $(BIN)/test_hlr
	@echo "=== H×S Product Manifold ===" && $(BIN)/test_hm
	@echo "=== Lorentz CLIP Similarity ===" && $(BIN)/test_lclip
	@echo "=== Entailment Cone ===" && $(BIN)/test_ec2
	@echo "=== Density-Peak Clustering ===" && $(BIN)/test_hdpc
	@echo "=== Graph-to-Ball Embedding ===" && $(BIN)/test_g2b
	@echo "=== Hierarchical Retrieval ===" && $(BIN)/test_hr
	@echo "=== Hyperbolic Decision Tree ===" && $(BIN)/test_hdt
	@echo "=== Hyperbolic Autoencoder ===" && $(BIN)/test_hae
	@echo "=== Hyperbolic VAE KL ===" && $(BIN)/test_hvakl
	@echo "=== Hyperbolic K-Medoids ===" && $(BIN)/test_hkm
	@echo "=== Product Manifold ===" && $(BIN)/test_pm
	@echo "=== Geodesic Path ===" && $(BIN)/test_gpath
	@echo "=== Hierarchical Classification ===" && $(BIN)/test_hier
	@echo "=== Attention Rollout ===" && $(BIN)/test_aroll
	@echo "=== Delta/Zigzag/Varint ===" && $(BIN)/test_delta
	@echo "=== Exponential-Golomb ===" && $(BIN)/test_eg
	@echo "=== Psychoacoustic Masking ===" && $(BIN)/test_pa
	@echo "=== Pair Retrieval (D006/D010) ===" && $(BIN)/test_pairs
	@echo "=== Kodak Audio-Image Tests ===" && $(BIN)/test_kodak
	@echo "=== P-Frame Infrared Pipeline ===" && $(BIN)/test_pframe_ir
	@echo "=== UV Band Tests ===" && $(BIN)/test_uv
	@echo "=== AV Fidelity Tests ===" && $(BIN)/test_avfid
	@echo "=== STFT Tests ===" && $(BIN)/test_stft
	@echo "=== Canvas Resolution Tests ===" && $(BIN)/test_canvas_res
	@echo "=== Latent Codec Tests ===" && $(BIN)/test_latent_codec
	@echo "=== WuBuNestGPT Tests ===" && $(BIN)/test_nest_gpt
	@echo "=== Quat Ops Tests ===" && $(BIN)/test_quat_ops
	@echo "=== Canvas Resolution Tests ===" && $(BIN)/test_canvas_res
	@echo "=== Nested Encoder Tests ===" && $(BIN)/test_nested_enc
	@echo "=== GAAD Encoder Tests ===" && $(BIN)/test_gaad
	@echo "=== Learned Codec Tests ===" && $(BIN)/test_learned


# GAP-I003: install + pkg-config for downstream linking (BearRL etc.)
libwubumath.a:
	$(AR) rcs $@ $(SRCS:.c=.o) 2>/dev/null || true

wubumath.pc: wubumath.pc.in
	sed "s|@PREFIX@|/usr/local|" $< > $@

install: libwubumath.a
	mkdir -p /usr/local/lib /usr/local/include/wubumath /usr/local/lib/pkgconfig
	cp include/*.h /usr/local/include/wubumath/
	cp libwubumath.a /usr/local/lib/ 2>/dev/null || true
	cp wubumath.pc /usr/local/lib/pkgconfig/
