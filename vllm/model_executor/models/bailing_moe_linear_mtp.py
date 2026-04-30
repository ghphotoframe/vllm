# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bailing MoE v2.5 Multi-Token Prediction (MTP) implementation.

This module implements MTP support for Bailing models with hybrid attention
(Linear Attention + MLA). The MTP layers use full MLA attention to avoid
CudaGraph limitations of Linear Attention.
"""
import copy
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .bailing_moe_linear import BailingMoeV25DecoderLayer
from .utils import PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


class SharedHead(nn.Module):
    """Shared head for computing logits in MTP layers.
    
    This module combines RMSNorm and LM head projection. The actual lm_head
    weight is shared with the main model and set externally.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # LM head will be set externally to share with main model
        self.head = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply normalization to hidden states."""
        return self.norm(hidden_states)


class BailingMultiTokenPredictorLayer(nn.Module):
    """Single MTP prediction layer for Bailing model.
    
    Each MTP layer consists of:
    1. enorm: RMSNorm for input embeddings
    2. hnorm: RMSNorm for previous hidden states
    3. eh_proj: Fusion projection (2*hidden_size -> hidden_size)
    4. mtp_block: A full BailingMoeV25DecoderLayer with MLA attention
    5. shared_head: Norm + LM head for logits computation
    
    Args:
        vllm_config: vLLM configuration
        layer_idx: Layer index in the MTP stack
        prefix: Parameter name prefix
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        self.config = config
        self.layer_idx = layer_idx

        # Normalization layers for embedding and hidden state fusion
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Fusion projection: concat([e_norm, h_norm]) -> hidden_size
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "eh_proj"),
        )

        # MTP block: Force MLA attention (attention_type=1)
        # This avoids Linear Attention's CudaGraph limitation
        layer_config = copy.deepcopy(config)
        layer_config.attention_type = 1  # Force MLA

        self.mtp_block = BailingMoeV25DecoderLayer(
            config=layer_config,
            quant_config=quant_config,
            layer_id=layer_idx,
            prefix=maybe_prefix(prefix, "mtp_block"),
            model_config=model_config,
            cache_config=cache_config,
        )

        # Shared head for logits computation
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=quant_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Forward pass for MTP layer.
        
        Args:
            input_ids: Input token IDs [batch_size]
            positions: Token positions [batch_size]
            previous_hidden_states: Hidden states from main model [batch_size, hidden_size]
            inputs_embeds: Input embeddings [batch_size, hidden_size]
            attn_metadata: Attention metadata
            
        Returns:
            Hidden states after MTP block [batch_size, hidden_size]
        """
        # Step 1: Mask position 0 (BOS token not needed by MTP)
        mask = (positions == 0).unsqueeze(-1).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds * (1.0 - mask)

        # Step 2: Normalize embeddings and hidden states
        e_norm = self.enorm(inputs_embeds)
        h_norm = self.hnorm(previous_hidden_states)

        # Step 3: Fuse and project
        eh_cat = torch.cat([e_norm, h_norm], dim=-1)
        eh_input = self.eh_proj(eh_cat)

        # Step 4: Pass through MTP block
        hidden_states, residual = self.mtp_block(
            hidden_states=eh_input,
            positions=positions,
            attn_metadata=attn_metadata,
            residual=None,
        )

        # Step 5: Add residual
        if residual is not None:
            hidden_states = hidden_states + residual

        return hidden_states


class BailingMultiTokenPredictor(nn.Module):
    """Multi-Token Predictor for Bailing model.
    
    This module contains multiple MTP layers that are used cyclically
    for speculative token prediction.
    
    Args:
        vllm_config: vLLM configuration
        prefix: Parameter name prefix
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        # MTP layer configuration
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        logger.info(
            f"Initializing Bailing MTP with {self.num_mtp_layers} layers "
            f"(indices {self.mtp_start_layer_idx} to "
            f"{self.mtp_start_layer_idx + self.num_mtp_layers - 1})"
        )

        # Create MTP layers
        self.layers = nn.ModuleDict(
            {
                str(idx): BailingMultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    layer_idx=idx,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        # Embedding layer (will be loaded from checkpoint)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Logits processor
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass for MTP predictor.
        
        Args:
            input_ids: Input token IDs
            positions: Token positions
            previous_hidden_states: Hidden states from main model
            inputs_embeds: Optional pre-computed embeddings
            spec_step_idx: Speculative step index (for layer cycling)
            
        Returns:
            Hidden states after MTP layer
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cycle through MTP layers based on spec_step_idx
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        layer = self.layers[str(layer_idx)]

        # Get attention metadata from forward context
        from vllm.forward_context import get_forward_context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        hidden_states = layer(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=previous_hidden_states,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden states from MTP layer
            spec_step_idx: Speculative step index
            
        Returns:
            Logits [batch_size, vocab_size]
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        mtp_layer = self.layers[str(layer_idx)]

        # Apply shared head (norm + lm_head)
        normalized = mtp_layer.shared_head(hidden_states)
        logits = self.logits_processor(
            mtp_layer.shared_head.head,
            normalized,
        )

        return logits


@support_torch_compile
class BailingMoeV25MTP(nn.Module):
    """Bailing MoE v2.5 Multi-Token Prediction model.
    
    This is the top-level MTP model that wraps BailingMultiTokenPredictor
    and handles weight loading.
    
    Args:
        vllm_config: vLLM configuration
        prefix: Parameter name prefix
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config

        # Create MTP predictor
        self.model = BailingMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Create LM head (shared with main model)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

            # Set shared head for all MTP layers
            for layer in self.model.layers.values():
                layer.shared_head.head = self.lm_head

            self.logits_processor = LogitsProcessor(self.config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        logger.info("BailingMoeV25MTP initialized successfully")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass for MTP model.
        
        Args:
            input_ids: Input token IDs
            positions: Token positions
            hidden_states: Hidden states from main model (previous_hidden_states)
            intermediate_tensors: Unused (for interface compatibility)
            inputs_embeds: Optional pre-computed embeddings
            spec_step_idx: Speculative step index
            
        Returns:
            Hidden states after MTP layer
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden states from MTP layer
            spec_step_idx: Speculative step index
            
        Returns:
            Logits or None if not last rank
        """
        if not get_pp_group().is_last_rank:
            return None

        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.
        
        This method handles weight name mapping from checkpoint format to
        model parameter format. Key mappings:
        - model.mtp.embed_tokens.* -> model.embed_tokens.*
        - model.mtp.layers.{i}.enorm.* -> model.layers.{i}.enorm.*
        - model.mtp.layers.{i}.hnorm.* -> model.layers.{i}.hnorm.*
        - model.mtp.layers.{i}.eh_proj.* -> model.layers.{i}.eh_proj.*
        - model.mtp.layers.{i}.{rest} -> model.layers.{i}.mtp_block.{rest}
        
        Args:
            weights: Iterable of (name, tensor) pairs from checkpoint
            
        Returns:
            Set of loaded parameter names
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        # Stacked parameter mappings
        stacked_mappings = [
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Expert parameter mappings
        expert_mappings = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=0,
        )

        def load_param(name: str, tensor: torch.Tensor, shard_id=None) -> bool:
            """Load a single parameter with optional sharding."""
            if name not in params_dict:
                return False

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            if shard_id is None:
                weight_loader(param, tensor)
            elif isinstance(shard_id, int):
                weight_loader(param, tensor, shard_id)
            else:
                # Expert param: (expert_id, shard_id)
                weight_loader(
                    param,
                    tensor,
                    name,
                    expert_id=shard_id[0],
                    shard_id=shard_id[1],
                )

            loaded_params.add(name)
            return True

        def get_spec_layer_idx(name: str) -> int | None:
            """Extract MTP layer index from weight name."""
            if "model.mtp.layers." not in name and "model.layers." not in name:
                return None

            try:
                if "model.mtp.layers." in name:
                    parts = name.split("model.mtp.layers.")[1].split(".")
                elif "model.layers." in name:
                    parts = name.split("model.layers.")[1].split(".")
                else:
                    return None

                layer_idx = int(parts[0])

                # Check if this is an MTP layer
                if layer_idx >= self.model.mtp_start_layer_idx:
                    return layer_idx
            except (IndexError, ValueError):
                pass

            return None

        def normalize_name(name: str) -> str | None:
            """Normalize checkpoint name to model parameter name."""
            # Remove model. prefix
            name = name.removeprefix("model.")

            # Handle MTP weights
            if name.startswith("mtp."):
                # model.mtp.layers.{i}.* -> model.layers.{i}.*
                name = name.replace("mtp.", "")

                # Determine if this weight belongs to MTP-specific components
                # or the transformer block
                spec_layer_idx = get_spec_layer_idx("model." + name)
                if spec_layer_idx is not None:
                    # Check if it's a MTP-specific component
                    mtp_specific = ["enorm", "hnorm", "eh_proj", "shared_head"]
                    is_mtp_specific = any(comp in name for comp in mtp_specific)

                    if not is_mtp_specific:
                        # Transformer block weights: add mtp_block prefix
                        # model.layers.{i}.self_attn.* -> model.layers.{i}.mtp_block.self_attn.*
                        layer_prefix = f"layers.{spec_layer_idx}."
                        if name.startswith(layer_prefix):
                            rest = name[len(layer_prefix):]
                            name = f"{layer_prefix}mtp_block.{rest}"

            # Handle attention.dense mapping based on layer type
            # (MTP layers always use MLA, so always map to o_proj)
            if "attention.dense" in name:
                name = name.replace("attention.dense", "self_attn.o_proj")

            # Standard mappings
            name = name.replace("attention.", "self_attn.")
            name = name.replace(
                "mlp.gate.e_score_correction_bias",
                "mlp.gate.expert_bias",
            )

            return maybe_remap_kv_scale_name(name, params_dict)

        # Load weights
        loaded_layers: set[int] = set()

        for orig_name, weight in weights:
            norm_name = normalize_name(orig_name)
            if norm_name is None:
                continue

            # Track which MTP layers have weights loaded
            spec_layer_idx = get_spec_layer_idx(orig_name)
            if spec_layer_idx is not None:
                loaded_layers.add(spec_layer_idx)

            # Try stacked mappings
            loaded = False
            for param_suf, weight_suf, shard_id in stacked_mappings:
                if weight_suf not in norm_name:
                    continue
                mapped = norm_name.replace(weight_suf, param_suf)
                if load_param(mapped, weight, shard_id):
                    loaded = True
                    break

            if loaded:
                continue

            # Handle expert weights
            if "mlp.experts" in norm_name:
                # Expert bias
                if (
                    "mlp.experts.e_score_correction_bias" in norm_name
                    or "mlp.experts.expert_bias" in norm_name
                ):
                    alt = norm_name.replace(
                        "mlp.experts.e_score_correction_bias",
                        "mlp.gate.expert_bias",
                    ).replace("mlp.experts.expert_bias", "mlp.gate.expert_bias")
                    if load_param(alt, weight) or load_param(norm_name, weight):
                        continue

                # Routed experts
                for param_name, weight_name, expert_id, shard_id in expert_mappings:
                    if weight_name not in norm_name:
                        continue
                    mapped = norm_name.replace(weight_name, param_name)
                    if load_param(mapped, weight, (expert_id, shard_id)):
                        break
                continue

            # General parameters
            # Share embedding weights: only load once
            if "embed_tokens" in norm_name and spec_layer_idx is not None:
                if spec_layer_idx != self.model.mtp_start_layer_idx:
                    continue

            load_param(norm_name, weight)

        # Validate that all MTP layers have weights loaded
        expected_layers = set(
            range(
                self.model.mtp_start_layer_idx,
                self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
            )
        )

        missing_layers = expected_layers - loaded_layers
        if missing_layers:
            raise ValueError(
                f"MTP layers {sorted(missing_layers)} are missing weights from "
                f"checkpoint. Expected layers {sorted(expected_layers)}, "
                f"but only found {sorted(loaded_layers)}. "
                f"The checkpoint may not include MTP weights, or "
                f"num_nextn_predict_layers may be misconfigured."
            )

        logger.info(
            f"Successfully loaded weights for {len(loaded_layers)} MTP layers: "
            f"{sorted(loaded_layers)}"
        )

        return loaded_params
