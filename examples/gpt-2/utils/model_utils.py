"""
Model utility functions
"""

# pylint: disable=protected-access

import json
import sys

import torch
from texar import HParams


def transform_gpt2_to_texar_config(input_json_path):
    """
    Remap the config file
    """
    config_gpt = json.loads(open(input_json_path).read())
    configs = dict()
    configs["vocab_size"] = config_gpt["n_vocab"]
    configs["context_size"] = config_gpt["n_ctx"]
    configs["embedding_size"] = config_gpt["n_embd"]
    hidden_dim = config_gpt["n_embd"]
    configs["embed"] = {
        "dim": hidden_dim,
    }
    configs["position_size"] = config_gpt["n_ctx"]
    configs["pos_embed"] = {
        "dim": hidden_dim
    }
    configs["decoder"] = {
        "dim": hidden_dim,
        "num_blocks": config_gpt["n_layer"],
        "use_gpt_config": True,
        "embedding_dropout": 0,
        "residual_dropout": 0,
        "multihead_attention": {
            "use_bias": True,
            "num_units": hidden_dim,
            "num_heads": config_gpt["n_head"],
            "output_dim": hidden_dim,
        },
        "initializer": {
            "type": "variance_scaling_initializer",
            "kwargs": {
                "scale": 1.0,
                "mode": "FAN_AVG",
                "uniform": True,
            },
        },
        "poswise_feedforward": {
            "layers": [
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": hidden_dim,
                        "out_features": hidden_dim * 4,
                        "bias": True,
                    }
                },
                {
                    "type": "GPTGELU",
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "kwargs": {
                        "in_features": hidden_dim * 4,
                        "out_features": hidden_dim,
                        "bias": True,
                    }
                }
            ],
            "name": "ffn",
        },
    }
    return HParams(configs, default_hparams=None)


def init_gpt2_checkpoint(model, init_checkpoint):
    """
    Initializes GPT-2 model parameters from a checkpoint
    Args:
        init_checkpoint (str): Path to the checkpoint.
    """

    try:
        import os
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to "
              "be installed. Please see https://www.tensorflow.org/install/ "
              "for installation instructions.")
        raise

    global_tensor_map = {
        "model/wte": "_embedding",
        "model/wpe": "_embedding",
        "model/ln_f/b": "final_layer_norm.bias",
        "model/ln_f/g": "final_layer_norm.weight",
    }
    layer_tensor_map = {
        "ln_1/b": "self_attn_layer_norm.{}.bias",
        "ln_1/g": "self_attn_layer_norm.{}.weight",
        "ln_2/b": "poswise_layer_norm.{}.bias",
        "ln_2/g": "poswise_layer_norm.{}.weight",
        "mlp/c_fc/b": "poswise_networks.{}._layers.0.bias",
        "mlp/c_proj/b": "poswise_networks.{}._layers.2.bias",
        "attn/c_proj/b": "self_attns.{}.O_dense.bias",
    }
    layer_transpose_map = {
        "mlp/c_fc/w": "poswise_networks.{}._layers.0.weight",
        "mlp/c_proj/w": "poswise_networks.{}._layers.2.weight",
        "attn/c_proj/w": "self_attns.{}.O_dense.weight",
    }

    tf_path = os.path.abspath(init_checkpoint)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, _ in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    tensor_names = []
    for name, _ in model.word_embedder.named_parameters():
        tensor_names.append(name)
    for name, _ in model.pos_embedder.named_parameters():
        tensor_names.append(name)
    for name, _ in model.decoder.named_parameters():
        tensor_names.append(name)

    idx = 0
    for name, array in zip(names, arrays):

        processing = (idx + 1.0) / len(names)
        idx += 1
        sys.stdout.write(f"\rLoading checkpoint: {processing:.1%}")
        sys.stdout.flush()

        if name in global_tensor_map:
            v_name = global_tensor_map[name]

            if name == "model/wte":
                pointer = model.word_embedder._embedding
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)

                output_pointer = name_to_variable(
                    model.decoder, "_output_layer.weight")
                assert output_pointer.shape == array.shape
                output_pointer.data = torch.from_numpy(array)
            elif name == "model/wpe":
                pointer = model.pos_embedder._embedding
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
            else:
                pointer = name_to_variable(model.decoder, v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)

        else:
            name_tmp = name.split("/")
            layer_no = name_tmp[1][1:]
            name = "/".join(name_tmp[2:])
            if name in layer_tensor_map:
                v_name = layer_tensor_map[name].format(layer_no)
                pointer = name_to_variable(model.decoder, v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
            elif name in layer_transpose_map:
                v_name = layer_transpose_map[name].format(layer_no)
                pointer = name_to_variable(model.decoder, v_name)
                array_t = np.transpose(array)
                assert pointer.shape == array_t.shape
                pointer.data = torch.from_numpy(array_t)
            elif name == "attn/c_attn/w":
                index_d = array.shape[-1] // 3

                Q_w = np.transpose(array[:, :index_d])
                K_w = np.transpose(array[:, index_d: 2 * index_d])
                V_w = np.transpose(array[:, 2 * index_d:])

                q_weight = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.Q_dense.weight")
                k_weight = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.K_dense.weight")
                v_weight = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.V_dense.weight")

                assert q_weight.shape == Q_w.shape
                assert k_weight.shape == K_w.shape
                assert v_weight.shape == V_w.shape

                q_weight.data = torch.from_numpy(Q_w)
                k_weight.data = torch.from_numpy(K_w)
                v_weight.data = torch.from_numpy(V_w)

            elif name == "attn/c_attn/b":
                d = array.shape[0]
                Q_b = array[: d // 3]
                K_b = array[d // 3: 2 * d // 3]
                V_b = array[2 * d // 3:]
                q_bias = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.Q_dense.bias")
                k_bias = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.K_dense.bias")
                v_bias = name_to_variable(
                    model.decoder, f"self_attns.{layer_no}.V_dense.bias")

                assert q_bias.shape == Q_b.shape
                assert k_bias.shape == K_b.shape
                assert v_bias.shape == V_b.shape

                q_bias.data = torch.from_numpy(Q_b)
                k_bias.data = torch.from_numpy(K_b)
                v_bias.data = torch.from_numpy(V_b)

            else:
                print("Name error", name)
                raise Exception


def name_to_variable(model, name):
    pointer = model
    name = name.split(".")
    for m_name in name:
        if m_name.isdigit():
            num = int(m_name)
            pointer = pointer[num]
        else:
            pointer = getattr(pointer, m_name)
    return pointer
