"""int8 keeps the model's outputs; with an outlier, per-channel beats naive per-tensor."""

import torch

from src.evaluation.downstream import fake_quantize_linear_weights, output_drift
from src.models.small_lm import make_tiny_model, random_inputs


def test_int8_per_channel_preserves_outputs():
    model = make_tiny_model(seed=0)
    inputs = random_inputs(seed=1)
    quantized = fake_quantize_linear_weights(
        model, bits=8, scheme="absmax", granularity="per_channel", axis=1
    )
    drift = output_drift(model, quantized, inputs)
    assert drift["cosine"] > 0.999


def test_per_channel_beats_naive_per_tensor_with_outlier():
    model = make_tiny_model(seed=0)
    inputs = random_inputs(seed=1)

    # Plant an outlier into the first linear layer so naive per-tensor int4 must stretch.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight[0, 0] = 20.0 * float(module.weight.abs().max())
                break

    q_tensor = fake_quantize_linear_weights(model, bits=4, granularity="per_tensor")
    q_channel = fake_quantize_linear_weights(model, bits=4, granularity="per_channel", axis=1)

    drift_tensor = output_drift(model, q_tensor, inputs)
    drift_channel = output_drift(model, q_channel, inputs)

    assert drift_channel["cosine"] > drift_tensor["cosine"]
