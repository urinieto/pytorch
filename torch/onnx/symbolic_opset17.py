"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

import functools
from typing import Optional, Sequence

import torch
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = ["layer_norm"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)


@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
):
    # normalized_shape: input shape from an expected input of size
    # axis: The first normalization dimension.
    # layer_norm normalizes on the last D dimensions,
    # where D is the size of normalized_shape
    axis = -len(normalized_shape)
    return g.op(
        "LayerNormalization",
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
    )


@_onnx_symbolic("aten::stft")
@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b")
@_beartype.beartype
def stft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[_C.Value] = None,
    normalized: bool = False,
    onesided: Optional[bool] = True,
    return_complex: Optional[bool] = False,
) -> _C.Value:
    """Associates `torch.stft` with the `STFT` ONNX operator.

    Note that torch.stft calls _VF.stft, without centering or padding options.
    Hence, this function does not contain these two arguments.
    See torch.stft source code for more info.

    Parameters
    ----------
    g : jit_utils.GraphContext
        Graph to write the ONNX representation into
    input : _C.Value
        Input tensor for the transformation
    n_fft : int
        FFT size
    hop_length : int, optional
        Size of the hop. Defaults to floot(n_fft // 4)
    win_length : int, optional
        Size of the analysis window. Defaults to `n_fft`
    window : _C.Value, optional
        Analysis window. Defaults to a window of all ones.
    normalized : bool, optional
        Whether to return a normalized STFT.
    onesided : bool, optional
        Whether to return only half (+1) of the results, given the symmetry
        of the STFT
    return_complex : bool, optional
        Whether to return the complex value (Note: Must be False or None)

    Returns
    -------
    op
        Operator for torch.stft associated with STFT (ONNX).
    """
    # Checks
    assert (
        return_complex is None or not return_complex,
        "STFT does not currently support complex types"
    )

    # Get STFT sizes
    frame_step = hop_length if not hop_length is None else n_fft // 4
    o_frame_step = g.op(
        "Constant", value_t=torch.tensor(frame_step, dtype=torch.int64)
    )
    o_frame_length = g.op(
        "Constant", value_t=torch.tensor(n_fft, dtype=torch.int64)
    )

    # Pre-process input if needed
    o_signal = input
    signal_rank = symbolic_helper._get_tensor_rank(o_signal)
    if signal_rank == 1:
        # Add batch dimension
        o_signal = g.op(
            "Unsqueeze",
            o_signal,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
        )
    elif signal_rank > 2:
        raise RuntimeError("STFT can only take inputs of 1 [signal] or 2 "
                           "[batch, signal] dimensions")

    # Get window and make sure it's the same size as the FFT size
    o_window = window
    n_win = symbolic_helper._get_tensor_dim_size(o_window, dim=0)
    if n_win is not None:
        assert (
            n_win == n_fft,
            "Analysis window's size must the same as the FFT size"
        )

    # Create window, if needed
    if symbolic_helper._is_none(window):
        if win_length:
            assert (
                win_length <= n_fft,
                "The analysis window can't be longer than the size of the FFT"
            )
            # Center window, if needed
            left = (n_fft - win_length) // 2
            right = n_fft - left - win_length
            torch_window = torch.hstack((
                torch.zeros((left)),
                torch.ones((win_length)),
                torch.zeros((right))
            ))
        else:
            # Rectangle window
            torch_window = torch.ones((n_fft))
        assert torch_window.shape[0] == n_fft
        o_window = g.op("Constant", value_t=torch_window)

    # Run STFT
    stft_op = g.op(
        "STFT",
        o_signal,
        o_frame_step,
        o_window,
        o_frame_length,
        onesided_i=1 if onesided is None or onesided else 0
    )

    # Transpose to mimic torch.stft's behavior
    stft_op = g.op("Transpose", stft_op, perm_i=[0, 2, 1, 3])

    # Remove batch dimension, if needed
    if signal_rank == 1:
        stft_op = g.op(
            "Squeeze",
            stft_op,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
        )

    # Normalize, if needed
    if normalized:
        sqrt_nfft = torch.sqrt(
            torch.tensor(n_fft, dtype=o_signal.type().dtype())
        )
        stft_op = g.op(
            "Div",
            stft_op,
            g.op("Constant", value_t=sqrt_nfft)
        )

    return stft_op