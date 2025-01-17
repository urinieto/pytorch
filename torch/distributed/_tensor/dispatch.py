# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Callable, cast, Dict, Tuple, Union, Optional

import torch

import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.op_schema import (
    ArgsType,
    KwargsType,
    OutputSpecType,
)
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed._tensor.sharding_prop import ShardingPropagator
from torch.distributed._tensor.redistribute import redistribute_dtensor
from torch.utils._pytree import tree_flatten, tree_unflatten


"""
If _ENABLE_FALLBACK set to False, dispatch will fail when an op doesn't
have a sharding rule registered.
"""
_ENABLE_FALLBACK = False


def wrap(res: object, spec: OutputSpecType) -> object:
    if isinstance(res, torch.Tensor):
        assert spec is not None and isinstance(
            spec, DTensorSpec
        ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
        assert spec.tensor_meta is not None
        return dtensor.DTensor(
            res,
            spec.mesh,
            spec.placements,
            shape=spec.tensor_meta.shape,
            dtype=spec.tensor_meta.dtype,
            requires_grad=res.requires_grad,
            stride=spec.tensor_meta.stride,
        )
    elif isinstance(res, (list, tuple)):
        assert spec is not None and isinstance(
            spec, (list, tuple)
        ), f"output spec does not match with output! Expected list/tuple, got {spec}."
        res_list = []
        for e, s in zip(res, spec):
            # NOTE: local results might return Optional Tensor from ATen op, so we need
            # to handle that case and make sure we don't wrap None with DTensor.
            # (i.e. native_layer_norm.backward)
            if e is not None and s is not None:
                assert s.tensor_meta is not None
                res_dt = dtensor.DTensor(
                    e,
                    s.mesh,
                    s.placements,
                    shape=s.tensor_meta.shape,
                    dtype=s.tensor_meta.dtype,
                    requires_grad=s.tensor_meta.requires_grad,
                    stride=s.tensor_meta.stride
                )
            else:
                res_dt = None

            res_list.append(res_dt)
        return tuple(res_list) if isinstance(res, tuple) else res_list
    else:
        # if the res contains only non tensor values, we simply return it without rewrapping
        return res


def pack_args_kwargs_with_local_tensor(
    args: Union[ArgsType, KwargsType],
    args_schema: Union[ArgsType, KwargsType],
    redistribute_with_schema: bool = False,
) -> Union[ArgsType, KwargsType]:
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema, _ = tree_flatten(args_schema)

    for i, arg in enumerate(flatten_args):
        if isinstance(arg, dtensor.DTensor):
            if redistribute_with_schema:
                target_spec = flatten_args_schema[i]
                arg = redistribute_dtensor(
                    arg, target_spec.mesh, target_spec.placements
                )

            # reuse the schema list and update it with local tensor
            flatten_args_schema[i] = arg._local_tensor

    return tree_unflatten(flatten_args_schema, args_tree_spec)


def _reshape_alias(
    x: torch.Tensor, shape: Tuple[int, ...], strides: Tuple[int, ...]
) -> torch.Tensor:
    return torch.ops.aten.view(x, shape)


_CURRENT_DECOMPOSITION_TABLE: Dict[Callable[..., object], Callable[..., object]] = {
    torch.ops.aten._reshape_alias.default: _reshape_alias,
}


def operator_dispatch(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    sharding_propagator: ShardingPropagator,
    custom_dispatch_ops: Optional[Dict[str, Callable[..., object]]] = None,
) -> object:
    # check that we are not getting mixed vanilla and Distributed tensors
    arg_list, _ = tree_flatten(args)
    mesh = None
    for arg in arg_list:
        if isinstance(arg, torch.Tensor) and not isinstance(arg, dtensor.DTensor):
            raise RuntimeError(
                f"{op_call}: got mixed torch.Tensor and DTensor, need to convert all"
                " torch.Tensor to DTensor before calling distributed operators!"
            )

        if isinstance(arg, dtensor.DTensor):
            if mesh is not None:
                if mesh != arg.device_mesh:
                    raise NotImplementedError(
                        f"{op_call}: DTensor does not support cross-mesh operation yet!"
                    )
            else:
                mesh = arg.device_mesh

    # first we need to lift some private aten aliases to public calls
    if op_call in _CURRENT_DECOMPOSITION_TABLE:
        return _CURRENT_DECOMPOSITION_TABLE[op_call](*args, **kwargs)

    # STEP 0. See if there's a user defined custom aten operator
    # implementations. Custom operators take the highest priority
    if custom_dispatch_ops is not None and str(op_call) in custom_dispatch_ops:
        # dispatch to user defined custom distributed tensor ops
        return custom_dispatch_ops[str(op_call)](*args, **kwargs)

    # unwrap the args/kwargs schema
    op_schema = sharding_propagator.prepare_op_schema(op_call, args, kwargs)

    output_sharding = sharding_propagator.propagate_op_sharding(op_call, op_schema)

    # if the schema suggestion from sharding prop is not the same instance as the
    # input op_schema, it indicates a reshard, we need to redistribute the input
    # tensors before calling the local op
    assert output_sharding.schema_suggestions is not None
    suggested_input_schema = output_sharding.schema_suggestions[0]
    needs_redistribute = suggested_input_schema is not op_schema

    if mesh is not None and mesh.get_coordinate() is None:
        # if we are on a non-participating device, we simply return
        # an empty tensor for now.
        # TODO: what if the op returns a non-tensor value, what if
        # the op returns a list of tensors, we need to figure out
        # a consistent way to handle that, and also need to figure
        # out if we should communicate the result to non-participating
        # ranks (i.e. a.sum() -> scalar, maybe we should set to 0)
        local_results = torch.tensor([])
    else:
        # compute locally with redistribute first if needed
        local_tensor_args = pack_args_kwargs_with_local_tensor(
            args,
            suggested_input_schema.args_schema,
            redistribute_with_schema=needs_redistribute,
        )
        local_tensor_kwargs = pack_args_kwargs_with_local_tensor(
            kwargs,
            suggested_input_schema.kwargs_schema,
            redistribute_with_schema=needs_redistribute,
        )

        # run local op computation with potentially modified args/kwargs
        local_tensor_args = cast(Tuple[object, ...], local_tensor_args)
        local_tensor_kwargs = cast(Dict[str, object], local_tensor_kwargs)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

    if suggested_input_schema.is_inplace:
        # inplace op should return self instead of re-wrapping
        self = cast(dtensor.DTensor, args[0])
        self._spec = cast(DTensorSpec, output_sharding.output_spec)
        return self
    elif suggested_input_schema.is_out_variant:
        # out variant could possibly have multiple out args (i.e. lu_unpack.out)
        output_specs = (
            (output_sharding.output_spec,)
            if not isinstance(output_sharding.output_spec, tuple)
            else output_sharding.output_spec
        )
        out_dts = []
        spec_idx = 0
        for arg in suggested_input_schema.func_schema.arguments:
            if arg.is_out:
                out_dt = cast(dtensor.DTensor, kwargs[arg.name])
                out_dt._spec = cast(DTensorSpec, output_specs[spec_idx])
                out_dts.append(out_dt)
                spec_idx += 1

        assert len(out_dts) >= 1, "out variant should have at least one out arg"
        return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
    else:
        return wrap(local_results, output_sharding.output_spec)
