import math

import numpy as np
from numba_cuda_mlir import cuda  # pyright: ignore[reportMissingImports]
from numba_cuda_mlir.numba_cuda.cudadrv.error import (  # pyright: ignore[reportMissingImports]
    CCSupportError,
    CudaDriverError,
    CudaRuntimeError,
    CudaSupportError,
    NvrtcError,
    NvvmError,
)
from numpy.typing import NDArray

CUDA_EXCEPTIONS = (
    CCSupportError,
    CudaDriverError,
    CudaRuntimeError,
    CudaSupportError,
    NvrtcError,
    NvvmError,
    RuntimeError,
)


@cuda.jit
def paint_lyapunov_cuda_kernel(
    points: NDArray[np.complex128],
    ans: NDArray[np.uint32],
    sequence: NDArray[np.uint8],
    warmup: int,
    iterations: int,
    color_scale: float,
    numerical_failures: NDArray[np.int32],
) -> None:
    point_index = cuda.grid(1)  # pyright: ignore[reportAttributeAccessIssue]
    if point_index >= len(points):
        return

    point = points[point_index]
    parameter_a = point.real
    parameter_b = point.imag
    sequence_length = len(sequence)
    x = 0.5
    valid = math.isfinite(parameter_a) and math.isfinite(parameter_b)

    for iteration in range(warmup):
        parameter = parameter_a if sequence[iteration % sequence_length] == 0 else parameter_b
        x = parameter * x * (1.0 - x)
        if not math.isfinite(x):
            valid = False
            break

    exponent_sum = 0.0
    is_superstable = False
    if valid:
        for iteration in range(iterations):
            sequence_index = (warmup + iteration) % sequence_length
            parameter = parameter_a if sequence[sequence_index] == 0 else parameter_b
            derivative = abs(parameter * (1.0 - 2.0 * x))
            if derivative == 0.0:
                is_superstable = True
            elif not math.isfinite(derivative):
                valid = False
                break
            elif not is_superstable:
                exponent_sum += math.log(derivative)
            x = parameter * x * (1.0 - x)
            if not math.isfinite(x):
                valid = False
                break

    if not valid:
        ans[point_index] = 0
        cuda.atomic.add(numerical_failures, 0, 1)  # pyright: ignore[reportAttributeAccessIssue]
    elif is_superstable:
        ans[point_index] = np.uint32(0xFFFFFFFF)
    else:
        exponent = exponent_sum / iterations
        magnitude_bin = math.floor(abs(exponent) * color_scale)
        magnitude_bin = min(magnitude_bin, 0x7FFFFFFE)
        ans[point_index] = np.uint32(2 * magnitude_bin + (1 if exponent < 0.0 else 2))


def paint_lyapunov_cuda(
    points: NDArray[np.complex128],
    ans: NDArray[np.uint32],
    sequence: NDArray[np.uint8],
    warmup: int,
    iterations: int,
    color_scale: float,
) -> int:
    if not len(points):
        return 0

    try:
        device_points = cuda.to_device(points)  # pyright: ignore[reportAttributeAccessIssue]
        device_ans = cuda.device_array_like(ans)  # pyright: ignore[reportAttributeAccessIssue]
        device_sequence = cuda.to_device(sequence)  # pyright: ignore[reportAttributeAccessIssue]
        device_failures = cuda.to_device(np.zeros(1, dtype=np.int32))  # pyright: ignore[reportAttributeAccessIssue]
        threads_per_block = 256
        blocks_per_grid = (len(points) + threads_per_block - 1) // threads_per_block
        paint_lyapunov_cuda_kernel[blocks_per_grid, threads_per_block](  # pyright: ignore[reportIndexIssue]
            device_points,
            device_ans,
            device_sequence,
            warmup,
            iterations,
            color_scale,
            device_failures,
        )
        device_ans.copy_to_host(ans)
        return int(device_failures.copy_to_host()[0])
    except CUDA_EXCEPTIONS as error:
        raise RuntimeError(f"CUDA execution failed: {error}") from error
