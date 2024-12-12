import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from typing import AsyncIterator, Tuple
from .dataset import FIDataset
from .metrics import compute_score

BATCH_SIZE = 32
InputType = np.ndarray
OutputType = np.ndarray
BatchType = Tuple[InputType, OutputType]


async def batches(config: dict) -> AsyncIterator[BatchType]:
    dataset = FIDataset("test", config["data"])
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    for input, expected in loader:
        input = input.numpy()
        expected = expected.numpy()
        yield input, expected


async def score(
    solution: OutputType,
    original_input: BatchType,
) -> float:
    return compute_score(Tensor(solution), Tensor(original_input[1]))


async def aggregate(outputs: AsyncIterator) -> float:
    total = 0
    count = 0
    async for output in outputs:
        solve_out = output.get("solve")
        if solve_out:
            solve_out = solve_out[0]
            total += solve_out.metric
            count += len(solve_out.output)
        else:
            return -1000

    return total / count if count else -1000
