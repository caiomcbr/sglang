import bisect
import importlib
import logging
import math
import os
import ipaddress
import netifaces as ni
from contextlib import contextmanager
from enum import IntEnum
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops

logger = logging.getLogger(__name__)

def mscclpp_is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def mscclpp_convert_to_bytes(size_str):
    """
    Converts a human-readable size string (e.g., "1MB", "2.5kb", "3 GB")
    into the equivalent number of bytes using binary units.

    Args:
        size_str (str): A string representing size with unit (KB, MB, GB).

    Returns:
        int: Number of bytes.
    """
    size_str = size_str.strip().lower()

    if not size_str:
        raise ValueError("Empty input string")

    # Extract numeric part and unit
    for i in range(len(size_str)):
        if not size_str[i].isdigit() and size_str[i] != ".":
            break
    num_str = size_str[:i]
    unit = size_str[i:].strip()

    try:
        num = float(num_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value in '{size_str}'")

    # Conversion factors
    if unit == "b":
        return int(num)
    elif unit == "kb":
        return int(num * 1024)
    elif unit == "mb":
        return int(num * 1024 * 1024)
    elif unit == "gb":
        return int(num * 1024 * 1024 * 1024)
    else:
        raise ValueError(f"Unsupported unit: {unit}, support B, KB, MB, GB only")


class PyMscclppCommunicator:
    _SUPPORTED_WORLD_SIZES = [8, 16]
    _MAX_BYTES = mscclpp_convert_to_bytes(os.getenv("SGLANG_MSCCLPP_MAX_BYTES", "1MB"))
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]

    def interfaces_for_ip_netifaces(self, ip: str):
        target = ipaddress.ip_address(ip)
        for interface in ni.interfaces():
            addresses = ni.ifaddresses(interface)
            if ni.AF_INET in addresses:
                for link in addresses[ni.AF_INET]:
                    if "addr" in link:
                        addr = ipaddress.ip_address(link["addr"])
                        if addr == target:
                            return interface
        return None

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_bytes=_MAX_BYTES,
    ) -> None:
        
        """ Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        try:
            self.mscclpp = importlib.import_module("mscclpp")
            self.mscclpp_ext = importlib.import_module("mscclpp.ext")
        except ImportError:
            self.available = False
            self.mscclpp = None
            return

        self.available = True
        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize mscclpp for single GPU case.
            return

        if world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_mscclpp=True explicitly.",
                world_size,
                str(PyMscclppCommunicator._SUPPORTED_WORLD_SIZES),
            )
            return

        master_addr = os.environ["MSCCLPP_MASTER_ADDR"]
        master_port = os.environ["MSCCLPP_MASTER_PORT"]

        self.ranks = torch.distributed.get_process_group_ranks(group)
        self.nranks_per_node = torch.cuda.device_count()
        # for now mscclpp with stride in the communicator is not tested
        if not (abs(self.ranks[-1] - self.ranks[0]) == world_size - 1):
            logger.warning(
                "PyMscclpp is disabled due to an unsupported group %s."
                "Please ensure all ranks in the group are consecutive."
                "To silence this warning, specify disable_mscclpp=True explicitly.",
                str(self.ranks),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        self.max_bytes = max_bytes
        self.rank = rank
        self.world_size = world_size

        interface = self.interfaces_for_ip_netifaces(master_addr)
        if interface is None:
            raise ValueError(f"Cannot find network interface for IP address {master_addr}")
        interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
        self.comm = self.mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world_size)
        self.executor = self.mscclpp.Executor(self.comm.communicator)

        dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        self.algorithms = self.mscclpp_ext.AlgorithmCollectionBuilder().build_default_algorithms(
            scratch_buffer=self.scratch_buffer.data_ptr(), scratch_buffer_size=self.scratch_buffer.nbytes, rank=self.rank
        )

    def destroy(self):
        self.algorithms = None
        self.executor = None
        self.scratch_buffer = None
        self.comm = None

    def should_mscclpp_allreduce(
        self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if not mscclpp_is_weak_contiguous(inp):
            return False
        # only support sum op
        if op != ReduceOp.SUM:
            return False
        if inp.numel() * inp.element_size() > self.max_bytes:
            return False
        return True

    def dtype_to_mscclpp_dtype(self, dtype: torch.dtype):
        if dtype == torch.float16:
            return self.mscclpp.DataType.float16
        elif dtype == torch.float32:
            return self.mscclpp.DataType.float32
        elif dtype == torch.int32:
            return self.mscclpp.DataType.int32
        elif dtype == torch.bfloat16:
            return self.mscclpp.DataType.bfloat16
        else:
            raise ValueError(f"Unknown data type: {dtype}")

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        assert op == torch.distributed.ReduceOp.SUM
        algo = self.algorithms[8]
        current_stream = torch.cuda.current_stream()
        result = torch.empty_like(tensor)
        nblocks = 0
        nthreads_per_block = 0
        
        if tensor.nbytes <= (1 << 15):
            algo = self.algorithms[7]
            if tensor.nbytes <= (1 << 13):
                nblocks = 32
                nthreads_per_block = 768
            else:
                nblocks = 24
                nthreads_per_block = 1024
        else:
            algo = self.algorithms[4]
            if tensor.nbytes <= (1 << 18):
                nblocks = 42
                nthreads_per_block = 512
            elif tensor.nbytes < (1 << 20):
                nblocks = 56
                nthreads_per_block = 512

        algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=result.data_ptr(),
            input_size=tensor.nbytes,
            output_size=result.nbytes,
            dtype=self.dtype_to_mscclpp_dtype(tensor.dtype),
            op=self.mscclpp.ReduceOp.SUM,
            stream=current_stream.cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads_per_block
        )

        return result

    def barrier_cpu(self):
        self.comm.barrier()

    @contextmanager
    def change_state(
        self,
        enable: Optional[bool] = None,
    ):
        if enable is None or self.available is False:
            # guess a default value when not specified
            # DO: Decided if raise an exception here or not
            enable = self.available

        old_disable = self.disabled
        self.disabled = not enable

        yield

        self.disabled = old_disable