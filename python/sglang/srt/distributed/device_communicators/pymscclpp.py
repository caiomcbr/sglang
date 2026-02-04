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
        """Find local interface that matches the given IP exactly."""
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

    def find_interface_in_same_subnet(self, ip: str):
        """Find a local interface that's in the same subnet as the target IP."""
        target = ipaddress.ip_address(ip)
        for interface in ni.interfaces():
            addresses = ni.ifaddresses(interface)
            if ni.AF_INET in addresses:
                for link in addresses[ni.AF_INET]:
                    if "addr" in link and "netmask" in link:
                        addr = ipaddress.ip_address(link["addr"])
                        netmask = link["netmask"]
                        network = ipaddress.ip_network(f"{link['addr']}/{netmask}", strict=False)
                        if target in network:
                            return interface, str(addr)
        return None, None

    def get_local_interface_and_ip(self, master_addr: str):
        """
        Get local interface for multi-node communication.
        Priority:
        1. MSCCLPP_INTERFACE env var (explicit override)
        2. Interface with exact IP match (master node)
        3. Interface in same subnet as master (worker nodes)
        """
        # Allow explicit interface override
        if "MSCCLPP_INTERFACE" in os.environ:
            interface = os.environ["MSCCLPP_INTERFACE"]
            # Get the IP of this interface
            addresses = ni.ifaddresses(interface)
            if ni.AF_INET in addresses:
                local_ip = addresses[ni.AF_INET][0]["addr"]
                return interface, local_ip
            raise ValueError(f"Interface {interface} has no IPv4 address")

        # Try exact IP match (for master node)
        interface = self.interfaces_for_ip_netifaces(master_addr)
        if interface is not None:
            return interface, master_addr

        # Find interface in same subnet (for worker nodes)
        interface, local_ip = self.find_interface_in_same_subnet(master_addr)
        if interface is not None:
            return interface, local_ip

        raise ValueError(f"Cannot find network interface for IP address {master_addr}")

    def _creating_dsl_algorithms(self):
        dsl_algorithms = []
        if self.world_size // self.nranks_per_node == 2:
            for tbg in [1, 2, 4, 8]:
                for num_threads_per_block in [256, 512, 768, 1024]:
                    spec = self.mscclpp.language.AlgoSpec(
                        name=f"allreduce_1node_{tbg}TBG_{num_threads_per_block}TPB",
                        collective=self.mscclpp.language.collectives.AllReduce(16, 1, True),
                        nranks_per_node=8,
                        world_size=16,
                        in_place=True,
                        instances=1,
                        protocol="LL",
                        auto_sync=False,
                        num_threads_per_block=num_threads_per_block,
                        reuse_resources=True,
                        use_double_scratch_buffer=True,
                        min_message_size=1 << 10,
                        max_message_size=2 << 20,
                        tags={"default": 1},
                    )
                    algo = self.mscclpp.compile(self.def_algo.allreduce_2nodes, spec, self.rank, thread_block_group_size=tbg)
                    dsl_algorithms.append(algo)
        return dsl_algorithms

    def _tune(self, n_warmup, n_graph_launches, n_ops_per_graph, algos):
        sizes = [1 << i for i in range(0, 30)]
        self.best_configs = {1024: (self._algorithm_nvls_packet, 0, 0)}

        tune_tensor = torch.rand(1 << 27, dtype=torch.float16, device="cuda")
        candidates_nblocks = [4, 8, 16, 21, 24, 28, 32, 42, 48, 56, 64, 128]
        candidates_nthreads = [512, 768, 1024]

        for size in sizes:
            best_time = float("inf")
            best_config = None

                if algo.is_native_algorithm():
                    for nb in candidates_nblocks:
                        if algo.name == "default_allreduce_nvls_packet" and (nb > 16):
                            continue
                        if algo.name == "default_allreduce_packet" and nb > 56:
                            continue
                        if algo.name != "default_allreduce_packet" and algo.name != "default_allreduce_nvls_packet":
                            continue

                        for nt in candidates_nthreads:
                            
                            if self._run_algo(algo, tune_tensor, size, nb, nt) != 0:
                                continue

                            for _ in range(n_warmup):
                                self._run_algo(algo, tune_tensor, size, nb, nt)
                            self.barrier()
                            capture_stream = torch.cuda.Stream()
                            capture_stream.wait_stream(torch.cuda.current_stream())

                            g = torch.cuda.CUDAGraph()
                            # Warmup on capture stream
                            with torch.cuda.stream(capture_stream):
                                self._run_algo(algo, tune_tensor, size, nb, nt)
                            capture_stream.synchronize()

                            with torch.cuda.graph(g, stream=capture_stream):
                                for _ in range(n_ops_per_graph):
                                    self._run_algo(algo, tune_tensor, size, nb, nt)

                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            start_event.record(capture_stream)
                            with torch.cuda.stream(capture_stream):
                                for _ in range(n_graph_launches):
                                    g.replay()
                            end_event.record(capture_stream)
                            end_event.synchronize()

                            elapsed = start_event.elapsed_time(end_event)

                            # Synchronize timing results across all ranks to ensure consistent algorithm selection
                            # replicate n times such due to algo limitations
                            time_tensor = torch.full((self.world_size,), elapsed, dtype=torch.float64, device="cuda").to(
                                dtype=torch.float32
                            )
                            torch.cuda.current_stream().wait_stream(capture_stream)
                            self.all_reduce(time_tensor, op=torch.distributed.ReduceOp.SUM)
                            avg_time = time_tensor[self.rank].item() / self.world_size
                            
                            if self.rank == 0:
                                tensor = torch.tensor([avg_time])
                            else:
                                tensor = torch.empty(1)

                            dist.broadcast(tensor, src=0, group=self.group)
                            avg_time = tensor.item()

                            if avg_time < best_time:
                                best_time = avg_time
                                best_config = (algo, nb, nt)
                else:
                    if self._run_algo(algo, tune_tensor, size, 0, 0) != 0:
                        continue

                    for _ in range(n_warmup):
                        self._run_algo(algo, tune_tensor, size, 0, 0)
                    capture_stream = torch.cuda.Stream()
                    capture_stream.wait_stream(torch.cuda.current_stream())

                    g = torch.cuda.CUDAGraph()
                    # Warmup on capture stream
                    with torch.cuda.stream(capture_stream):
                        self._run_algo(algo, tune_tensor, size, 0, 0)
                    capture_stream.synchronize()

                    with torch.cuda.graph(g, stream=capture_stream):
                        for _ in range(n_ops_per_graph):
                            self._run_algo(algo, tune_tensor, size, 0, 0)

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(capture_stream)
                    with torch.cuda.stream(capture_stream):
                        for _ in range(n_graph_launches):
                            g.replay()
                    end_event.record(capture_stream)
                    end_event.synchronize()

                    elapsed = start_event.elapsed_time(end_event)

                    # Synchronize timing results across all ranks to ensure consistent algorithm selection
                    # replicate n times such due to algo limitations
                    time_tensor = torch.full((self.world_size,), elapsed, dtype=torch.float64, device="cuda")
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    dist.all_reduce(time_tensor, op=dist.ReduceOp.SUM)
                    #self.all_reduce(time_tensor, op=torch.distributed.ReduceOp.SUM)
                    avg_time = time_tensor[self.rank].item() / self.world_size
                    
                    self.comm.boots

                    if avg_time < best_time:
                        best_time = avg_time
                        best_config = (algo, 0, 0)

            if best_config:
                self.best_configs[size] = best_config
                if self.rank == 0:
                    print(
                        f"Size {size}: Best Algo {best_config[0].name} nblocks {best_config[1]} nthreads {best_config[2]} Time {(best_time/(n_graph_launches * n_ops_per_graph))*1000:.2f} us"
                    )

        torch.cuda.synchronize()
        for algo in algos:
            algo.reset()


    def _run_algo(self, algo, tensor, size, nblocks, nthreads):
        return algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=size,
            output_size=size,
            dtype=self.dtype_to_mscclpp_dtype(tensor.dtype),
            op=self.mscclpp.ReduceOp.SUM,
            stream=torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
        )

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
            self.def_algo = importlib.import_module("mscclpp.default_algos")
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

        interface, local_ip = self.get_local_interface_and_ip(master_addr)
        interfaceIpPortTrio = f"{interface}:{master_addr}:{master_port}"
        self.comm = self.mscclpp.CommGroup(interfaceIpPortTrio=interfaceIpPortTrio, rank=rank, size=world_size)
        self.executor = self.mscclpp.Executor(self.comm.communicator)

        self.dsl_algos =  self._creating_dsl_algorithms()
        dlpack = self.mscclpp.RawGpuBuffer(1 << 27).to_dlpack(data_type=str(torch.float16))
        self.scratch_buffer = torch.utils.dlpack.from_dlpack(dlpack)
        self.flag_buffer = torch.ones(128, dtype=torch.uint32, device="cuda")
        self.algorithms = self.mscclpp_ext.AlgorithmCollectionBuilder().build_default_algorithms(
            scratch_buffer=self.scratch_buffer.data_ptr(), flag_buffer=self.flag_buffer.data_ptr(), flag_buffer_size=self.flag_buffer.nbytes, scratch_buffer_size=self.scratch_buffer.nbytes, rank=self.rank
        )
        self._algorithm_nvls_packet = [
            algo
            for algo in self.algorithms
            if algo.collective == "allreduce" and algo.name == "default_allreduce_packet"
        ][0]
        self.best_configs = {}
        
        if world_size == 8:
            self._tune(5, 10, 20, self.algorithms)
        elif world_size == 16:
            self._tune(5, 10, 20, self.dsl_algos)

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

    def get_tuned_config(self, size):
        if size <= 1024:
            target_size = 1024
        elif size > 256 * 1024 * 1024:
            target_size = 256 * 1024 * 1024
        else:
            target_size = 1 << (size - 1).bit_length()
        return self.best_configs.get(target_size)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream: torch.cuda.Stream = None):
        assert op == torch.distributed.ReduceOp.SUM
        algo, nblocks, nthreads = self.get_tuned_config(tensor.nbytes)
        result = torch.empty_like(tensor)
        
        print(f"Using MSCClpp all_reduce with algo: {self.rank} {algo.name}, nblocks: {nblocks}, nthreads: {nthreads} for size: {tensor.nbytes} bytes", flush=True)

        if algo.execute(
            comm=self.comm.communicator,
            executor=self.executor,
            input_buffer=tensor.data_ptr(),
            output_buffer=tensor.data_ptr(),
            input_size=tensor.nbytes,
            output_size=tensor.nbytes,
            dtype=self.dtype_to_mscclpp_dtype(tensor.dtype),
            op=self.mscclpp.ReduceOp.SUM,
            stream=stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads,
        ) != 0:
            raise RuntimeError("MSCClpp all_reduce failed")

        return tensor

    def barrier(self):
        tensor = torch.empty(self.world_size, dtype=torch.float, device=torch.device("cuda"))
        #dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        self.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

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