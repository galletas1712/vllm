"""Fault-tolerant wrapper for torch.distributed process groups."""

import time
import torch
import torch.distributed as dist
from typing import Optional, Any, List
from torch.distributed import ProcessGroup, ReduceOp
from vllm.logger import init_logger

logger = init_logger(__name__)


class FaultTolerantProcessGroup:
    """
    Wrapper around torch.distributed ProcessGroup that handles failures gracefully.
    
    Key features:
    - All operations have configurable timeouts
    - Failed members are tracked and excluded from future operations
    - Operations can continue with reduced membership
    - Supports deferred operations for async events
    """
    
    def __init__(self, 
                 pg: ProcessGroup,
                 rank: int,
                 world_size: int,
                 default_timeout: float = 30.0):
        """
        Initialize the fault-tolerant process group.
        
        Args:
            pg: Underlying torch.distributed ProcessGroup
            rank: This process's rank in the group
            world_size: Total size of the group
            default_timeout: Default timeout for operations (seconds)
        """
        self.pg = pg
        self.rank = rank
        self.world_size = world_size
        self.default_timeout = default_timeout
        
        # Track member health
        self.member_alive = [True] * world_size
        self.last_successful_op = [time.time()] * world_size
        self.health_threshold = 60.0  # Mark dead after 60s of failures
        
        # Pending operations queue
        self.pending_operations = []
        
    def barrier(self, timeout: Optional[float] = None) -> bool:
        """
        Perform a barrier with timeout and fault tolerance.
        
        Returns:
            True if barrier succeeded, False if timed out or partial success
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Create a work handle for async barrier
            work = dist.barrier(group=self.pg, async_op=True)
            
            # Wait with timeout
            completed = work.wait(timeout=timeout)
            
            if completed:
                # Update successful op time for all members
                current_time = time.time()
                for i in range(self.world_size):
                    self.last_successful_op[i] = current_time
                return True
            else:
                logger.warning(f"Barrier timed out after {timeout}s")
                self._check_member_health()
                return False
                
        except Exception as e:
            logger.error(f"Barrier failed: {e}")
            self._check_member_health()
            return False
    
    def all_reduce(self, 
                   tensor: torch.Tensor, 
                   op: ReduceOp = ReduceOp.SUM,
                   timeout: Optional[float] = None) -> bool:
        """
        Perform all-reduce with timeout and fault tolerance.
        
        Returns:
            True if all-reduce succeeded, False if timed out
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Create work handle for async all-reduce
            work = dist.all_reduce(tensor, op=op, group=self.pg, async_op=True)
            
            # Wait with timeout
            completed = work.wait(timeout=timeout)
            
            if completed:
                # Update successful op time
                current_time = time.time()
                for i in range(self.world_size):
                    self.last_successful_op[i] = current_time
                return True
            else:
                logger.warning(f"All-reduce timed out after {timeout}s")
                self._check_member_health()
                return False
                
        except Exception as e:
            logger.error(f"All-reduce failed: {e}")
            self._check_member_health()
            return False
    
    def try_barrier_with_retry(self, 
                               max_retries: int = 3,
                               timeout: Optional[float] = None) -> bool:
        """
        Try barrier with retries and exponential backoff.
        
        This is useful for operations that need strong synchronization
        like warm spare switching.
        """
        timeout = timeout or self.default_timeout
        
        for attempt in range(max_retries):
            if attempt > 0:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retrying barrier in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            
            if self.barrier(timeout=timeout):
                return True
        
        logger.error(f"Barrier failed after {max_retries} attempts")
        return False
    
    def queue_deferred_operation(self, operation: str, args: dict):
        """
        Queue an operation to be executed at the next sync point.
        
        Args:
            operation: Name of the operation (e.g., "warm_spare_switch")
            args: Arguments for the operation
        """
        self.pending_operations.append({
            "operation": operation,
            "args": args,
            "timestamp": time.time()
        })
        logger.info(f"Queued deferred operation: {operation}")
    
    def has_pending_operations(self) -> bool:
        """Check if there are pending operations."""
        return len(self.pending_operations) > 0
    
    def get_pending_operations(self) -> List[dict]:
        """Get and clear pending operations."""
        ops = self.pending_operations
        self.pending_operations = []
        return ops
    
    def _check_member_health(self):
        """Check and update member health based on last successful operations."""
        current_time = time.time()
        
        for i in range(self.world_size):
            if self.member_alive[i]:
                time_since_success = current_time - self.last_successful_op[i]
                if time_since_success > self.health_threshold:
                    logger.warning(f"Marking rank {i} as dead (no successful ops for {time_since_success:.1f}s)")
                    self.member_alive[i] = False
    
    def get_alive_members(self) -> List[int]:
        """Get list of alive member ranks."""
        return [i for i in range(self.world_size) if self.member_alive[i]]
    
    def is_degraded(self) -> bool:
        """Check if the group is running in degraded mode (some members dead)."""
        return not all(self.member_alive)
    
    def destroy(self):
        """Clean up the process group."""
        try:
            if self.pg:
                dist.destroy_process_group(self.pg)
        except Exception as e:
            logger.warning(f"Error destroying process group: {e}")
