"""Configuration for companion processes used in IPC model loading."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CompanionConfig:
    """Configuration for companion processes.
    
    Companion processes are used for IPC (Inter-Process Communication) model
    loading, allowing model weights to be shared across multiple worker processes
    without duplicating memory.
    """
    
    companion_master_port: int = 55700
    """Master port for companion process communication.
    
    This port is used for the companion coordinator to communicate with
    companion server processes. Each companion server will get its own
    dynamically allocated port, but they all connect to the coordinator
    on this master port.
    """
    
    coordinator_port: int = 55800
    """Port for the companion coordinator service.
    
    This is the port where the companion coordinator listens for client
    connections. Having a fixed port ensures all ranks can connect to it
    without needing complex coordination.
    """
    
    coordinator_address: Optional[str] = None
    """Address of the companion coordinator.
    
    Format: "tcp://host:port" (e.g., "tcp://127.0.0.1:55800")
    This is automatically set based on coordinator_port when the system starts.
    """
    
    # Add any other companion-specific settings here in the future
    # For example:
    # companion_timeout: int = 60  # Timeout for companion operations
    # companion_cache_size: int = 1024  # Cache size for companion data
