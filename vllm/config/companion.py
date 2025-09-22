"""Configuration for companion processes used in IPC model loading."""

from dataclasses import dataclass


@dataclass
class CompanionConfig:
    """Configuration for companion processes.
    
    Companion processes are used for IPC (Inter-Process Communication) model
    loading, allowing model weights to be shared across multiple worker 
    processes without duplicating memory.
    """
    
    coordinator_port: int = 55800
    """Port for the companion coordinator service.
    
    This is the port where the companion coordinator listens for client
    connections. The coordinator always runs on localhost. Having a fixed 
    default port simplifies configuration across distributed setups.
    
    The companion master port (for coordinator-to-server communication) is
    specified via CLI arguments when launching the companion.
    """
    
    # Add any other companion-specific settings here in the future
    # For example:
    # companion_timeout: int = 60  # Timeout for companion operations
    # companion_cache_size: int = 1024  # Cache size for companion data
