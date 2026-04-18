"""Transportation problem builders for inventory optimization."""

from .transportation_instance_builder import (
    TransportationInstance,
    build_transportation_instances,
)

__all__ = ["TransportationInstance", "build_transportation_instances"]
