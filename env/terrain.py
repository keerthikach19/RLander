"""Terrain model for RLander.

Phase A: flat terrain at y=0.
Phase B stub: procedural heightmap (not implemented yet).
"""

import jax.numpy as jnp


def terrain_height(x: jnp.ndarray) -> jnp.ndarray:
    """Return terrain height at horizontal position x.

    Phase A: flat terrain at y = 0.

    Args:
        x: horizontal position (unused in Phase A)

    Returns:
        terrain height (always 0.0 in Phase A)
    """
    return jnp.zeros_like(x)
