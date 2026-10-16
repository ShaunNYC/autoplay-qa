"""Explicit version dispatch for opt-in exploration and historical replay."""

ADAPTIVE_VERSIONS = ("adaptive-v1", "adaptive-v2", "adaptive-v3", "adaptive-v4", "focused-v1")


def context_class(version: str):
    if version == "adaptive-v1":
        from .adaptive import AdaptiveContext
        return AdaptiveContext
    if version == "adaptive-v2":
        from .adaptive_v2 import AdaptiveContextV2
        return AdaptiveContextV2
    if version == "adaptive-v3":
        from .adaptive_v3 import AdaptiveContextV3
        return AdaptiveContextV3
    if version == "adaptive-v4":
        from .adaptive_v4 import AdaptiveContextV4
        return AdaptiveContextV4
    if version == "focused-v1":
        from .focused import FocusedContext
        return FocusedContext
    raise ValueError("Unsupported adaptive exploration protocol.")
