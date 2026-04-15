# Shannon-Prime VHT2: ComfyUI custom node entry point.
# Copyright (C) 2026 Ray Daniels. Licensed under AGPLv3 / Commercial.
#
# Symlink or copy this directory into ComfyUI/custom_nodes/ so ComfyUI
# auto-discovers the nodes on startup.

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"  # reserved for future JS extensions (none today)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
