"""Helpers for running IDM-VTON try-on flows."""

from __future__ import annotations

from typing import Any

__all__ = ["run_idm_vton", "IDMVTONTryOn"]


def __getattr__(name: str) -> Any:
	if name == "run_idm_vton":
		from .pipeline import run_idm_vton as _run_idm_vton

		return _run_idm_vton
	if name == "IDMVTONTryOn":
		from .runner import IDMVTONTryOn as _IDMVTONTryOn

		return _IDMVTONTryOn
	raise AttributeError(name)
