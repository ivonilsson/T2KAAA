"""Test aggregator to run all project tests in one place."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
	"""Run the full pytest suite when executed directly."""

	tests_dir = PROJECT_ROOT / "tests"
	return pytest.main([str(tests_dir)])


if __name__ == "__main__":
	raise SystemExit(main())
