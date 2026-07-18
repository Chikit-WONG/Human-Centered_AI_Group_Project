#!/usr/bin/env python3
"""Disabled compatibility entry point for the unsafe legacy downloader."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This legacy entry point is disabled because it can expose unverified "
        "partial files. Run download_v2_5_safe.py instead."
    )


if __name__ == "__main__":
    main()
