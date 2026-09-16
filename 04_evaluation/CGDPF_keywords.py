#!/usr/bin/env python3
"""CGDPF using cleaned descriptions plus selected KeyBERT keywords."""

from CGDPF import run_pipeline


if __name__ == "__main__":
    run_pipeline(use_keywords=True)
