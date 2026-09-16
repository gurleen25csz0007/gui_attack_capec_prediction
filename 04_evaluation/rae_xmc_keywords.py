#!/usr/bin/env python3
"""RAE-XMC using cleaned descriptions plus selected KeyBERT keywords."""

from rae_xmc import run_pipeline


if __name__ == "__main__":
    run_pipeline(use_keyphrases=True)
