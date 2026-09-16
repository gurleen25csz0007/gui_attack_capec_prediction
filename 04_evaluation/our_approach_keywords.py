#!/usr/bin/env python3
"""Our Approach using cleaned descriptions and modified-KeyBERT keywords."""

import sys

from our_approch import main


if __name__ == "__main__":
    if "--keywords" not in sys.argv:
        sys.argv.append("--keywords")
    main()
