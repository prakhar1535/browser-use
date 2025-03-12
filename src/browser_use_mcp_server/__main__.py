"""
Main module for browser-use-mcp-server.
"""

import sys
from .cli import cli

if __name__ == "__main__":
    sys.exit(cli())
