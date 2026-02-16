"""
Entry point for running the pipeline as a module:
    python -m src path/to/stereo.wav --output results.json --stage 3
"""

from .pipeline import main

main()
