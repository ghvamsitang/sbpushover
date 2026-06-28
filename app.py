#!/usr/bin/env python3
"""RC Beam Pushover Analysis - Desktop Application.

Performs nonlinear pushover analysis of a simply supported reinforced
concrete beam with top and bottom reinforcement layers.
"""

from src.gui import PushoverApp


def main():
    app = PushoverApp()
    app.run()


if __name__ == "__main__":
    main()
