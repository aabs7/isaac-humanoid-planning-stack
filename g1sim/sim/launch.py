"""App launch + shared CLI. Safe to import *before* the sim app exists
(only touches ``argparse`` and ``AppLauncher``)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# Default spawn: open living-room floor, ~1.2 m clearance from all furniture.
DEFAULT_SPAWN = [7.51, 0.08]


def make_parser(description: str) -> argparse.ArgumentParser:
    """A parser preloaded with the args every entry point shares (``--spawn``)
    plus all of IsaacLab's AppLauncher args. Scripts add their own extras
    (e.g. ``--smoke``) before calling :func:`launch`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--spawn", type=float, nargs=2, default=DEFAULT_SPAWN, metavar=("X", "Y"),
        help="X Y spawn location on the floor in meters (default: open living-room floor).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def launch(args: argparse.Namespace):
    """Launch Isaac Sim and return the running ``simulation_app``.

    Opens the Kit GUI by default for interactive runs. This IsaacLab build is
    headless-by-default and only shows a window when a Kit visualizer is
    requested, so we request one unless the user is headless, running a
    ``--smoke`` self-test, or already chose a visualizer via ``--viz``.
    """
    smoke = getattr(args, "smoke", 0)
    if not args.headless and not smoke and getattr(args, "visualizer", None) is None:
        args.visualizer = ["kit"]
    return AppLauncher(args).app
