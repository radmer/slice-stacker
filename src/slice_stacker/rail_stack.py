"""Focus stacking for images from a focus rail (focus racking).

This module provides focus stacking functionality specifically
designed for images captured using a macro rail, where the camera
moves through space rather than changing focal plane.
"""

import argparse


def main():
    """CLI entry point for rail-stack command."""
    parser = argparse.ArgumentParser(
        description="Focus stack images from a macro rail sequence"
    )
    parser.add_argument("images", nargs="+", help="Input images to stack")
    parser.add_argument("-o", "--output", default="stacked.jpg", help="Output filename")
    
    args = parser.parse_args()
    print(f"Rail stacking {len(args.images)} images -> {args.output}")
    print("TODO: Implement rail stacking logic")


if __name__ == "__main__":
    main()
