"""
Standalone CLI Preprocessing Script for US Accidents Dataset.
Executes chunked sampling, null/duplicate cleaning, and Parquet caching
so the Streamlit dashboard loads instantly without waiting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    DEFAULT_PARQUET,
    DEFAULT_RAW_CSV,
    process_and_cache_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and cache US Accidents Kaggle dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_RAW_CSV),
        help=f"Path to raw CSV file (default: {DEFAULT_RAW_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_PARQUET),
        help=f"Path to output Parquet file (default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to sample uniformly across chunks (default: 0.05 = ~385k rows)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Chunk size for reading raw CSV (default: 250,000)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    print(f" Starting preprocessing pipeline...")
    print(f"   Input : {input_path} ({input_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"   Output: {output_path}")
    print(f"   Sample fraction: {args.sample_fraction:.2%} (~{int(7_700_000 * args.sample_fraction):,} rows)")
    print(f"   Chunk size: {args.chunk_size:,}")

    start_time = time.time()

    def console_progress(progress: float, message: str):
        pct = int(progress * 100)
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {pct:3d}% - {message}")
        sys.stdout.flush()

    df = process_and_cache_dataset(
        raw_csv_path=input_path,
        parquet_path=output_path,
        sample_fraction=args.sample_fraction,
        chunk_size=args.chunk_size,
        progress_callback=console_progress,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Preprocessing completed successfully in {elapsed:.1f}s!")
    print(f"   Final Cleaned Records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Output file size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"   Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   States Represented: {df['State'].nunique()} states/jurisdictions")
    print(f"   Memory in RAM: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
