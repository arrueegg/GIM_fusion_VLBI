import os
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_year", type=int)
    parser.add_argument("start_doy", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("end_doy", type=int)
    parser.add_argument("--doy_list_file", default="scripts/lists/Unified_doy_list.txt")
    return parser.parse_args()

def metrics_missing(experiments_dir, year, doy):
    doy_str = f"{doy:03d}"
    variants = [
        f"DTEC_Fusion_{year}_{doy_str}_SW100_LW1",
        f"DTEC_Fusion_{year}_{doy_str}_SW1_LW100",
        f"Fusion_{year}_{doy_str}_SW1000_LW1",
        f"Fusion_{year}_{doy_str}_SW1_LW1000",
        f"GNSS_{year}_{doy_str}_SW1_LW1",
    ]
    for variant in variants:
        metrics_file = Path(experiments_dir) / variant / "SA_plots" / "metrics.txt"
        if not metrics_file.exists():
            return True
    return False

def in_range(year, doy, start_year, start_doy, end_year, end_doy):
    if start_year == end_year:
        return year == start_year and start_doy <= doy <= end_doy
    if year == start_year:
        return doy >= start_doy
    if year == end_year:
        return doy <= end_doy
    return start_year < year < end_year

def main():
    args = parse_args()
    experiments_dir = "experiments"
    eval_script = "src/eval_with_SA.py"

    if not Path(args.doy_list_file).exists():
        print(f"DOY list file not found: {args.doy_list_file}")
        return

    with open(args.doy_list_file, 'r') as f:
        for line in f:
            try:
                year, doy = map(int, line.strip().split())
            except ValueError:
                continue  # skip malformed lines

            if not in_range(year, doy, args.start_year, args.start_doy, args.end_year, args.end_doy):
                print(f"{year} {doy} not in specified range")
                continue

            if metrics_missing(experiments_dir, year, doy):
                print(f"Running eval_with_SA.py for {year} {doy}")
                subprocess.run([
                    "python", eval_script,
                    "--year", str(year),
                    "--doy", str(doy)
                ])
            else:
                print(f"Metrics already exist for {year} {doy}")

if __name__ == "__main__":
    main()
