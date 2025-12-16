"""
Analyze day availability for Jason-3 evaluation.
This script shows which days were included/excluded in the final evaluation
and the reasons for exclusion at each filtering stage.
"""

import os
import re
import pandas as pd

# Define the methods we're analyzing
SELECTED_METHODS = [
    'GNSS_1_1',
    'DTEC_1_100',
    'Fusion_1_1000',
]

METHOD_MAP = {
    'GNSS_1_1':      'GNSS only',
    'Fusion_1000_1': 'VLBI VTEC',
    'Fusion_1_1000': 'VLBI VTEC',
    'DTEC_100_1':    'VLBI DTEC',
    'DTEC_1_100':    'VLBI DTEC',
}

METHOD_MAP = {k: METHOD_MAP[k] for k in SELECTED_METHODS if k in METHOD_MAP}


def extract_key_from_folder(folder_name):
    """Extract method key from folder name."""
    for key in METHOD_MAP.keys():
        parts = key.split('_')
        if len(parts) == 3:
            approach, SW, LW = parts
            if folder_name.startswith(approach) and f'_SW{SW}_LW{LW}' in folder_name:
                return key
    return None


def parse_markdown_table(text: str, section_header: str) -> pd.DataFrame:
    """Parse markdown table from VLBI summary."""
    pattern = re.escape(section_header) + r"(.*?)(\n##|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return pd.DataFrame()
    table_text = match.group(1).strip()
    lines = [l for l in table_text.splitlines()
             if not re.match(r"^\s*\|[-:\s|]+\|$", l)]
    cleaned = "\n".join([l.strip().strip("|") for l in lines if l.strip()])
    from io import StringIO
    df = pd.read_csv(StringIO(cleaned), sep=r"\s*\|\s*", engine="python")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def load_vlbi_meta(vlbi_root: str, year: int, doy: int) -> pd.DataFrame:
    """Check if VLBI sessions exist for a given day."""
    def doy_paths(subdir, year, doy):
        m = (pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy-1)).strftime("%Y%m%d")
        folder = os.path.join(vlbi_root, subdir, str(year))
        if not os.path.exists(folder):
            return []
        return [os.path.join(folder, p, "summary.md")
                for p in os.listdir(folder) if p.startswith(m+"-")]

    md_paths = set()
    for subdir in ["SX", "VGOS"]:
        md_paths.update(doy_paths(subdir, year, doy))
        if doy > 1:
            md_paths.update(doy_paths(subdir, year, doy-1))

    all_dfs = []
    for fp in md_paths:
        if not os.path.exists(fp):
            continue
        raw = open(fp).read()
        vtec = parse_markdown_table(raw, "## VTEC Time Series")
        if not vtec.empty:
            vtec['datetime'] = pd.to_datetime(vtec['date'] + ' ' + vtec['epoch'])
            all_dfs.append(vtec)

    if not all_dfs:
        return pd.DataFrame(columns=['station', 'first_datetime', 'last_datetime'])

    combined = pd.concat(all_dfs, ignore_index=True)
    return (combined
            .groupby('station')['datetime']
            .agg(['min', 'max'])
            .reset_index()
            .rename(columns={'min': 'first_datetime', 'max': 'last_datetime'}))


def analyze_day_availability(experiments_folder='/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/'):
    """Analyze which days are available and which were filtered out."""
    
    print("\n" + "="*80)
    print("DAY AVAILABILITY ANALYSIS FOR JASON-3 EVALUATION")
    print("="*80)
    
    # Get all experiment folders
    exp_list = sorted(os.listdir(experiments_folder))
    exp_list = [m for m in exp_list if extract_key_from_folder(m) in METHOD_MAP]
    
    # Collect all unique days
    all_doys = set()
    for method in exp_list:
        match = re.search(r'_(\d{4}_\d{3})_', method)
        if match:
            all_doys.add(match.group(1))
    
    print(f"\nTotal unique days found in experiments folder: {len(all_doys)}")
    
    # Track filtering stages
    days_with_all_methods = []
    days_missing_methods = {}
    days_with_sa_plots = []
    days_missing_sa_plots = {}
    days_with_vlbi = []
    days_without_vlbi = []
    
    for doy in sorted(all_doys):
        methods_for_doy = [method for method in exp_list if f"_{doy}_" in method]
        
        # Stage 1: Check if all methods exist
        if len(methods_for_doy) < len(METHOD_MAP):
            days_missing_methods[doy] = len(methods_for_doy)
            continue
        
        days_with_all_methods.append(doy)
        
        # Stage 2: Check if SA_plots/results.csv exists for all methods
        valid_methods = []
        for method in methods_for_doy:
            sa_folder = os.path.join(experiments_folder, method, 'SA_plots')
            results_csv = os.path.join(sa_folder, 'results.csv')
            if os.path.isdir(sa_folder) and os.path.isfile(results_csv):
                valid_methods.append(method)
        
        if len(valid_methods) < len(METHOD_MAP):
            days_missing_sa_plots[doy] = len(valid_methods)
            continue
        
        days_with_sa_plots.append(doy)
        
        # Stage 3: Check if VLBI data is available
        year_doy = doy.split('_')
        year = int(year_doy[0])
        doy_num = int(year_doy[1])
        vlbi_meta = load_vlbi_meta('/scratch2/arrueegg/WP1/VLBIono/Results/', year, doy_num)
        
        if vlbi_meta.empty:
            days_without_vlbi.append(doy)
        else:
            days_with_vlbi.append(doy)
    
    # Print summary
    print(f"\n{'─'*80}")
    print(f"FILTERING STAGES:")
    print(f"{'─'*80}")
    print(f"1. Days with all {len(METHOD_MAP)} methods trained: {len(days_with_all_methods)}/{len(all_doys)}")
    print(f"   → Excluded (incomplete methods): {len(days_missing_methods)}")
    if days_missing_methods:
        examples = list(days_missing_methods.items())[:5]
        for doy, count in examples:
            print(f"      • {doy}: {count}/{len(METHOD_MAP)} methods")
        if len(days_missing_methods) > 5:
            print(f"      ... and {len(days_missing_methods)-5} more")
    
    print(f"\n2. Days with SA_plots/results.csv for all methods: {len(days_with_sa_plots)}/{len(days_with_all_methods)}")
    print(f"   → Excluded (missing SA results): {len(days_missing_sa_plots)}")
    if days_missing_sa_plots:
        examples = list(days_missing_sa_plots.items())[:5]
        for doy, count in examples:
            print(f"      • {doy}: {count}/{len(METHOD_MAP)} have SA plots")
        if len(days_missing_sa_plots) > 5:
            print(f"      ... and {len(days_missing_sa_plots)-5} more")
    
    print(f"\n3. Days with VLBI sessions available: {len(days_with_vlbi)}/{len(days_with_sa_plots)}")
    print(f"   → Excluded (no VLBI data): {len(days_without_vlbi)}")
    if days_without_vlbi:
        for doy in days_without_vlbi:
            print(f"      • {doy}")
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {len(days_with_vlbi)} days included in evaluation")
    excluded_pct = 100*(len(all_doys) - len(days_with_vlbi))/len(all_doys) if len(all_doys) > 0 else 0
    print(f"EXCLUDED: {len(all_doys) - len(days_with_vlbi)} days total ({excluded_pct:.1f}%)")
    print(f"{'='*80}\n")
    
    # Save detailed report
    os.makedirs('evaluation', exist_ok=True)
    report_lines = [
        "="*80,
        "DAY AVAILABILITY ANALYSIS FOR JASON-3 EVALUATION",
        "="*80,
        "",
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiments Folder: {experiments_folder}",
        f"Methods Analyzed: {', '.join(METHOD_MAP.values())}",
        "",
        "="*80,
        "SUMMARY",
        "="*80,
        f"Total unique days found: {len(all_doys)}",
        f"Days used in final evaluation: {len(days_with_vlbi)}",
        f"Days excluded: {len(all_doys) - len(days_with_vlbi)} ({excluded_pct:.1f}%)",
        "",
        "="*80,
        "FILTERING STAGES BREAKDOWN",
        "="*80,
        "",
        f"Stage 1: All {len(METHOD_MAP)} methods trained",
        f"  Passed: {len(days_with_all_methods)} days",
        f"  Excluded: {len(days_missing_methods)} days",
        ""
    ]
    
    if days_missing_methods:
        report_lines.append("  Days excluded (missing training runs):")
        for doy, count in sorted(days_missing_methods.items()):
            report_lines.append(f"    {doy}: only {count}/{len(METHOD_MAP)} methods available")
        report_lines.append("")
    
    report_lines.extend([
        f"Stage 2: SA_plots/results.csv available for all methods",
        f"  Passed: {len(days_with_sa_plots)} days",
        f"  Excluded: {len(days_missing_sa_plots)} days",
        ""
    ])
    
    if days_missing_sa_plots:
        report_lines.append("  Days excluded (missing SA evaluation):")
        for doy, count in sorted(days_missing_sa_plots.items()):
            report_lines.append(f"    {doy}: only {count}/{len(METHOD_MAP)} methods have SA plots")
        report_lines.append("")
    
    report_lines.extend([
        f"Stage 3: VLBI sessions available",
        f"  Passed: {len(days_with_vlbi)} days",
        f"  Excluded: {len(days_without_vlbi)} days",
        ""
    ])
    
    if days_without_vlbi:
        report_lines.append("  Days excluded (no VLBI data):")
        for doy in sorted(days_without_vlbi):
            report_lines.append(f"    {doy}")
        report_lines.append("")
    
    report_lines.extend([
        "="*80,
        "DAYS INCLUDED IN FINAL EVALUATION",
        "="*80,
        ""
    ])
    
    for doy in sorted(days_with_vlbi):
        year, day = doy.split('_')
        date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(day)-1)
        report_lines.append(f"  {doy} ({date.strftime('%Y-%m-%d, %a')})")
    
    report_path = 'evaluation/day_availability_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Detailed report saved to: {report_path}\n")
    
    return {
        'all_doys': len(all_doys),
        'days_with_vlbi': len(days_with_vlbi),
        'excluded': len(all_doys) - len(days_with_vlbi),
        'days_missing_methods': len(days_missing_methods),
        'days_missing_sa_plots': len(days_missing_sa_plots),
        'days_without_vlbi': len(days_without_vlbi)
    }


if __name__ == '__main__':
    stats = analyze_day_availability()
