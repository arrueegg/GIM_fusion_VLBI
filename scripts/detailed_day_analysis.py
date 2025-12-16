"""
Detailed analysis of day availability including:
- VLBI session availability (accounting for multi-day sessions)
- Jason-3 data availability
- Reasons for missing SA evaluations
"""

import os
import re
import pandas as pd
import glob
from io import StringIO
from collections import defaultdict

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
    df = pd.read_csv(StringIO(cleaned), sep=r"\s*\|\s*", engine="python")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def get_vlbi_sessions_for_doy(vlbi_root: str, year: int, doy: int) -> dict:
    """Get detailed VLBI session information for a given DOY and adjacent days."""
    def get_session_info(subdir, year, doy):
        """Get session info for a specific day."""
        date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy-1)
        m = date.strftime("%Y%m%d")
        folder = os.path.join(vlbi_root, subdir, str(year))
        
        if not os.path.exists(folder):
            return []
        
        sessions = []
        for session_folder in os.listdir(folder):
            if session_folder.startswith(m + "-"):
                summary_path = os.path.join(folder, session_folder, "summary.md")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        raw = f.read()
                    
                    vtec = parse_markdown_table(raw, "## VTEC Time Series")
                    if not vtec.empty:
                        vtec['datetime'] = pd.to_datetime(vtec['date'] + ' ' + vtec['epoch'])
                        sessions.append({
                            'session': session_folder,
                            'type': subdir,
                            'start': vtec['datetime'].min(),
                            'end': vtec['datetime'].max(),
                            'stations': vtec['station'].unique().tolist(),
                            'n_obs': len(vtec)
                        })
        return sessions
    
    # Check current day and adjacent days (sessions can span multiple days)
    all_sessions = []
    for check_doy in [doy-1, doy, doy+1]:
        if check_doy < 1 or check_doy > 366:
            continue
        for subdir in ["SX", "VGOS"]:
            all_sessions.extend(get_session_info(subdir, year, check_doy))
    
    # Filter to sessions that have data on the target day
    target_date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy-1)
    target_start = target_date
    target_end = target_date + pd.Timedelta(days=1)
    
    relevant_sessions = []
    for session in all_sessions:
        # Check if session overlaps with target day
        if session['start'] < target_end and session['end'] > target_start:
            relevant_sessions.append(session)
    
    return relevant_sessions


def check_jason3_data(year: int, doy: int) -> dict:
    """Check if Jason-3 data is available for a given day."""
    jason3_base = "/home/space/data/SA/jason3"
    
    # Jason-3 data is organized by cycle
    # Each cycle has multiple passes, files are named with date
    date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy-1)
    date_str = date.strftime("%Y%m%d")
    
    # Search for files with this date
    files = glob.glob(f"{jason3_base}/cycle*/*{date_str}*.nc")
    
    return {
        'available': len(files) > 0,
        'n_files': len(files),
        'files': files
    }


def analyze_missing_sa_evaluation(experiments_folder, year, doy):
    """Analyze why SA evaluation is missing for a specific day."""
    doy_str = f"{year}_{doy:03d}"
    
    # Check which methods have SA_plots
    methods_with_sa = []
    methods_without_sa = []
    
    for method_key in METHOD_MAP.keys():
        # Find the experiment folder for this method and day
        pattern = f"*{doy_str}_*"
        matching_folders = [f for f in os.listdir(experiments_folder) 
                          if f"{year}_{doy:03d}_" in f and extract_key_from_folder(f) == method_key]
        
        if matching_folders:
            exp_folder = matching_folders[0]
            sa_folder = os.path.join(experiments_folder, exp_folder, 'SA_plots')
            results_csv = os.path.join(sa_folder, 'results.csv')
            
            if os.path.isdir(sa_folder) and os.path.isfile(results_csv):
                methods_with_sa.append(METHOD_MAP[method_key])
            else:
                methods_without_sa.append(METHOD_MAP[method_key])
    
    # Check Jason-3 data
    jason3_info = check_jason3_data(year, doy)
    
    # Check VLBI sessions
    vlbi_sessions = get_vlbi_sessions_for_doy('/scratch2/arrueegg/WP1/VLBIono/Results/', year, doy)
    
    # Determine reason
    if not jason3_info['available']:
        reason = "No Jason-3 data available"
    elif len(methods_without_sa) == len(METHOD_MAP):
        reason = "SA evaluation not run for any method"
    else:
        reason = f"SA evaluation incomplete ({len(methods_with_sa)}/{len(METHOD_MAP)} methods)"
    
    return {
        'doy': doy,
        'doy_str': doy_str,
        'methods_with_sa': methods_with_sa,
        'methods_without_sa': methods_without_sa,
        'jason3_available': jason3_info['available'],
        'jason3_n_files': jason3_info['n_files'],
        'vlbi_sessions': vlbi_sessions,
        'n_vlbi_sessions': len(vlbi_sessions),
        'reason': reason
    }


def detailed_analysis(experiments_folder='/scratch2/arrueegg/WP2/GIM_fusion_VLBI/experiments/'):
    """Perform detailed analysis of all days."""
    
    print("\n" + "="*80)
    print("DETAILED DAY AVAILABILITY ANALYSIS")
    print("="*80)
    
    # Get all experiment folders
    exp_list = sorted(os.listdir(experiments_folder))
    exp_list = [m for m in exp_list if extract_key_from_folder(m) in METHOD_MAP]
    
    # Collect all unique days
    all_doys = set()
    for method in exp_list:
        match = re.search(r'_(\d{4})_(\d{3})_', method)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            all_doys.add((year, doy))
    
    print(f"\nAnalyzing {len(all_doys)} days...")
    
    # Analyze each day
    day_analysis = []
    missing_sa_details = []
    
    for year, doy in sorted(all_doys):
        doy_str = f"{year}_{doy:03d}"
        
        # Check which methods have SA_plots
        methods_for_doy = [m for m in exp_list if f"_{doy_str}_" in m]
        
        has_all_methods = len([m for m in methods_for_doy 
                               if extract_key_from_folder(m) in METHOD_MAP]) == len(METHOD_MAP)
        
        # Check SA plots
        valid_methods = []
        for method in methods_for_doy:
            sa_folder = os.path.join(experiments_folder, method, 'SA_plots')
            results_csv = os.path.join(sa_folder, 'results.csv')
            if os.path.isdir(sa_folder) and os.path.isfile(results_csv):
                valid_methods.append(method)
        
        has_all_sa = len(valid_methods) == len(METHOD_MAP)
        
        # Check VLBI sessions
        vlbi_sessions = get_vlbi_sessions_for_doy('/scratch2/arrueegg/WP1/VLBIono/Results/', year, doy)
        
        # Check Jason-3
        jason3_info = check_jason3_data(year, doy)
        
        day_analysis.append({
            'year': year,
            'doy': doy,
            'doy_str': doy_str,
            'has_all_methods': has_all_methods,
            'has_all_sa': has_all_sa,
            'n_sa_methods': len(valid_methods),
            'n_vlbi_sessions': len(vlbi_sessions),
            'has_jason3': jason3_info['available'],
            'n_jason3_files': jason3_info['n_files']
        })
        
        # If SA is missing, analyze why
        if not has_all_sa:
            details = analyze_missing_sa_evaluation(experiments_folder, year, doy)
            missing_sa_details.append(details)
    
    # Convert to DataFrame
    df = pd.DataFrame(day_analysis)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total days analyzed: {len(df)}")
    print(f"Days with all methods trained: {df['has_all_methods'].sum()}")
    print(f"Days with complete SA evaluation: {df['has_all_sa'].sum()}")
    print(f"Days with VLBI sessions: {(df['n_vlbi_sessions'] > 0).sum()}")
    print(f"Days with Jason-3 data: {df['has_jason3'].sum()}")
    
    # VLBI session distribution
    print("\n" + "="*80)
    print("VLBI SESSION DISTRIBUTION")
    print("="*80)
    vlbi_counts = df['n_vlbi_sessions'].value_counts().sort_index()
    for n_sessions, count in vlbi_counts.items():
        print(f"  {n_sessions} session(s): {count} days")
    
    # Days without VLBI
    no_vlbi = df[df['n_vlbi_sessions'] == 0]
    if len(no_vlbi) > 0:
        print(f"\nDays WITHOUT VLBI sessions: {len(no_vlbi)}")
        print("  (These should not have been trained)")
    
    # Missing SA evaluation analysis
    print("\n" + "="*80)
    print("MISSING SA EVALUATION ANALYSIS ({} days)".format(len(missing_sa_details)))
    print("="*80)
    
    # Group by reason
    reasons = defaultdict(list)
    for detail in missing_sa_details:
        reasons[detail['reason']].append(detail)
    
    for reason, days in sorted(reasons.items()):
        print(f"\n{reason}: {len(days)} days")
        for detail in sorted(days, key=lambda x: x['doy']):
            date = pd.Timestamp(detail['doy_str'].split('_')[0] + '-01-01') + pd.Timedelta(days=detail['doy']-1)
            print(f"  {detail['doy_str']} ({date.strftime('%b %d, %a')})")
            print(f"    VLBI sessions: {detail['n_vlbi_sessions']}")
            print(f"    Jason-3 files: {detail['jason3_n_files']}")
            if detail['methods_with_sa']:
                print(f"    Methods WITH SA: {', '.join(detail['methods_with_sa'])}")
            if detail['methods_without_sa']:
                print(f"    Methods WITHOUT SA: {', '.join(detail['methods_without_sa'])}")
    
    # Save detailed report
    os.makedirs('evaluation', exist_ok=True)
    
    report_lines = [
        "="*80,
        "DETAILED DAY AVAILABILITY ANALYSIS",
        "="*80,
        "",
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "="*80,
        "SUMMARY",
        "="*80,
        f"Total days analyzed: {len(df)}",
        f"Days with all methods trained: {df['has_all_methods'].sum()}",
        f"Days with complete SA evaluation: {df['has_all_sa'].sum()}",
        f"Days with VLBI sessions: {(df['n_vlbi_sessions'] > 0).sum()}",
        f"Days with Jason-3 data: {df['has_jason3'].sum()}",
        "",
        "="*80,
        "VLBI SESSION DISTRIBUTION",
        "="*80,
    ]
    
    for n_sessions, count in vlbi_counts.items():
        report_lines.append(f"  {n_sessions} session(s): {count} days")
    
    report_lines.extend([
        "",
        "="*80,
        "MISSING SA EVALUATION DETAILS",
        "="*80,
        ""
    ])
    
    for reason, days in sorted(reasons.items()):
        report_lines.append(f"\n{reason}: {len(days)} days")
        report_lines.append("-" * 80)
        for detail in sorted(days, key=lambda x: x['doy']):
            date = pd.Timestamp(detail['doy_str'].split('_')[0] + '-01-01') + pd.Timedelta(days=detail['doy']-1)
            report_lines.append(f"\n{detail['doy_str']} ({date.strftime('%Y-%m-%d, %A')})")
            
            # VLBI session details
            if detail['vlbi_sessions']:
                report_lines.append(f"  VLBI Sessions: {len(detail['vlbi_sessions'])}")
                for session in detail['vlbi_sessions']:
                    report_lines.append(f"    • {session['session']} ({session['type']})")
                    report_lines.append(f"      Time: {session['start']} to {session['end']}")
                    report_lines.append(f"      Stations: {', '.join(session['stations'][:5])}" + 
                                      (f" (+{len(session['stations'])-5} more)" if len(session['stations']) > 5 else ""))
                    report_lines.append(f"      Observations: {session['n_obs']}")
            else:
                report_lines.append(f"  VLBI Sessions: None")
            
            report_lines.append(f"  Jason-3 data: {'Yes' if detail['jason3_available'] else 'No'} ({detail['jason3_n_files']} files)")
            
            if detail['methods_with_sa']:
                report_lines.append(f"  Methods WITH SA evaluation: {', '.join(detail['methods_with_sa'])}")
            if detail['methods_without_sa']:
                report_lines.append(f"  Methods WITHOUT SA evaluation: {', '.join(detail['methods_without_sa'])}")
    
    report_path = 'evaluation/detailed_day_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n{'='*80}")
    print(f"✓ Detailed report saved to: {report_path}")
    print(f"{'='*80}\n")
    
    # Save summary CSV
    df.to_csv('evaluation/day_analysis_summary.csv', index=False)
    print(f"✓ Summary CSV saved to: evaluation/day_analysis_summary.csv\n")
    
    return df, missing_sa_details


if __name__ == '__main__':
    df, missing_details = detailed_analysis()
