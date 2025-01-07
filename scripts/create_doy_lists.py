import os
from datetime import datetime, timedelta

def extract_doy_and_year(filename):
    try:
        # Assuming the filename format is YYbbDD or similar
        date_str = filename[:7]  # Adjust this slice according to your filename format
        date_obj = datetime.strptime(date_str, '%y%b%d')
        doy = date_obj.timetuple().tm_yday
        year = date_obj.year
        next_day_obj = date_obj + timedelta(days=1)
        next_doy = next_day_obj.timetuple().tm_yday
        next_year = next_day_obj.year
        return year, doy, next_year, next_doy
    except ValueError:
        return None, None, None, None

if __name__ == "__main__":
    base_path = '/scratch2/arrueegg/WP1/publish_complete/VTEC_per_session/'
    os.makedirs('scripts/lists', exist_ok=True)
    vgos_file = open('scripts/lists/VGOS_doy_list.txt', 'w')
    vlbi_file = open('scripts/lists/VLBI_doy_list.txt', 'w')
    unified_file = open('scripts/lists/Unified_doy_list.txt', 'w')
    intersect_file = open('scripts/lists/Intersect_doy_list.txt', 'w')

    vgos_set = set()
    vlbi_set = set()

    for tech in os.listdir(base_path):
        tech_path = os.path.join(base_path, tech)
        if os.path.isdir(tech_path):
            for year in os.listdir(tech_path):
                for session in os.listdir(os.path.join(base_path, tech, year)):
                    year, doy, next_year, next_doy = extract_doy_and_year(session)
                    if doy and year:
                        entry1 = f"{year} {doy}"
                        entry2 = f"{next_year} {next_doy}"
                        if tech == 'VGOS':
                            vgos_set.add(entry1)
                            vgos_set.add(entry2)
                        elif tech == 'VLBI':
                            vlbi_set.add(entry1)
                            vlbi_set.add(entry2)

    vgos_list = sorted(vgos_set, key=lambda x: (int(x.split()[0]), int(x.split()[1])))
    vlbi_list = sorted(vlbi_set, key=lambda x: (int(x.split()[0]), int(x.split()[1])))
    unified_set = sorted(vgos_set.union(vlbi_set), key=lambda x: (int(x.split()[0]), int(x.split()[1])))
    intersect_set = sorted(vgos_set.intersection(vlbi_set), key=lambda x: (int(x.split()[0]), int(x.split()[1])))

    for entry in vgos_list:
        vgos_file.write(f"{entry}\n")

    for entry in vlbi_list:
        vlbi_file.write(f"{entry}\n")

    for entry in unified_set:
        unified_file.write(f"{entry}\n")

    for entry in intersect_set:
        intersect_file.write(f"{entry}\n")

    vgos_file.close()
    vlbi_file.close()
    unified_file.close()
    intersect_file.close()