"""
Restore fuel_data.csv from the compressed binary format.

Usage:
    1. Extract fuel_data_compressed.7z to get fuel_minimal.bin and fuel_lookups_min.json
    2. Run: python restore_fuel_data.py

Or just run: 7z x fuel_data_compressed.7z && python restore_fuel_data.py
"""

import json
import struct
import numpy as np
import pandas as pd
from pathlib import Path

def restore_csv(bin_path='fuel_minimal.bin',
                json_path='fuel_lookups_min.json',
                output_path='fuel_data_restored.csv'):

    # Load lookup tables
    with open(json_path) as f:
        lookups = json.load(f)

    sites = lookups['sites']
    grades = lookups['grades']
    days = lookups['days']
    b_unit = lookups['b_unit']
    created_ats = lookups['created_ats']
    num_rows = lookups['num_rows']

    # Read binary data
    with open(bin_path, 'rb') as f:
        n_rows = struct.unpack('<I', f.read(4))[0]

        site_deltas = np.frombuffer(f.read(n_rows * 1), dtype=np.int8)
        grade_idxs = np.frombuffer(f.read(n_rows * 1), dtype=np.int8)
        day_deltas = np.frombuffer(f.read(n_rows * 2), dtype=np.int16)
        volumes = np.frombuffer(f.read(n_rows * 2), dtype=np.float16)
        is_estimated = np.frombuffer(f.read(n_rows * 1), dtype=np.uint8)
        total_sales = np.frombuffer(f.read(n_rows * 2), dtype=np.float16)
        targets = np.frombuffer(f.read(n_rows * 2), dtype=np.float16)
        created_idxs = np.frombuffer(f.read(n_rows * 1), dtype=np.uint8)

    # Reconstruct indices from deltas
    site_idxs = np.cumsum(site_deltas.astype(np.int32))
    day_idxs = np.cumsum(day_deltas.astype(np.int32))

    # Build dataframe
    rows = []
    for i in range(n_rows):
        site_idx = site_idxs[i]
        site_data = sites[site_idx]

        row = {
            'site_id': site_data['site_id'],
            'grade': grades[grade_idxs[i]],
            'day': days[day_idxs[i]],
            'brand': site_data['brand'],
            'site': site_data['site'],
            'address': site_data['address'],
            'city': site_data['city'],
            'state': site_data['state'],
            'owner': site_data['owner'],
            'b_unit': b_unit,
            'stock': None,
            'delivered': None,
            'volume': float(volumes[i]) if not np.isnan(volumes[i]) else None,
            'is_estimated': int(is_estimated[i]),
            'total_sales': float(total_sales[i]) if not np.isnan(total_sales[i]) else None,
            'target': float(targets[i]) if not np.isnan(targets[i]) else None,
            'created_at': created_ats[created_idxs[i]] if created_idxs[i] < 255 else None
        }
        rows.append(row)

        if i % 100000 == 0:
            print(f'Processed {i:,} / {n_rows:,} rows...')

    df = pd.DataFrame(rows)

    # Reorder columns to match original
    cols = ['site_id', 'grade', 'day', 'brand', 'site', 'address', 'city', 'state',
            'owner', 'b_unit', 'stock', 'delivered', 'volume', 'is_estimated',
            'total_sales', 'target', 'created_at']
    df = df[cols]

    df.to_csv(output_path, index=False)
    print(f'Restored {n_rows:,} rows to {output_path}')

if __name__ == '__main__':
    restore_csv()
