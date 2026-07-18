#!/usr/bin/env python3
"""Compare any two benchmark CSV files side-by-side.

Usage:
    python3 scripts/compare_bench.py <csv_a> <csv_b>
"""

import csv
import os
import sys

def load(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def stats(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col, '') != '']
    if not vals:
        return None
    return {'avg': sum(vals)/len(vals), 'min': min(vals), 'max': max(vals), 'n': len(vals)}

def short_name(path):
    base = os.path.basename(path)
    # bench_<scene>_<steps>_gpu<N>.csv -> gpu<N>
    if '_gpu' in base:
        return base.split('_gpu')[-1].replace('.csv', '')
    return base.replace('.csv', '')

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <csv_a> <csv_b>")
        sys.exit(1)

    a_path, b_path = sys.argv[1], sys.argv[2]
    a_rows = load(a_path)
    b_rows = load(b_path)
    a_name = short_name(a_path)
    b_name = short_name(b_path)

    # Detect multi-GPU (has mig_crossed column with 0/1 variation)
    has_multi = any(r.get('mig_crossed', '0') == '1' for r in b_rows) or \
                any(r.get('mig_crossed', '0') == '1' for r in a_rows)

    # Split B into idle/crossing if multi-GPU
    if has_multi:
        a_idle    = [r for r in a_rows if r.get('mig_crossed', '0') == '0']
        a_crossed = [r for r in a_rows if r.get('mig_crossed', '0') == '1']
        b_idle    = [r for r in b_rows if r.get('mig_crossed', '0') == '0']
        b_crossed = [r for r in b_rows if r.get('mig_crossed', '0') == '1']
    else:
        a_idle, a_crossed = a_rows, []
        b_idle, b_crossed = b_rows, []

    metrics = [
        ('wall_ms',        'Wall clock',     'ms'),
        ('halo_ms',        'Halo',           'ms'),
        ('halo_ghosts',    'Ghosts',         ''),
        ('force_ms',       'Force',          'ms'),
        ('force_contacts', 'Contacts',       ''),
        ('integrate_ms',   'Integrate',      'ms'),
        ('sync_ms',        'Sync',           'ms'),
        ('mig_dl_ms',      'Mig download',   'ms'),
        ('mig_merge_ms',   'Mig merge',      'ms'),
        ('mig_ul_ms',      'Mig upload',     'ms'),
        ('vtk_ms',         'VTK',            'ms'),
    ]

    # Column headers
    cols = [f"{'Metric':<18}", f"{'A':>10}", f"{'B idle':>10}"]
    if b_crossed:
        cols.append(f"{'B cross':>10}")
        cols.append(f"{'ratio':>8}")
    header = "".join(cols)

    print(f"\nA = {a_name} ({len(a_rows)} steps)    B = {b_name} ({len(b_rows)} steps)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for col, label, _unit in metrics:
        if col not in a_rows[0]:
            continue

        a_s  = stats(a_idle, col)
        bi_s = stats(b_idle, col) if b_idle else None
        bc_s = stats(b_crossed, col) if b_crossed else None

        def fmt(s):
            return f"{s['avg']:.3f}" if s else "     -"

        row = f"{label:<18} {fmt(a_s):>10} {fmt(bi_s):>10}"
        if b_crossed:
            row += f" {fmt(bc_s):>10}"
            if bi_s and bc_s and bi_s['avg'] > 0.001 and col != 'halo_ghosts':
                row += f" {bc_s['avg']/bi_s['avg']:>7.1f}x"
            else:
                row += "        "
        print(row)

    print("-" * len(header))

    # Step counts
    sc = f"{'Steps':<18} {len(a_idle):>10} {len(b_idle):>10}"
    if b_crossed:
        sc += f" {len(b_crossed):>10}"
    print(sc)

    # Crossing rate
    if b_crossed:
        rate_a = len(a_crossed)/max(len(a_rows),1)*100
        rate_b = len(b_crossed)/max(len(b_rows),1)*100
        print(f"{'Crossing rate':<18} {rate_a:>9.1f}% {rate_b:>9.1f}%")

    # Load variance
    for label, rows in [(a_name, a_rows), (b_name, b_rows)]:
        if 'load_var' in rows[0]:
            vars_ = [float(r['load_var']) for r in rows]
            if max(vars_) > 0:
                print(f"Load var ({label}): avg={sum(vars_)/len(vars_):.1f}  max={max(vars_):.1f}")

    # Overhead breakdown (if wall clock differs meaningfully)
    if a_idle and b_idle:
        a_wall = stats(a_idle, 'wall_ms')['avg']
        b_wall = stats(b_idle, 'wall_ms')['avg']
        overhead = abs(b_wall - a_wall)
        if overhead > 0.005:
            slower = b_name if b_wall > a_wall else a_name
            print(f"\nOverhead ({slower} idle vs other):")
            print(f"  {a_name}: {a_wall:.3f} ms   {b_name}: {b_wall:.3f} ms   delta: {overhead:.3f} ms")
            h = stats(b_idle, 'halo_ms')
            if h and h['avg'] > 0:
                print(f"  of which halo: {h['avg']:.3f} ms ({h['avg']/overhead*100:.0f}%)" if overhead > 0 else "")
    print()

if __name__ == '__main__':
    main()
