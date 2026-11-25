"""Inventory of every run in the store: configuration, state, every report."""
import yaml, glob, os, json, csv
STORE = "/data/bfys/gscriven/rkpinn_store/runs"
rows = []
for d in sorted(glob.glob(STORE + "/*/configuration.yaml")):
    k = d.split('/')[-2]; y = yaml.safe_load(open(d)); n = y.get('network', {})
    st = json.load(open(f"{STORE}/{k}/state.json"))
    lastval = ""
    try:
        rr = list(csv.DictReader(open(f"{STORE}/{k}/rounds.csv")))
        if rr: lastval = rr[-1].get('validation_endpoint_error_larger_of_x_and_y_median_micrometres', '')
    except Exception: pass
    base = dict(key=k, tracks=y['tracks']['key'], map=y['equation_of_motion']['field_map'],
                kind=n.get('kind'), output_form=n.get('output_form'), scale=n.get('scale_of_outputs'),
                sweeps_inside=n.get('sweeps_inside'), opt=y['training']['optimiser']['name'],
                loss=y['loss']['name'], target=y['target']['name'], steps=y['steps']['number_of_steps'],
                stages=y['steps']['number_of_stages'], seed=y.get('seed'), state=st.get('state'),
                rounds=st.get('rounds_done'), last_val_larger_xy=lastval)
    reps = sorted(glob.glob(f"{STORE}/{k}/reports/*/summary.json"))
    if not reps: rows.append(dict(base, report='', split='', sweeps='', median='', p95='', us='')); continue
    for r in reps:
        s = json.load(open(r)); rd = os.path.dirname(r); p95 = ''; us = ''
        try:
            for m in csv.DictReader(open(rd + "/endpoint_error_magnitudes.csv")):
                if m['band'] == 'all momenta' and m['magnitude'] == 'radial': p95 = m['percentile_95_micrometres']
        except Exception: pass
        try: us = next(csv.DictReader(open(rd + "/cost.csv")))['microseconds_to_carry_one_track']
        except Exception: pass
        rows.append(dict(base, report=os.path.basename(rd), split=s.get('split'), sweeps=s.get('sweeps'),
                         median=s.get('endpoint_radial_median_micrometres'), p95=p95, us=us))
out = os.path.join(os.path.dirname(__file__), "..", "_work", "inventory.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
w = csv.DictWriter(open(out, "w", newline=""), list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows))
