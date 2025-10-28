import json, glob

pairs = []
for run_dir in sorted(glob.glob("runs/*")):
    for s in sorted(glob.glob(f"{run_dir}/state_round_*.json")):
        r = s.split("_")[-1].split(".")[0]  # round id
        a = f"{run_dir}/action_round_{r}.json"
        try:
            with open(s) as fs, open(a) as fa:
                state  = json.load(fs)
                action = json.load(fa)
            # keep only the fields we train on; drop bytes if you want
            pairs.append({"state": state, "action": action})
        except FileNotFoundError:
            pass

out = "sft_pairs.jsonl"
with open(out, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"Wrote {out} with {len(pairs)} pairs")