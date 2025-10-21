Ada-FCL Roadmap — LLM-Driven Federated Continual Learning

Objective: Build and validate an LLM meta-controller (SFT → PPO) that adaptively sets FL/CL policies per round and per client, dominating fixed and heuristic baselines not only on accuracy, but also stability and efficiency (communication + time).

We proceed in four phases with explicit acceptance gates. We already have:
	•	Baseline (fixed hyper-params) on CIFAR-100.
	•	Controller-v4 (adaptive heuristic), competitive but slightly below baseline.

⸻

Phase 0 — Housekeeping & Locked Baselines  ✅ (we’re here)

Canonical CIFAR-100 config (frozen)
	•	Clients: 4 (equal split), continual batches per client: 7 (initial ≈ 46.6% of client data, then 6 equal increments).
	•	Model: ResNet-18 (keep initialization consistent across runs).
	•	Optimizer: Adam, base LR = 1e-4, weight_decay = 0.0.
	•	Local epochs per round: 20 (with early_patience = 5).
	•	Batch size: 256; num_workers: 4.
	•	Replay buffer capacity: 2000; baseline replay ratio: 0.50.
	•	Aggregation: FedAvg.
	•	Device: CUDA.
	•	Seeds: 42 (and later 43, 44 for final reporting).

Required logging (per round + per client)
	•	Round: round_id, acc_global, loss_global, ema_loss, divergence, forget_mean, forget_max, aulc_running, comm_bytes_round, comm_bytes_cum, wall_time_s, num_clients_active.
	•	Client: client_id, vloss, vacc, lr_effective, replay_ratio_effective, ewc_lambda_effective, selected(bool).

Acceptance gate
	•	Baseline and v4 both finish 7 rounds; CSVs contain all fields above. v4 is within a small band of baseline (already observed).

⸻

Phase 1 — Freeze the LLM Interface + Mock Agent Loop (no LLM yet)

1.1 State JSON (input to controller)

Minimal but sufficient signals; these fields are fixed.

<pre>
```json
{
  "round_id": 3,
  "global": {
    "acc": 0.714,
    "loss": 1.395,
    "ema_loss": 1.462,
    "forget_mean": 0.024,
    "forget_max": 0.081,
    "divergence": 0.102,
    "bytes_last_round": 201326592
  },
  "clients": [
    {
      "id": 0,
      "vloss": 1.28,
      "vacc": 0.665,
      "new_batch_size": 1001,
      "replay_capacity": 2000,
      "last_lr": 0.00010,
      "last_replay_ratio": 0.50,
      "last_ewc_lambda": 0.0
    }
    // ... for each participating client
  ]
}
```
</pre>

Notes
	•	divergence: normalized std of client-to-global parameter distance (already computed in v4).
	•	bytes_last_round ≈ (#active clients × model_size_bytes × 2 [up & down]).

1.2 Action JSON (output from controller)

Hard bounds enforced & clamped (safety guarantee).

<pre>
```json
{
  "client_selection_k": 4,              // [2, 4]
  "aggregation": { "method": "FedAvg" },// {"FedAvg"}; {"FedProx"} with mu later in ablations
  "client_params": [
    {
      "id": 0,
      "replay_ratio": 0.60,             // [0.20, 0.70]
      "lr_scale": 1.10,                  // [0.80, 1.20] applied to base LR
      "ewc_lambda": 0.0                  // [0.0, 1000.0] (0 ok to start)
    }
    // ... for selected clients
  ]
}
```
</pre>

All parameter ranges and defaults follow the Global Invariants table at the end of this document.

1.3 Reward (compute & log now; used for RL later)
	•	AULC: running mean of acc_global up to round t. (Primary gain term.)
	•	Forgetting: average & max of class-wise recall drop vs best-so-far.
	•	Comm cost: bytes_last_round and cumulative.
	•	Divergence: the same metric from State.
	•	Stability: EMA loss trend and deadband violations of Δacc.

Initial weights (CIFAR sandbox)
	•	α (AULC gain) = 0.50
	•	β (Forgetting penalty) = 0.20
	•	γ (Comm penalty) = 0.20
	•	δ (Divergence penalty) = 0.05
	•	η (Stability penalty via EMA/deadband) = 0.05

1.4 Mock Agent (no LLM yet; infrastructure check)

Deterministic rules on State → Action:
	•	If forget_mean > 0.03 or divergence > 0.10 → increase replay_ratio by +0.10 (clamp).
	•	If Δacc > +0.01 and ema_loss > 1.5 → increase lr_scale by +0.05 (clamp).
	•	Else: maintain current settings (deadband).
	•	Always set client_selection_k = 4 initially.

Acceptance gate
	•	One full 7-round CIFAR run with State → Action → Applied → Logged via the Mock Agent, with schema validation & clamping on every round; no crashes.

⸻

Phase 2 — SFT Data + Baseline Ablations (no RL yet)

2.1 Build SFT dataset (format-and-priors learning)

Create (State → Action) pairs from two sources:
	1.	Controller-v4 traces collected on CIFAR (multiple seeds/runs).
	2.	Random/extreme State simulations passed through v4 to generate actions
(cover rare combos like high forgetting + low loss, etc.) → more diverse SFT.

Target: ≥ 500 pairs (JSONL is fine). This teaches the LLM the schema + sensible priors.

2.2 Strong baselines for integrity

All baselines must consume the exact same State JSON fields as the LLM.
	•	Fixed: paper default (FedAvg, replay=0.50, lr=1e-4).
	•	Controller-v4: frozen heuristic.
	•	Bandit: ε-greedy over a small discrete action grid
(replay ∈ {0.3, 0.5, 0.7}, lr_scale ∈ {0.9, 1.0, 1.1}, K ∈ {2,4}).
	•	Tiny-MLP policy: simple MLP mapping State features → discrete action grid.
(Critical for the paper: MLP sees the same features as the LLM.)

Acceptance gate
	•	SFT dataset written and checked (valid JSON).
	•	All ablations run end-to-end, produce the same metrics logged in Phase 1.

⸻

Phase 3 — LLM Policy: SFT → PPO (CIFAR sandbox)

3.1 LLM SFT (cheap & stable start)
	•	Model: Llama-3 8B or TinyLlama (whatever we can train cheaply).
	•	Train to output valid Action JSON given State JSON from Phase 2.
	•	Decoding: greedy/low-temp; always schema-validate + clamp before applying.
	•	Safety fallback: if JSON invalid/out-of-bounds → auto-fallback to Controller-v4 for that round (log fallback events).

3.2 PPO (on-policy RL in-the-loop)
	•	Episode = a full 7-round CIFAR run.
	•	Per-step reward = α·AULC − β·Forgetting − γ·Comm − δ·Divergence − η·Stability.
	•	Add KL penalty to keep policy near SFT early on; anneal over time.
	•	Keep v4 fallback if reward collapses or outputs go invalid.

Primary figure to target
	•	Pareto frontier: AULC (y) vs Total Communication (x).
Aim for LLM-PPO to sit above fixed, v4, bandit, and MLP.

Acceptance gate
	•	LLM-SFT beats at least one dimension (e.g., same AULC with lower comm or better stability).
	•	LLM-PPO improves the Pareto frontier vs SFT & all baselines; no catastrophic dips.

⸻

Phase 4 — Transfer to DWRL (final study)
	•	Keep State/Action JSON identical.
	•	Keep bounds identical (replay up to 0.70 still ok).
	•	Use best CIFAR weights (α…η) and best LLM checkpoint; light SFT on a small DWRL trace if needed (optional).
	•	Maintain v4 fallback for safety.

Final experiment matrix
	•	Fixed baseline
	•	Controller-v4
	•	Bandit
	•	Tiny-MLP policy
	•	LLM-SFT
	•	LLM-PPO (Ada-FCL)

Report
	•	Learning curves (acc over rounds) + AULC bars.
	•	Pareto: AULC vs total communication.
	•	Forgetting: avg/max over rounds.
	•	Divergence vs rounds (stability).
	•	Wall-time per round box plots.
	•	#Rounds to reach 75% (or chosen target).

Finish line (publishable)
	•	Ada-FCL (LLM-PPO) dominates on Pareto (better AULC for same/lower comm) and/or yields statistically significant AULC/forgetting gains at similar cost, across ≥3 seeds.

⸻

Global Invariants (enforced throughout)

Bounds & defaults
	•	Base LR: [1e-4, 2e-3] (default 1e-4).
	•	Per-client LR scale: [0.80, 1.20].
	•	Replay ratio: [0.20, 0.70] (default 0.50).
	•	EWC λ: [0.0, 1000.0].
	•	Client selection K: [2, 4] (default 4).
	•	Aggregation: FedAvg; FedProx μ ∈ [0.0, 0.1] (ablation only).
	•	Early stopping: patience = 5.
	•	Local epochs/round: 20 (CIFAR; adjust cautiously for DWRL).
	•	Batch size: 256.
	•	Replay capacity: 2000.
	•	Seeds: 42, 43, 44 (use all for final tables).

Safety & integrity
	•	Schema validation + clamping for every Action field, every round.
	•	v4 fallback on invalid Action or reward collapse; log the fallback.
	•	Tiny-MLP and Bandit must consume the same State fields as the LLM.

Data you must log for plots
	•	AULC running (per round), total comm bytes, wall time, divergence, forgetting (avg & max), plus all chosen actions.

⸻

Paper Outline (so you collect the right artifacts now)
	1.	Motivation: Real-world FCL (non-IID, evolving) needs adaptive control, not fixed hyper-params.
	2.	Method: Ada-FCL — LLM meta-controller (SFT → PPO) emits per-round, per-client actions via a simple JSON protocol (safety-clamped).
	3.	Baselines: Fixed, Controller-v4, Bandit, Tiny-MLP.
	4.	Metrics: AULC (primary), forgetting, communication, time, stability.
	5.	Results: CIFAR sandbox → DWRL transfer; Pareto superiority and stability; ablations prove LLM’s context advantage over simple learners.
	6.	Conclusion: LLMs can act as autonomous controllers for FL/CL—beyond text—delivering better cost/benefit trade-offs.

⸻

Phase Gates (quick checklist)
	•	P1 Gate: Mock Agent round-trips valid JSON; logs complete; clamping + fallback tested.
	•	P2 Gate: SFT dataset (≥500 pairs) built from v4 + simulated states; ablations run clean.
	•	P3 Gate: LLM-SFT improves a dimension; LLM-PPO advances Pareto frontier; no catastrophic dips.
	•	P4 Gate: DWRL reproduces CIFAR gains; plots/tables ready.

⸻

Nice-to-Have (time permitting)
	•	Add client energy/time constraints to State; extend reward to penalize slow clients.
	•	Allow FedProx only when divergence spikes (controller toggles μ).
	•	Track per-class AULC for rare-class stability.
	•	Include predictive uncertainty (e.g., entropy) as an extra State feature for decision quality.

⸻

End of ROADMAP

