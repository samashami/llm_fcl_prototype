# LLM-guided Federated Continual Learning (LLM-FCL)

Prototype research project exploring how **Large Language Models (LLMs)** can guide
**Federated Continual Learning (FCL)** strategies in image classification tasks.

## 📂 Project Structure
llm_fcl_prototype
├── src
│   ├── run_llm_fcl.py        # Main training loop (entry point)
│   ├── data.py               # Dataset loading + client splits
│   ├── model.py              # Model definitions (e.g., ResNet18)
│   ├── fl.py                 # Federated Learning logic (Client, Server, FedAvg)
│   ├── policy.py             # LLM-guided policy for tuning hyperparams
│   └── strategies            # Continual learning strategies
│       ├── replay.py         # Replay buffer
│       └── ewc.py            # Elastic Weight Consolidation (optional)
├── prompts
│   └── policy_prompt.txt     # Prompt template for the LLM policy
├── experiments
│   └── plan.md               # Experiment plan & notes
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── .gitignore                # Git ignore file

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/llm_fcl_prototype.git
cd llm_fcl_prototype

2. Install dependencies
pip install -r requirements.txt

3. Run the prototype (step-by-step implementation)
python -m src.run_llm_fcl

📝 Roadmap
	•	Repo scaffold created
	•	Implement data loading (CIFAR-100 with non-IID client splits)
	•	Add ResNet18 model
	•	Implement Federated Learning (FedAvg)
	•	Add Continual Learning strategies (Replay, EWC)
	•	Connect LLM-guided policy
	•	Run experiments & generate plots

   📖 Research Context

This project builds on previous work in:
	•	Federated Learning (FL): collaborative model training without centralizing data.
	•	Continual Learning (CL): adapting models to evolving data streams while mitigating catastrophic forgetting.
	•	Federated Continual Learning (FCL): combines FL and CL, but suffers from instability under non-IID data.
	•	LLMs for meta-learning: here we explore if LLMs can guide hyperparameter tuning or replay strategies dynamically.

⸻

📊 Planned Experiments
	•	Baselines: FedAvg + Replay (fixed), FedAvg + Replay + EWC (fixed).
	•	LLM-FCL: LLM-guided dynamic tuning of replay ratio, learning rate, and EWC λ.
	•	Datasets: CIFAR-100, TrashNet, and optionally DWRL if available.
	•	Metrics: accuracy, per-class recall, forgetting, stability.

⸻

🧑‍💻 Authors
	•	Somayeh Shami (PhD candidate, TU Graz)
	•	Collaborators: [to be added]
## 📊 Results (CIFAR-100 pilot)

**Setup:** ResNet-18 (ImageNet pretrained), 4 clients (α=0.2), 2 epochs/round, 3k images/client, batch 128, SGD lr=1e-3, replay on.

| Round | Baseline Acc | Policy Acc | Policy LR | Policy Replay |
|------:|-------------:|-----------:|-----------:|--------------:|
| -1    | 0.010        | 0.010      | 0.00800    | 0.20 |
| 0     | 0.052        | 0.129      | 0.00960    | 0.25 |
| 1     | 0.164        | 0.404      | 0.01152    | 0.30 |
| 2     | 0.274        | 0.561      | 0.01382    | 0.35 |
| 3     | 0.335        | 0.621      | 0.01659    | 0.40 |
| 4     | 0.378        | 0.658      | —          | — |

**Summary:** The adaptive policy outperforms fixed hyperparameters by **+28.0 pp** at Round 4 (65.8% vs 37.8%), mainly by ramping LR (≈0.008→0.0166) and replay (0.20→0.40).  
See details in `experiments/results/cifar100_policy_vs_baseline_subset3000_rounds5.md`.

