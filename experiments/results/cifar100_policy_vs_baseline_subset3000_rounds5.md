# CIFAR-100 (subset 3k/client), 4 clients, α=0.2, 2 epochs/round

**Device:** Colab GPU

## Baseline (fixed hyperparams)
Round | Acc
---|---
-1 | 0.010
0  | 0.052
1  | 0.164
2  | 0.274
3  | 0.335
4  | 0.378

## Policy (adaptive controller)
Round | Acc | LR | Replay
---|---|---|---
-1 | 0.010 | 0.00800 | 0.20
0  | 0.129 | 0.00960 | 0.25
1  | 0.404 | 0.01152 | 0.30
2  | 0.561 | 0.01382 | 0.35
3  | 0.621 | 0.01659 | 0.40
4  | 0.658 | — | —

**Takeaway:** Policy beat baseline by **+28.0 pp** at Round 4 (65.8% vs 37.8%).
