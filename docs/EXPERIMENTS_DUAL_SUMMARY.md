# Dual Experiments Summary

- metric: `macro_f1`

- note: `outputs/` is gitignored; this file summarizes local runs.


## 1%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_resnet200_3d | ok | 11 | 0.1659 | 2.2255 | 0.3023 | 0.2017 | 0.8444 | 0.2722 | 2 | `outputs/dual_resnet200_3d_1pct_20251223_121328` |
| dual_resnet50_3d | ok | 21 | 0.1628 | 2.6943 | 0.2633 | 0.1804 | 0.8404 | 0.2572 | 5 | `outputs/dual_resnet50_3d_1pct_20251223_103450` |
| dual_resnet152_3d | ok | 25 | 0.1598 | 2.6360 | 0.2438 | 0.1866 | 0.8415 | 0.2472 | 3 | `outputs/dual_resnet152_3d_1pct_20251223_112725` |
| dual_resnet34_3d | ok | 29 | 0.1521 | 2.9132 | 0.3414 | 0.1838 | 0.8401 | 0.2722 | 10 | `outputs/dual_resnet34_3d_1pct_20251223_100441` |
| dual_resnet101_3d | ok | 8 | 0.1479 | 1.7111 | 0.3557 | 0.1874 | 0.8432 | 0.2769 | 4 | `outputs/dual_resnet101_3d_1pct_20251223_110515` |


## 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_resnet101_3d | ok | 31 | 0.2340 | 4.0128 | 0.3609 | 0.2340 | 0.8500 | 0.3181 | 4 | `outputs/dual_resnet101_3d_20pct_20251223_171303` |
| dual_resnet50_3d | ok | 15 | 0.2313 | 1.8588 | 0.3563 | 0.2459 | 0.8580 | 0.3446 | 5 | `outputs/dual_resnet50_3d_20pct_20251223_160110` |
| dual_resnet18_3d | ok | 21 | 0.2262 | 2.1562 | 0.3635 | 0.2303 | 0.8545 | 0.3404 | 13 | `outputs/dual_resnet18_3d_20pct_20251223_134541` |
| dual_resnet34_3d | ok | 18 | 0.2257 | 1.5714 | 0.4018 | 0.2419 | 0.8616 | 0.3621 | 10 | `outputs/dual_resnet34_3d_20pct_20251223_145709` |
| dual_resnet152_3d | ok | 2 | 0.1744 | 1.6401 | 0.4031 | 0.2179 | 0.8562 | 0.3197 | 3 | `outputs/dual_resnet152_3d_20pct_20251223_193103` |


## 100%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_vit_3d | ok | 1 | 0.0667 | 1.9881 | 0.2500 | 0.1667 | 0.8333 | 0.1000 | 1 | `outputs/dual_vit_3d_100pct_20251223_074932` |
| dual_resnet10_3d | ok | 1 | 0.0000 | 1.8031 | 0.0000 | 0.0000 | 0.8333 | 0.0000 | 1 | `outputs/dual_resnet10_3d_100pct_20251223_074920` |
| dual_resnet10_3d | ok | 1 | 0.0000 | 1.8031 | 0.0000 | 0.0000 | 0.8333 | 0.0000 | 1 | `outputs/dual_resnet10_3d_100pct_20251223_075741` |
| dual_resnet10_3d | ok | 1 | 0.0000 | 1.8031 | 0.0000 | 0.0000 | 0.8333 | 0.0000 | 1 | `outputs/dual_resnet10_3d_100pct_20251223_080425` |
| dual_resnet10_3d | ok | 1 | 0.0000 | 1.8031 | 0.0000 | 0.0000 | 0.8333 | 0.0000 | 1 | `outputs/dual_resnet10_3d_100pct_20251223_082822` |
