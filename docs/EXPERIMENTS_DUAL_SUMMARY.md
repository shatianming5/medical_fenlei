# Dual Experiments Summary

- metric: `macro_f1`

- note: `outputs/` is gitignored; this file summarizes local runs.


## task: `cholesteatoma_vs_other_abnormal`


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet__cholesteatoma_vs_other_abnormal_20pct_e50_20251225_065121` |


## task: `normal_vs_abnormal`


### 1%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet50 | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet50__normal_vs_abnormal_1pct_20251224_042339` |


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_dual_resnet200_3d | ok | 24 | 0.5948 | 0.6295 | 0.6582 | 0.5966 | 0.5966 | 0.6611 | 11 | `outputs/dual_dual_resnet200_3d__normal_vs_abnormal_20pct_e200_20251225_014817` |
| dual_dual_resnet200_3d | ok | 2 | 0.4812 | 1.0994 | 0.5049 | 0.5025 | 0.5025 | 0.5271 | 11 | `outputs/dual_dual_resnet200_3d__normal_vs_abnormal_20pct_e200_20251225_011212` |
| dual_dual_unet_3d | ok | 1 | 0.4142 | 0.6409 | 0.7070 | 0.5000 | 0.5000 | 0.5856 | 128 | `outputs/dual_dual_unet_3d__normal_vs_abnormal_20pct_e200_20251225_042430` |
| dual_dual_vit_3d | ok | 1 | 0.4142 | 0.6417 | 0.7070 | 0.5000 | 0.5000 | 0.5856 | 128 | `outputs/dual_dual_vit_3d__normal_vs_abnormal_20pct_e200_20251225_041331` |
| dual_dual_vit_3d | ok | 1 | 0.4142 | 0.6692 | 0.7070 | 0.5000 | 0.5000 | 0.5856 | 128 | `outputs/dual_dual_vit_3d__normal_vs_abnormal_20pct_e200_20251225_041756` |


### 100%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_dual_resnet200_3d | ok | 22 | 0.5972 | 0.6246 | 0.6914 | 0.5928 | 0.5928 | 0.6778 | 11 | `outputs/dual_dual_resnet200_3d__normal_vs_abnormal_100pct_e200_20251224_225533` |
| dual_dual_resnet200_3d | ok | 20 | 0.5954 | 0.6242 | 0.6608 | 0.5965 | 0.5965 | 0.6627 | 11 | `outputs/dual_dual_resnet200_3d__normal_vs_abnormal_100pct_e200_20251224_193604` |
| dual_dual_resnet200_3d | ok | 3 | 0.5498 | 0.6679 | 0.5724 | 0.5788 | 0.5788 | 0.5916 | 11 | `outputs/dual_dual_resnet200_3d__normal_vs_abnormal_100pct_e200_20251225_004431` |
| dual_dual_vit_3d | ok | 1 | 0.2266 | 1.4864 | 0.2930 | 0.5000 | 0.5000 | 0.1328 | 32 | `outputs/dual_dual_vit_3d__normal_vs_abnormal_100pct_e200_20251225_051614` |
| ear2d_resnet50 | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet50__normal_vs_abnormal_100pct_20251224_042339` |


## task: `normal_vs_cholesteatoma`


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet__normal_vs_cholesteatoma_20pct_e50_20251225_065121` |


## task: `normal_vs_cholesterol_granuloma`


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet__normal_vs_cholesterol_granuloma_20pct_e50_20251225_065121` |


## task: `normal_vs_csoma`


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet__normal_vs_csoma_20pct_e50_20251225_065121` |


## task: `normal_vs_diseased`


### 1%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| _debug_dual_resnet10_task1 | ok | 4 | 0.4095 | 0.7749 | 0.6813 | 0.5017 | 0.5017 | 0.5542 | 16 | `outputs/_debug_dual_resnet10_task1_1pct_ls0` |
| _smoke_dual_resnet10_task1 | ok | 1 | 0.2421 | 0.7103 | 0.3194 | 0.5000 | 0.5000 | 0.1546 | 8 | `outputs/_smoke_dual_resnet10_task1_1pct_v2` |


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet__normal_vs_diseased_20pct_e50_20251225_082146` |
| ear2d_unet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_unet__normal_vs_diseased_20pct_e50_20251225_082146` |
| ear2d_vit | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_vit__normal_vs_diseased_20pct_e50_20251225_082146` |


### 100%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_dual_vit_3d | ok | 113 | 0.5493 | 0.6834 | 0.6154 | 0.5486 | 0.5486 | 0.6116 | 128 | `outputs/dual_dual_vit_3d__normal_vs_diseased_100pct_e200_20251225_195351` |
| _debug_dual_resnet50_task1 | ok | 2 | 0.5103 | 0.6931 | 0.5106 | 0.5769 | 0.5769 | 0.5147 | 16 | `outputs/_debug_dual_resnet50_task1_100pct_e5` |
| _debug_dual_unet_task1 | ok | 1 | 0.4050 | 0.6656 | 0.6806 | 0.5000 | 0.5000 | 0.5512 | 2 | `outputs/_debug_dual_unet_task1_100pct_cw_e3` |
| _debug_dual_vit_task1 | ok | 1 | 0.4050 | 0.7681 | 0.6806 | 0.5000 | 0.5000 | 0.5512 | 8 | `outputs/_debug_dual_vit_task1_100pct_nocw_e3` |
| dual_dual_unet_3d | ok | 1 | 0.4050 | 0.6547 | 0.6806 | 0.5000 | 0.5000 | 0.5512 | 128 | `outputs/dual_dual_unet_3d__normal_vs_diseased_100pct_e200_20251225_195351` |


## task: `normal_vs_ome`


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear2d_resnet | ok | - | - | - | - | - | - | - | - | `outputs/ear2d_resnet__normal_vs_ome_20pct_e50_20251225_065121` |


## task: `six_class`


### 1%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_dual_resnet50_3d | ok | 4 | 0.0834 | 1.8583 | 0.2891 | 0.1682 | 0.8335 | 0.1363 | 22 | `outputs/dual_dual_resnet50_3d__six_class_1pct_20251224_042339` |


### 20%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_dual_resnet50_3d | ok | 6 | 0.1766 | 1.5782 | 0.3554 | 0.2162 | 0.8529 | 0.2973 | 22 | `outputs/dual_dual_resnet50_3d__six_class_20pct_20251224_042339` |


### 100%

| model | status | best_epoch | best_metric | val_loss | acc | macro_recall | macro_spec | weighted_f1 | batch | run_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dual_dual_resnet200_3d | ok | 78 | 0.2541 | 2.0258 | 0.3587 | 0.2621 | 0.8595 | 0.3596 | 11 | `outputs/dual_dual_resnet200_3d__six_class_100pct_e200_20251224_064154` |
| dual_dual_resnet50_3d | ok | 10 | 0.2308 | 1.4972 | 0.4477 | 0.2583 | 0.8673 | 0.3837 | 22 | `outputs/dual_dual_resnet50_3d__six_class_100pct_20251224_042339` |
