# Current Data / Tasks / Models Report

- generated_at: `2025-12-25 00:16:20`
- manifest: `artifacts/manifest_ears.csv`
- dual splits: `artifacts/splits_dual_patient_clustered_v1` (pct=100)

## Label Mapping (label_code 1..6)

| label_code | label_id | name |
|---:|---:|---|
| 1 | 0 | 慢性化脓性中耳炎 |
| 2 | 1 | 中耳胆脂瘤 |
| 3 | 2 | 分泌性中耳炎 |
| 4 | 3 | 胆固醇肉芽肿 |
| 5 | 4 | 正常 |
| 6 | 5 | 其他 |

## Manifest (Ear-level)

- ears(rows)=8024  exams=4012  patients=3989  series=4012
- has_label: true=7692 false=332
- side_counts: {"left": 4012, "right": 4012}
- label_code_counts(labeled ears): 1:2626(34.14%), 2:1156(15.03%), 3:974(12.66%), 4:36(0.47%), 5:2214(28.78%), 6:686(8.92%)
- date_match(labeled ears): {"false": 5185, "true": 2507}
- manufacturer_top15_labeled_ears: {"SIEMENS": 6045, "GE MEDICAL SYSTEMS": 1341, "UIH": 270, "Siemens": 36}
- manufacturer_model_top15_labeled_ears: {"Sensation 64": 4793, "SOMATOM Force": 1252, "Discovery CT750 HD": 1024, "Revolution EVO": 313, "uCT 780": 270, "SOMATOM go.Up (DE)": 36, "Revolution CT": 4}
- convolution_kernel_top15_labeled_ears: {"U75u": 4767, "BONEPLUS2": 1002, "Ur73u": 911, "BONEPLUS": 321, "Ur69u": 239, "H_VSHARP_C": 124, "H_SHARP_C": 118, "['Ur73u', '1']": 52, "Hr60f": 36, "['Ur77u', '3']": 32, "H_SHARP_B": 28, "H60f": 14, "H45f": 10, "['Br59d', '3']": 10, "BONE": 8}
- spacing_z_desc: n=7692 min=-0.625 p10=0.3125 p25=0.6 p50=0.6 p75=0.6 p90=0.6 p95=0.6 p99=0.625 max=5
- abs_spacing_z_desc: n=7692 min=0.3 p10=0.5 p25=0.6 p50=0.6 p75=0.6 p90=0.6 p95=0.6 p99=0.625 max=5
- slice_thickness_desc: n=7692 min=0.4 p10=0.6 p25=0.6 p50=0.6 p75=0.6 p90=0.625 p95=0.625 p99=0.625 max=1.25
- n_instances_desc: n=7692 min=63 p10=133 p25=152 p50=171 p75=189 p90=209 p95=224.4 p99=252 max=467
- date_range: {"min": "2016-01-06", "max": "2022-12-30", "n": 7692}
- label_date_range: {"min": "2015-12-25", "max": "2022-12-30", "n": 7692}
- folder_date_range: {"min": "2016-01-06", "max": "2022-12-30", "n": 7692}

## Caches (Preprocessed)

- dual (3D): `cache/dual_volumes/d32_s224`  files=3859  size=23.08GB
- ear2d: `cache/ears_hu/d32_s224_c192_even`  files=7692  size=23.01GB

## Dual Splits (Exam-level)

- train: 3054 exams  (3043 patients)
- val:   772 exams  (769 patients)
- missing labels: train(left=14, right=7)  val(left=3, right=2)
- train date_match: {"false": 2057, "true": 997}
- val date_match: {"false": 521, "true": 251}
- train both-sides code counts: {"1": 2058, "2": 928, "3": 772, "4": 31, "5": 1745, "6": 553}
- val both-sides code counts: {"1": 546, "2": 214, "3": 196, "4": 5, "5": 451, "6": 127}
- train top15 (left_code,right_code) pairs: {"1,5": 523, "5,1": 442, "1,1": 376, "5,2": 284, "2,5": 240, "3,3": 218, "1,6": 138, "5,3": 111, "6,1": 106, "2,6": 104, "3,5": 99, "6,2": 83, "2,2": 61, "3,6": 38, "1,2": 38}
- val top15 (left_code,right_code) pairs: {"1,5": 136, "5,1": 126, "1,1": 102, "2,5": 69, "3,3": 64, "5,2": 62, "6,1": 32, "1,6": 27, "5,3": 24, "2,6": 23, "3,5": 22, "6,2": 20, "2,2": 12, "3,6": 7, "1,2": 7}

## Outlier Cleaning (By (left_code,right_code) groups)

- removed_rows: 33 (train-only by default)
- removed_pair_counts_top15: {"5_2": 6, "1_1": 5, "5_1": 4, "1_5": 3, "2_5": 3, "6_1": 3, "1_2": 2, "3_3": 2, "6_2": 2, "2_6": 1, "3_5": 1, "5_3": 1}
- files: `artifacts/splits_dual_patient_clustered_v1/100pct/outliers_removed.csv` `artifacts/splits_dual_patient_clustered_v1/100pct/outliers_summary.csv`

## Task Catalog (On This Split)

| task | kind | classes | relevant_codes | train_exams | val_exams | train(pos,neg) | val(pos,neg) |
|---|---|---:|---|---:|---:|---|---|
| cholesteatoma_vs_csoma | binary | 2 | 1,2 | 2486 | 634 | 2058,928 | 546,214 |
| cholesteatoma_vs_other_abnormal | binary | 2 | 1,2,3,4,6 | 3051 | 772 | 928,3414 | 214,874 |
| normal_vs_abnormal | binary | 2 | 1,2,3,4,5,6 | 3054 | 772 | 4342,1745 | 1088,451 |
| normal_vs_cholesteatoma | binary | 2 | 2,5 | 2088 | 522 | 928,1745 | 214,451 |
| normal_vs_cholesterol_granuloma | binary | 2 | 4,5 | 1754 | 452 | 31,1745 | 5,451 |
| normal_vs_csoma | binary | 2 | 1,5 | 2462 | 633 | 2058,1745 | 546,451 |
| normal_vs_diseased | binary | 2 | 1,2,3,4,5 | 3043 | 770 | 3789,1745 | 961,451 |
| normal_vs_ome | binary | 2 | 3,5 | 2089 | 537 | 772,1745 | 196,451 |
| ome_vs_cholesterol_granuloma | binary | 2 | 3,4 | 583 | 137 | 31,772 | 5,196 |
| six_class | multiclass | 6 | - | 3054 | 772 | - | - |

## Current Dual Training Setup (What We Run Now)

- launcher: `scripts/run_dual_models_200ep_max.sh`
- models (3): `dual_resnet200_3d`, `dual_vit_3d`, `dual_unet_3d`
- tasks (6): `normal_vs_abnormal`, `normal_vs_cholesteatoma`, `normal_vs_csoma`, `normal_vs_ome`, `cholesteatoma_vs_other_abnormal`, `normal_vs_cholesterol_granuloma`
- split used: `artifacts/splits_dual_patient_clustered_v1/100pct` (train outliers removed)
- epochs: 200  pct: 100
- speed knobs: `--auto-batch` (max=128), `num_workers=64`, cache=float16
- precision: AMP on, TF32 on, cudnn benchmark on, torch.compile off

