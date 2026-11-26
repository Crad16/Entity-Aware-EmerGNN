### 0. Performance (test macro-F1, mean over epochs 10-20) / validation-best epoch

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| baseline F1 (mean, epochs 10-20) | 0.617 | 0.625 | 0.627 |
| llm F1 (mean, epochs 10-20) | 0.610 | 0.618 | 0.634 |
| name F1 (mean, epochs 10-20) | 0.601 | 0.623 | 0.629 |
| baseline best epoch / test acc | 6 / 0.698 | 20 / 0.748 | 8 / 0.797 |
| llm best epoch / test acc | 16 / 0.719 | 10 / 0.767 | 10 / 0.775 |
| name best epoch / test acc | 20 / 0.699 | 8 / 0.766 | 8 / 0.798 |

### 1. The original attention cannot single out an entity (relation_ties.json)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| test pairs | 32,322 | 30,731 | 32,845 |
| pairs with a 2-hop path | 22,953 | 21,196 | 22,228 |
| 2-hop tied share (all) | 0.889 | 0.886 | 0.875 |
| 2-hop tied share (metabolism) | 0.908 | 0.889 | 0.868 |
| 2-hop tied share (non-PK) | 0.893 | 0.889 | 0.871 |
| 2-hop tie size, median | 23.067 | 20.929 | 23.000 |
| 2-hop chance to single out one path | 0.170 | 0.186 | 0.192 |
| 3-hop tied share | 0.693 | 0.685 | 0.707 |
| 3-hop chance to single out one path | 0.385 | 0.394 | 0.374 |
| metabolism pairs sharing a bound gene | 0.827 | 0.829 | 0.823 |
| metabolism pairs sharing a CYP | 0.825 | 0.826 | 0.818 |
| non-PK pairs sharing a CYP | 0.233 | 0.223 | 0.207 |
| CYP share of tied genes (metabolism) | 0.902 | 0.902 | 0.894 |
| chance to name one specific CYP (metabolism) | 0.531 | 0.523 | 0.528 |

### 2-2. Accuracy / confidence by stratum (mean of llm and name with the gate off, test sample)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| all: accuracy / confidence | 0.705 / 0.900 (n=8195) | 0.761 / 0.899 (n=8200) | 0.791 / 0.918 (n=8191) |
| metabolism: accuracy / confidence | 0.694 / 0.883 (n=1867) | 0.775 / 0.886 (n=1675) | 0.765 / 0.880 (n=1575) |
| other PK: accuracy / confidence | 0.606 / 0.879 (n=1699) | 0.738 / 0.894 (n=1533) | 0.706 / 0.901 (n=1404) |
| PD: accuracy / confidence | 0.615 / 0.901 (n=2115) | 0.641 / 0.889 (n=2349) | 0.716 / 0.914 (n=2537) |
| unspecified: accuracy / confidence | 0.856 / 0.925 (n=2514) | 0.873 / 0.919 (n=2643) | 0.922 / 0.951 (n=2675) |
| shared CYP: accuracy / confidence | 0.617 / 0.877 (n=3657) | 0.728 / 0.889 (n=3245) | 0.703 / 0.895 (n=3068) |
| no shared gene: accuracy / confidence | 0.782 / 0.918 (n=4043) | 0.785 / 0.904 (n=4413) | 0.848 / 0.932 (n=4525) |
| KG low: accuracy / confidence | 0.719 / 0.901 (n=2607) | 0.771 / 0.893 (n=2747) | 0.820 / 0.926 (n=2661) |
| KG mid: accuracy / confidence | 0.720 / 0.906 (n=2709) | 0.750 / 0.897 (n=2717) | 0.818 / 0.924 (n=2685) |
| KG high: accuracy / confidence | 0.679 / 0.893 (n=2879) | 0.763 / 0.907 (n=2736) | 0.739 / 0.904 (n=2845) |
| metabolism & CYP: accuracy / confidence | 0.678 / 0.882 (n=1553) | 0.774 / 0.884 (n=1418) | 0.751 / 0.878 (n=1297) |
| PD & CYP: accuracy / confidence | 0.481 / 0.880 (n=501) | 0.563 / 0.876 (n=491) | 0.556 / 0.898 (n=570) |
| PD & no shared gene: accuracy / confidence | 0.668 / 0.909 (n=1426) | 0.671 / 0.892 (n=1648) | 0.773 / 0.920 (n=1774) |

### 2-3. Share predicted as a metabolism type (full KG)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| metabolism & CYP: baseline | 0.723 | 0.696 | 0.734 |
| metabolism & CYP: llm | 0.714 | 0.788 | 0.746 |
| metabolism & CYP: name | 0.665 | 0.770 | 0.766 |
| other PK & CYP: baseline | 0.296 | 0.179 | 0.242 |
| other PK & CYP: llm | 0.270 | 0.200 | 0.282 |
| other PK & CYP: name | 0.279 | 0.244 | 0.251 |
| other PK & no shared CYP: baseline | 0.080 | 0.025 | 0.028 |
| other PK & no shared CYP: llm | 0.065 | 0.042 | 0.036 |
| other PK & no shared CYP: name | 0.073 | 0.053 | 0.028 |
| PD & CYP: baseline | 0.153 | 0.126 | 0.060 |
| PD & CYP: llm | 0.165 | 0.163 | 0.080 |
| PD & CYP: name | 0.133 | 0.100 | 0.059 |
| PD & no shared CYP: baseline | 0.026 | 0.012 | 0.014 |
| PD & no shared CYP: llm | 0.018 | 0.021 | 0.012 |
| PD & no shared CYP: name | 0.016 | 0.019 | 0.019 |

### 2-4. Removing the CYP edges (F4)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| baseline: accuracy, full KG | 0.714 | 0.749 | 0.796 |
| baseline: accuracy, CYP edges removed | 0.662 | 0.707 | 0.771 |
| baseline: accuracy, controls removed (hub / rand0 / rand1) | 0.710 / 0.712 / 0.711 | 0.742 / 0.745 / 0.745 | 0.789 / 0.793 / 0.793 |
| baseline: metabolism dlogp (CYP edges removed) | -1.031 [-1.355, -0.710]* | -1.316 [-1.702, -0.978]* | -0.544 [-0.881, -0.171]* |
| baseline: PD dlogp | +0.142 [+0.035, +0.249]* | +0.199 [+0.028, +0.424]* | +0.064 [-0.008, +0.137] |
| baseline: other PK dlogp | +0.280 [+0.076, +0.443]* | +0.199 [+0.052, +0.359]* | +0.432 [+0.155, +0.680]* |
| baseline: shared-CYP pairs predicted as metabolism, before -> after | 0.43 -> 0.25 | 0.37 -> 0.22 | 0.39 -> 0.26 |
| baseline: precision of those predictions, before -> after | 0.72 -> 0.73 | 0.81 -> 0.87 | 0.80 -> 0.84 |
| baseline: dlogp of PD pairs that lost a shared receptor (hub) | +0.201 [-0.234, +0.675] | -0.017 [-0.258, +0.166] | -0.080 [-0.323, +0.148] |
| llm: accuracy, full KG | 0.716 | 0.767 | 0.778 |
| llm: accuracy, CYP edges removed | 0.645 | 0.734 | 0.738 |
| llm: accuracy, controls removed (hub / rand0 / rand1) | 0.702 / 0.709 / 0.709 | 0.762 / 0.764 / 0.764 | 0.770 / 0.774 / 0.774 |
| llm: metabolism dlogp (CYP edges removed) | -1.448 [-1.900, -1.018]* | -0.706 [-0.878, -0.566]* | -1.401 [-1.904, -0.962]* |
| llm: PD dlogp | +0.268 [+0.158, +0.395]* | +0.263 [+0.038, +0.610]* | +0.109 [+0.016, +0.197]* |
| llm: other PK dlogp | +0.201 [-0.044, +0.483] | +0.158 [-0.009, +0.363] | +0.286 [+0.023, +0.543]* |
| llm: shared-CYP pairs predicted as metabolism, before -> after | 0.42 -> 0.19 | 0.43 -> 0.30 | 0.40 -> 0.20 |
| llm: precision of those predictions, before -> after | 0.73 -> 0.76 | 0.81 -> 0.84 | 0.79 -> 0.87 |
| llm: dlogp of PD pairs that lost a shared receptor (hub) | -0.014 [-0.354, +0.335] | +0.196 [-0.103, +0.590] | +0.061 [-0.171, +0.340] |
| name: accuracy, full KG | 0.695 | 0.763 | 0.798 |
| name: accuracy, CYP edges removed | 0.659 | 0.731 | 0.762 |
| name: accuracy, controls removed (hub / rand0 / rand1) | 0.692 / 0.694 / 0.694 | 0.755 / 0.761 / 0.760 | 0.794 / 0.797 / 0.796 |
| name: metabolism dlogp (CYP edges removed) | -1.219 [-1.736, -0.736]* | -0.902 [-1.183, -0.645]* | -0.729 [-1.095, -0.362]* |
| name: PD dlogp | +0.269 [+0.088, +0.525]* | +0.226 [-0.027, +0.598] | +0.047 [-0.006, +0.103] |
| name: other PK dlogp | +0.384 [+0.166, +0.618]* | +0.214 [+0.037, +0.415]* | +0.237 [+0.068, +0.387]* |
| name: shared-CYP pairs predicted as metabolism, before -> after | 0.39 -> 0.24 | 0.42 -> 0.29 | 0.40 -> 0.25 |
| name: precision of those predictions, before -> after | 0.72 -> 0.76 | 0.81 -> 0.86 | 0.81 -> 0.83 |
| name: dlogp of PD pairs that lost a shared receptor (hub) | +0.237 [-0.195, +0.737] | +0.197 [-0.066, +0.549] | +0.092 [-0.171, +0.447] |
| llm: metabolism dlogp, gate off | -1.355 [-1.743, -0.983]* | -0.707 [-0.892, -0.537]* | -1.397 [-1.885, -0.949]* |
| llm: metabolism dlogp, gate on - off | -0.092 [-0.189, +0.002] | +0.001 [-0.045, +0.053] | -0.004 [-0.113, +0.121] |
| name: metabolism dlogp, gate off | -1.119 [-1.613, -0.696]* | -0.823 [-1.109, -0.577]* | -0.692 [-1.028, -0.362]* |
| name: metabolism dlogp, gate on - off | -0.100 [-0.223, +0.025] | -0.079 [-0.142, -0.027]* | -0.037 [-0.092, +0.021] |

### 2-5. Removing the emerging drug's KG edges (M1): accuracy

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| baseline: 0% / 50% / 90% / 100% | 0.714 / 0.667 / 0.451 / 0.310 | 0.749 / 0.704 / 0.518 / 0.384 | 0.796 / 0.736 / 0.539 / 0.396 |
| llm: 0% / 50% / 90% / 100% | 0.716 / 0.659 / 0.470 / 0.371 | 0.767 / 0.721 / 0.562 / 0.462 | 0.778 / 0.689 / 0.437 / 0.300 |
| name: 0% / 50% / 90% / 100% | 0.695 / 0.647 / 0.485 / 0.376 | 0.763 / 0.721 / 0.561 / 0.447 | 0.798 / 0.746 / 0.574 / 0.432 |
| reference without KG (known drug's most frequent type) | 0.529 | 0.557 | 0.543 |
| llm - name difference in accuracy drop (90%) | -0.036 [-0.065, -0.008]* | -0.003 [-0.029, +0.019] | -0.117 [-0.155, -0.079]* |

### 2-6. Over-confidence by number of shared neighbours (kg_calibration.json, gated models)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| shared-neighbour cut points a, b (np.digitize: low < a <= mid < b <= high) | 1, 29 | 1, 25 | 1, 27 |
| llm: accuracy low / mid / high | 0.808 / 0.694 / 0.662 | 0.822 / 0.760 / 0.723 | 0.835 / 0.794 / 0.709 |
| llm: confidence low / mid / high | 0.923 / 0.892 / 0.882 | 0.920 / 0.887 / 0.881 | 0.930 / 0.919 / 0.888 |
| llm: accuracy difference (low - high) | +0.145 [+0.094, +0.199]* | +0.099 [+0.048, +0.151]* | +0.127 [+0.070, +0.180]* |
| llm: gap (confidence - accuracy) difference (low - high), raw | -0.104 [-0.150, -0.062]* | -0.060 [-0.104, -0.018]* | -0.084 [-0.129, -0.037]* |
| llm: same, after temperature scaling | -0.079 [-0.125, -0.039]* | -0.033 [-0.078, +0.007] | -0.066 [-0.114, -0.022]* |
| name: accuracy low / mid / high | 0.793 / 0.671 / 0.639 | 0.827 / 0.757 / 0.709 | 0.867 / 0.811 / 0.720 |
| name: confidence low / mid / high | 0.921 / 0.893 / 0.884 | 0.925 / 0.900 / 0.889 | 0.930 / 0.927 / 0.899 |
| name: accuracy difference (low - high) | +0.154 [+0.100, +0.203]* | +0.118 [+0.062, +0.175]* | +0.147 [+0.090, +0.204]* |
| name: gap (confidence - accuracy) difference (low - high), raw | -0.117 [-0.159, -0.073]* | -0.082 [-0.129, -0.037]* | -0.116 [-0.164, -0.068]* |
| name: same, after temperature scaling | -0.085 [-0.128, -0.042]* | -0.062 [-0.112, -0.017]* | -0.097 [-0.149, -0.048]* |

### 3. Gate interventions: query replacement and key permutation (F1 + F2)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| llm: accuracy original / off / mean / shuffle_all / shuffle_type / key_perm | 0.716 / 0.709 / 0.711 / 0.707 / 0.714 / 0.714 | 0.767 / 0.764 / 0.763 / 0.761 / 0.763 / 0.763 | 0.778 / 0.782 / 0.782 / 0.776 / 0.774 / 0.777 |
| name: accuracy original / off / mean / shuffle_all / shuffle_type / key_perm | 0.695 / 0.701 / 0.700 / 0.698 / 0.702 / 0.695 | 0.763 / 0.759 / 0.759 / 0.756 / 0.761 / 0.759 | 0.798 / 0.800 / 0.801 / 0.792 / 0.794 / 0.795 |
| llm: F2 key_perm dlogp (metabolism) | -0.057 [-0.105, -0.010]* | -0.034 [-0.062, -0.009]* | +0.004 [-0.147, +0.140] |
| llm: F2 key_perm dlogp (non-PK) | +0.009 [-0.024, +0.043] | +0.022 [-0.017, +0.070] | -0.041 [-0.083, -0.003]* |
| llm: share of predictions changed by key_perm / off (metabolism) | 6.2% / 5.9% | 5.2% / 5.0% | 11.6% / 9.1% |
| llm: key_perm - off (metabolism) | -0.087 [-0.142, -0.040]* | -0.068 [-0.132, -0.005]* | -0.078 [-0.133, -0.025]* |
| llm: shuffle_all - off (metabolism) | -0.075 [-0.105, -0.047]* | -0.056 [-0.075, -0.039]* | -0.114 [-0.153, -0.072]* |
| llm: mean - off (metabolism) | +0.001 [-0.016, +0.018] | -0.008 [-0.014, -0.001]* | -0.011 [-0.027, +0.004] |
| llm: F1 KL(same type) - KL(any pair) (metabolism) | -0.013 [-0.022, -0.005]* | -0.010 [-0.023, -0.001]* | -0.031 [-0.070, +0.001] |
| llm: F1 (non-PK) | -0.004 [-0.009, +0.001] | -0.002 [-0.009, +0.007] | -0.003 [-0.010, +0.003] |
| llm: key_perm, metabolism & shared CYP | -0.061 [-0.105, -0.017]* | -0.043 [-0.075, -0.015]* | -0.005 [-0.167, +0.145] |
| llm: key_perm, non-PK & shared CYP | +0.053 [+0.013, +0.100]* | +0.002 [-0.061, +0.061] | -0.099 [-0.208, +0.005] |
| llm: key_perm, type 46 | -0.072 [-0.122, -0.034]* | -0.040 [-0.067, -0.013]* | -0.006 [-0.146, +0.140] |
| llm: gate gain NLL(off)-NLL(original), metabolism & CYP | -0.040 [-0.111, +0.025] | -0.034 [-0.104, +0.028] | -0.085 [-0.202, +0.034] |
| llm: gate gain, all | -0.024 [-0.060, +0.010] | -0.003 [-0.033, +0.026] | +0.002 [-0.039, +0.043] |
| name: F2 key_perm dlogp (metabolism) | -0.079 [-0.152, -0.031]* | -0.133 [-0.186, -0.076]* | -0.019 [-0.050, +0.009] |
| name: F2 key_perm dlogp (non-PK) | +0.020 [-0.018, +0.070] | +0.018 [-0.040, +0.106] | +0.009 [-0.000, +0.019] |
| name: share of predictions changed by key_perm / off (metabolism) | 4.7% / 7.4% | 6.1% / 4.8% | 2.6% / 6.2% |
| name: key_perm - off (metabolism) | -0.023 [-0.152, +0.125] | -0.120 [-0.164, -0.084]* | -0.037 [-0.135, +0.069] |
| name: shuffle_all - off (metabolism) | -0.003 [-0.053, +0.050] | -0.035 [-0.056, -0.016]* | -0.033 [-0.074, +0.011] |
| name: mean - off (metabolism) | +0.049 [+0.016, +0.080]* | +0.018 [+0.007, +0.028]* | +0.003 [-0.016, +0.028] |
| name: F1 KL(same type) - KL(any pair) (metabolism) | -0.029 [-0.044, -0.015]* | -0.002 [-0.012, +0.008] | -0.007 [-0.016, +0.002] |
| name: F1 (non-PK) | -0.008 [-0.024, +0.004] | -0.011 [-0.018, -0.004]* | -0.006 [-0.012, +0.000] |
| name: key_perm, metabolism & shared CYP | -0.078 [-0.135, -0.031]* | -0.135 [-0.187, -0.084]* | -0.028 [-0.060, -0.000]* |
| name: key_perm, non-PK & shared CYP | +0.000 [-0.048, +0.040] | +0.024 [-0.078, +0.140] | +0.034 [+0.005, +0.067]* |
| name: key_perm, type 46 | -0.049 [-0.083, -0.023]* | -0.129 [-0.181, -0.070]* | -0.031 [-0.060, -0.005]* |
| name: gate gain NLL(off)-NLL(original), metabolism & CYP | +0.070 [-0.086, +0.247] | +0.013 [-0.040, +0.062] | -0.006 [-0.098, +0.092] |
| name: gate gain, all | +0.037 [-0.025, +0.106] | -0.013 [-0.070, +0.029] | -0.003 [-0.052, +0.050] |
| llm: strata with a significant gate gain | 0 / 19 (gain 0, loss 0) | 3 / 19 (gain 3, loss 0) | 2 / 19 (gain 1, loss 1) |
| name: strata with a significant gate gain | 1 / 19 (gain 1, loss 0) | 3 / 19 (gain 3, loss 0) | 3 / 19 (gain 0, loss 3) |

### 4. Does the gate rank the shared CYP first? (E2, E4)

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| EmerGNN (ties): AUC metabolism | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| EmerGNN (ties): AUC non-PK | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| EmerGNN (ties): AUC gap | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| EmerGNN (ties): AUC metabolism, no degree advantage | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| degree order: AUC metabolism | 0.716 [0.657, 0.792]* | 0.737 [0.678, 0.807]* | 0.745 [0.692, 0.827]* |
| degree order: AUC non-PK | 0.866 [0.828, 0.909]* | 0.879 [0.835, 0.921]* | 0.882 [0.834, 0.923]* |
| degree order: AUC gap | -0.150 [-0.207, -0.085]* | -0.142 [-0.206, -0.081]* | -0.137 [-0.186, -0.067]* |
| degree order: AUC metabolism, no degree advantage | 0.000 [0.000, 0.000]* | 0.000 [0.000, 0.001]* | 0.000 [0.000, 0.001]* |
| llm best: AUC metabolism | 0.650 [0.522, 0.786]* | 0.429 [0.347, 0.498]* | 0.574 [0.515, 0.632]* |
| llm best: AUC non-PK | 0.603 [0.540, 0.665]* | 0.450 [0.403, 0.504] | 0.477 [0.431, 0.527] |
| llm best: AUC gap | +0.047 [-0.065, +0.169] | -0.021 [-0.119, +0.061] | +0.097 [+0.039, +0.147]* |
| llm best: AUC metabolism, no degree advantage | 0.623 [0.465, 0.774] | 0.381 [0.259, 0.485]* | 0.598 [0.536, 0.662]* |
| name best: AUC metabolism | 0.568 [0.526, 0.625]* | 0.739 [0.682, 0.792]* | 0.558 [0.469, 0.633] |
| name best: AUC non-PK | 0.455 [0.407, 0.512] | 0.515 [0.447, 0.584] | 0.589 [0.539, 0.638]* |
| name best: AUC gap | +0.114 [+0.060, +0.173]* | +0.225 [+0.136, +0.312]* | -0.031 [-0.114, +0.040] |
| name best: AUC metabolism, no degree advantage | 0.530 [0.497, 0.566] | 0.723 [0.668, 0.779]* | 0.523 [0.405, 0.611] |
| llm: rank correlation of gate and degree, largest absolute value (genes) | 0.057 | 0.147 | 0.139 |
| llm: CYP gate shift after controlling for degree | +0.052 [+0.030, +0.072]* | -0.057 [-0.066, -0.045]* | +0.063 [+0.053, +0.074]* |
| name: rank correlation of gate and degree, largest absolute value (genes) | 0.086 | 0.090 | 0.061 |
| name: CYP gate shift after controlling for degree | +0.044 [+0.035, +0.053]* | +0.080 [+0.066, +0.095]* | -0.032 [-0.036, -0.027]* |

### 5. llm against name

| item | S1_1 | S1_12 | S1_123 |
|---|---|---|---|
| AUC metabolism, llm - name | 0.082 | -0.311 | 0.016 |
| AUC non-PK, llm - name | 0.148 | -0.065 | -0.112 |
| NLL(name) - NLL(llm), all | +0.095 [-0.098, +0.299] | +0.059 [-0.076, +0.202] | +0.046 [-0.153, +0.287] |
| NLL(name) - NLL(llm), metabolism | +0.055 [-0.264, +0.387] | +0.133 [-0.092, +0.427] | +0.277 [-0.427, +1.091] |
| NLL(name) - NLL(llm), other PK | +0.054 [-0.284, +0.405] | +0.303 [-0.074, +0.666] | -0.024 [-0.216, +0.188] |
| NLL(name) - NLL(llm), PD | -0.035 [-0.560, +0.438] | -0.167 [-0.465, +0.142] | -0.003 [-0.305, +0.330] |
| NLL(name) - NLL(llm), unspecified | +0.261 [+0.054, +0.537]* | +0.070 [-0.058, +0.195] | -0.006 [-0.170, +0.133] |
