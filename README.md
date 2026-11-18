# 학습된 체크포인트 (DrugBank)

이 브랜치는 **체크포인트만** 담고 있습니다. 코드는 없습니다.
코드는 `main` / `analysis` / `fold-replication` 브랜치에 있습니다.

- 작성: 2026-11-18
- 모두 `seed0`, `--shuffle_order`, `--attn_mode=learned`로 학습한 것입니다.
- 파일 이름 규칙은 `analysis/REPRODUCE.md`가 요구하는 것과 같습니다. 그대로 복사하면 분석 스크립트가 바로 찾습니다.

## 받는 법

레포를 이미 받았다면 체크포인트만 현재 작업 트리에 꺼낼 수 있습니다.

```bash
git fetch origin checkpoints && git checkout origin/checkpoints -- DrugBank/saved_models
```

새로 받는다면 이 브랜치만 받으면 됩니다 (약 230 MB).

```bash
git clone --branch checkpoints --single-branch git@github.com:Crad16/Entity-Aware-EmerGNN.git ckpt
```

개별 파일은 깃헙 웹에서 하나씩 내려받아도 됩니다.

## 쓰기 전에 필요한 것

체크포인트만으로는 게이트가 돌지 않습니다. 엔티티 임베딩 `.npy`가 있어야 합니다.
이 파일들은 레포에 넣지 않습니다 (각 100 MB). 코드 브랜치에서 한 번 만들면 됩니다.

```bash
python entity_text/embed.py
```

`entity_text/emb/drugbank_{name,llm,random}.npy`가 만들어집니다. GPU로 1분 걸립니다.

데이터 분할(`DrugBank/data/S1_*`)은 Zenodo에서 받습니다. `main`의 README를 보세요.

## 목록

공통 하이퍼파라미터: `length=3`, `n_dim=64`, `gate_dim=32`, `feat=M`, `lr=1e-3`,
`lamb=1e-8`, `n_batch=32`, 20 epoch. 게이트가 있는 run은 `gate_lr=3e-3`.
`best`는 validation macro-F1이 가장 높은 epoch, `last`는 마지막 epoch(20)입니다.

### S1_1, gate v2 — 지금 분석에 쓰는 것

| 파일 (`S1_1_learned_shuf...`) | entity_feat | best epoch | valid F1 | test F1 | test acc | test F1 평균 (ep10-20) |
|---|---|---|---|---|---|---|
| `_seed0_{best,last}.pt` | none (baseline) | 14 | 0.6366 | — | — | — |
| `_gate-id-v2_seed0_{best,last}.pt` | id | 18 | 0.6525 | 0.6534 | 0.7094 | 0.628 |
| `_gate-random-v2_seed0_{best,last}.pt` | random | 18 | 0.6480 | 0.6561 | 0.7214 | 0.630 |
| `_gate-name-v2_seed0_{best,last}.pt` | name | 20 | 0.6388 | 0.5959 | 0.6992 | 0.601 |
| `_gate-llm-v2_seed0_{best,last}.pt` | llm | 16 | 0.6445 | 0.6019 | 0.7186 | 0.610 |

### S1_1, gate v1 — 옛 게이트 식, 비교용

`--gate_version v1`을 줘야 불러와집니다. v1은 `2 sigmoid(...)`, v2는 `1 + 0.5 tanh(...)`입니다.

| 파일 (`S1_1_learned_shuf...`) | entity_feat | best epoch | valid F1 | test F1 | test acc | test F1 평균 (ep10-20) |
|---|---|---|---|---|---|---|
| `_gate-id_seed0_{best,last}.pt` | id | 12 | 0.6564 | 0.6649 | 0.7342 | 0.632 |
| `_gate-random_seed0_{best,last}.pt` | random | 16 | 0.6600 | 0.6367 | 0.7065 | 0.612 |
| `_gate-name_seed0_{best,last}.pt` | name | 20 | 0.6645 | 0.6384 | 0.7223 | 0.614 |
| `_gate-llm_seed0_{best,last}.pt` | llm | 20 | 0.6640 | 0.6150 | 0.7185 | 0.603 |

### S1_12, gate v2

| 파일 (`S1_12_learned_shuf...`) | entity_feat | best epoch | valid F1 | test F1 | test acc | test F1 평균 (ep10-20) |
|---|---|---|---|---|---|---|
| `_seed0_{best,last}.pt` | none (baseline) | 20 | 0.7301 | 0.6552 | 0.7481 | 0.625 |
| `_gate-llm-v2_seed0_{best,last}.pt` | llm | 10 | 0.7163 | 0.6223 | 0.7669 | 0.618 |
| `_gate-name-v2_seed0_{best,last}.pt` | name | 8 | 0.7319 | 0.6099 | 0.7657 | 0.623 |

### S1_123, gate v2

| 파일 (`S1_123_learned_shuf...`) | entity_feat | best epoch | valid F1 | test F1 | test acc | test F1 평균 (ep10-20) |
|---|---|---|---|---|---|---|
| `_seed0_{best,last}.pt` | none (baseline) | 8 | 0.6446 | 0.6333 | 0.7966 | 0.627 |
| `_gate-llm-v2_seed0_{best,last}.pt` | llm | 10 | 0.6248 | 0.6314 | 0.7753 | 0.634 |
| `_gate-name-v2_seed0_{best,last}.pt` | name | 8 | 0.6259 | 0.6372 | 0.7985 | 0.629 |

숫자 출처: `results/curves/drugbank_<split>_<run>.csv` (코드 브랜치).
S1_1 baseline만 예외입니다. 아래를 보세요.

## 주의: S1_1 baseline은 재학습본입니다

`S1_1_learned_shuf_seed0_best.pt`는 2026-10-30에 **다시 학습한** baseline입니다.
`results/curves/drugbank_S1_1_emergnn.csv`에 있는 40 epoch run과는 **다른 run**입니다.

- 재학습본 곡선: `DrugBank/saved_models/retrained_baseline/validation_curve.csv`
  (이 브랜치에 같이 넣었습니다). best는 epoch 14, valid F1 0.6366입니다.
- `results/fold_summary.md`의 "baseline 최적 epoch / test acc = 6 / 0.698"과 "F1 0.617"은
  곡선 파일(옛 run)에서 나온 값입니다.
- "baseline: 정확도 전체 KG = 0.714"처럼 체크포인트를 직접 돌려서 나온 값은 재학습본 기준입니다.

같은 baseline을 두 군데서 읽고 있으니, 보고서에 숫자를 옮길 때 어느 쪽인지 확인하세요.

## 게이트만 꺼내 쓰기

게이트 가중치는 GPU 없이도 읽힙니다. 약 쌍 임베딩과 엔티티 key만 있으면 되니
GNN을 돌리지 않고 게이트 값을 그대로 계산할 수 있습니다.
`analysis/gate_degree.py`가 그 방식을 그대로 보여 줍니다. 요지는 세 줄입니다.

1. 체크포인트에서 `entity_gate.` 로 시작하는 항목만 떼어 `EntityGate`에 넣습니다.
   (`EntityGate(key, 'entity_text/emb/drugbank', 34124, 64, 3, 32, 'v2')`)
2. `ent_kg`, `Went.weight`, `Went.bias`로 엔티티를 약 공간으로 보냅니다.
3. 약 쌍 벡터와 `gate.keys()`를 게이트에 넣고 층 평균을 냅니다.

## 무결성

```bash
sha256sum -c SHA256SUMS
```
