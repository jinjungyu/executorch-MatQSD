# 9-1. Speculative Decoding Batch Verification 결과 (2026-03-27)

## 달성 결과

**단일 mqint8 .pte로 Speculative Decoding 구현, K=3에서 AR 대비 1.14× speedup 달성.**

## SD 성능 (Llama-3.2-1B, M4 MacBook, 128 tokens)

| K | α (accept) | τ (tok/step) | SD TPS | AR-8bit TPS | Speedup |
|---|-----------|-------------|--------|-------------|---------|
| **3** | 80.4% | 3.39 | **62.1** | 54.7 | **1.14×** |
| 5 | 75.4% | 4.78 | 51.1 | 54.1 | 0.95× |
| 7 | 75.4% | 6.14 | 54.1 | 55.1 | 0.98× |

## Cycle Timing Breakdown

### Draft (4-bit, mr=1 GeMV)

| K | Draft time | Per token |
|---|-----------|-----------|
| 3 | 32.8 ms | 10.9 ms |
| 5 | 55.3 ms | 11.1 ms |
| 7 | 75.0 ms | 10.7 ms |

### Verify (8-bit, mr=4 GeMM batch) — 핵심 최적화

| K+1 tokens | mr=1 (이전) | **mr=4 (현재)** | 가속 |
|-------------|------------|----------------|------|
| 4 (K=3) | 45.6 ms | **21.8 ms** | **2.1×** |
| 6 (K=5) | 69.2 ms | **38.2 ms** | **1.8×** |
| 8 (K=7) | 89.7 ms | **38.6 ms** | **2.3×** |

### Verify 내부 상세

| 구간 | K=3 | K=5 | K=7 |
|------|-----|-----|-----|
| Prep (tensor) | ~0 ms | ~0 ms | ~0 ms |
| **Step (embed+fwd+unembed)** | **20.9 ms** | **36.8 ms** | **36.8 ms** |
| Argmax | 0.9 ms | 1.4 ms | 1.8 ms |
| Accept/reject | ~0 ms | ~0 ms | ~0 ms |
| **Per-token (Step)** | **5.2 ms** | **6.1 ms** | **4.6 ms** |

## Embed / Fwd / Unembed 분석

| 구간 | Per-token marginal |
|------|--------------------|
| Embed + Transformer Fwd | ~10.5 ms |
| Unembed (output projection) | ~0.2 ms |
| **전체 bottleneck: Transformer Forward** | |

Unembedding은 무시 가능. full_logits vs last_logit 모델 차이 <1%.

## 구현 상세

### 4x16c8 GeMM 커널
- `qd8-f32-mqint8-gemm-4x16c8-minmax-neoni8mm.c` (신규)
- 4 activation rows 동시 처리 (a0-a3)
- Weight 1회 로드 → 4 row pair (01, 23)에 재사용
- 16 accumulator pairs (vacc01x..., vacc23x...)
- Mode switching: 4-bit (nibble extract) / 8-bit (RECON)
- Global region separation 지원 (lower pointer from params)

### Config 등록
```c
dqgemm[XNN_MR_TO_INDEX(1)] = 1x16c8__neoni8mm;  // batch=1 (AR decode)
dqgemm[XNN_MR_TO_INDEX(4)] = 4x16c8__neoni8mm;  // batch>1 (SD verify)
config.mr = 4;  // batch>1일 때 자동 선택
```

### Dispatch 동작
```
batch_size=1 → mr=1 → 1x16c8 GeMV (AR decode, 87/55 TPS)
batch_size>1 → mr=4 → 4x16c8 GeMM (SD verify, weight 재사용)
```

### SD Runner (`sd_runner.cpp`)
- `-mqint8_model`: 단일 .pte 경로
- Draft: MQINT8_MODE=0 (4-bit), 순차 K tokens
- Verify: MQINT8_MODE=1 (8-bit), batch `[1, K+1]` forward
- `--generate_full_logits` export 필요 (모든 position logits)
- Batch verification: `all_argmax()` 로 전체 position 비교

### Export
```bash
python export_matqsd_8bit.py --matgptq_dir ... -- ... --generate_full_logits --output_name matqsd_fulllogits.pte
```

## AR Inference 성능 (변경 없음)

| Mode | TPS | 비고 |
|------|-----|------|
| mqint8 4-bit | 87 | qb4w(90) 대비 97% |
| mqint8 8-bit | 55 | 8da8w(53)와 동일 |
| qb4w reference | 90 | |
| 8da8w reference | 53 | |

## 핵심 최적화 히스토리 (전체)

| 단계 | 4-bit TPS | SD speedup | 핵심 |
|------|----------|------------|------|
| 초기 | 49 | - | dotprod, interleaved layout |
| i8mm 커널 | 50 | - | vmmlaq (compute bound 아님) |
| NR 내부 region 분리 | 63 | - | upper+scale → lower |
| Global region 분리 | 88 | - | lower를 전체 뒤로 |
| Batch verify (mr=1) | 88 | 0.77× | sequential GeMV verify |
| **Batch verify (mr=4)** | **88** | **1.14× (K=3)** | **4x16c8 GeMM 커널** |

## 알려진 제한사항

1. **K=3이 최적**: K가 커지면 verify 시간 증가로 speedup 감소
2. **generate_full_logits 필수**: SD용 모델 재export 필요
3. **모델 2개 로드**: draft/target 별도 인스턴스 (메모리 2×)
4. **mr=4까지만 구현**: mr=6,8 커널으로 추가 가속 가능성 있음
