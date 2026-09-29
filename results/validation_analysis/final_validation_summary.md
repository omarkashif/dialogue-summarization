# Final validation summary

Validation only; the SAMSum test split was not loaded or evaluated.

## Architecture comparison — greedy decoding

| Model | Validation loss | Parameters | R1 | R2 | RL | Mean / median length | EOS % | Max-length % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 4.4025 | 17,122,465 | 0.1321 | 0.0143 | 0.1151 | 13.18 / 12 | 99.63 | 0.37 |
| Attention | 3.9091 | 22,865,569 | 0.2828 | 0.0782 | 0.2381 | 15.72 / 13 | 96.70 | 3.30 |

## Decoding comparison

| Model | Decoding | R1 | R2 | RL | Mean length | Decode time |
|---|---|---:|---:|---:|---:|---:|
| Vanilla | greedy | 0.1321 | 0.0143 | 0.1151 | 13.18 | 1.74 s |
| Vanilla | beam-3 | 0.1314 | 0.0187 | 0.1085 | 12.63 | 52.53 s |
| Vanilla | beam-5 | 0.1174 | 0.0186 | 0.0995 | 11.72 | 87.54 s |
| Attention | greedy | 0.2828 | 0.0782 | 0.2381 | 15.72 | 3.24 s |
| Attention | beam-3 | 0.2706 | 0.0800 | 0.2332 | 12.41 | 69.90 s |
| Attention | beam-5 | 0.2672 | 0.0791 | 0.2318 | 11.63 | 98.45 s |

## Length-stratified architecture comparison

Training-derived stored-source boundaries: short ≤ 71; medium ≤ 149; long > 149. Stored source length includes `<SOS>` and `<EOS>`.

| Group | n | Mean source length | Vanilla RL | Attention RL |
|---|---:|---:|---:|---:|
| short | 291 | 46.58 | 0.1062 | 0.2823 |
| medium | 262 | 105.41 | 0.1176 | 0.2265 |
| long | 265 | 221.66 | 0.1210 | 0.1998 |
