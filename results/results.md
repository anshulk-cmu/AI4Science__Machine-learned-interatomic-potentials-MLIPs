# Audit results

## Al + Ni, gap 100 A (meV; ideal geometry, then the mean, sd and 95 percent CI over the shaken runs)

| model | task | ideal | mean | sd | 95% CI | verdict |
|---|---|---|---|---|---|---|
| medium | notask | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | CONSISTENT |
| orb-v3 | notask | -427.3 | -501.1 | 6.397 | [-504.1, -498.1] | VIOLATION |
| uma-m-1p1 | omat | 4,283.9 | 3,524.2 | 74.4 | [3,489.4, 3,559.0] | VIOLATION |
| uma-m-1p1 | omol | -317,553.2 | -338,720.4 | 10,601.6 | [-343,682.1, -333,758.7] | VIOLATION |
| uma-s-1p1 | omat | -2,295.5 | -2,154.2 | 39.7 | [-2,172.8, -2,135.6] | VIOLATION |
| uma-s-1p1 | omol | 107,875.9 | 109,367.1 | 922.0 | [108,935.6, 109,798.6] | VIOLATION |
| uma-s-1p1-v1 | omat | -2,237.7 | -2,075.3 | 11.8 | [-2,089.9, -2,060.7] | VIOLATION |
| uma-s-1p1-v1 | omol | 105,424.8 | 106,079.0 | 458.4 | [105,509.8, 106,648.2] | VIOLATION |
| uma-s-1p2 | omat | -2,358.7 | -2,163.3 | 31.9 | [-2,178.2, -2,148.3] | VIOLATION |
| uma-s-1p2 | omol | 130,845.6 | 125,427.7 | 708.2 | [125,096.2, 125,759.2] | VIOLATION |
| uma-s-1p2-v1.1 | omat | -2,312.2 | -2,117.0 | 33.7 | [-2,158.9, -2,075.1] | VIOLATION |
| uma-s-1p2-v1.1 | omol | 127,495.8 | 123,103.7 | 573.9 | [122,391.1, 123,816.4] | VIOLATION |

## Same-element controls, gap 100 A (meV)

| model | task | case | mean | sd | verdict |
|---|---|---|---|---|---|
| medium | notask | AlAl | -0.000 | 0.000 | CONSISTENT |
| medium | notask | NiNi | 0.000 | 0.000 | CONSISTENT |
| orb-v3 | notask | AlAl | 0.029 | 0.055 | INCONCLUSIVE |
| orb-v3 | notask | NiNi | -0.001 | 0.019 | INCONCLUSIVE |
| uma-m-1p1 | omat | AlAl | 0.010 | 0.016 | CONSISTENT |
| uma-m-1p1 | omat | NiNi | 0.064 | 0.025 | VIOLATION |
| uma-m-1p1 | omol | AlAl | 1.398 | 2.033 | CONSISTENT |
| uma-m-1p1 | omol | NiNi | 0.360 | 1.786 | CONSISTENT |
| uma-s-1p1 | omat | AlAl | -0.000 | 0.000 | CONSISTENT |
| uma-s-1p1 | omat | NiNi | -0.000 | 0.000 | CONSISTENT |
| uma-s-1p1 | omol | AlAl | -0.000 | 0.000 | CONSISTENT |
| uma-s-1p1 | omol | NiNi | -0.000 | 0.000 | CONSISTENT |
| uma-s-1p1-v1 | omat | AlAl | -48.3 | 0.622 | VIOLATION |
| uma-s-1p1-v1 | omat | NiNi | 0.910 | 0.185 | VIOLATION |
| uma-s-1p1-v1 | omol | AlAl | 681.9 | 65.5 | VIOLATION |
| uma-s-1p1-v1 | omol | NiNi | -517.6 | 13.6 | VIOLATION |
| uma-s-1p2 | omat | AlAl | -298.2 | 1.420 | VIOLATION |
| uma-s-1p2 | omat | NiNi | -0.011 | 0.298 | INCONCLUSIVE |
| uma-s-1p2 | omol | AlAl | -308.3 | 7.567 | VIOLATION |
| uma-s-1p2 | omol | NiNi | 3,238.3 | 186.5 | VIOLATION |
| uma-s-1p2-v1.1 | omat | AlAl | 0.052 | 0.053 | INCONCLUSIVE |
| uma-s-1p2-v1.1 | omat | NiNi | -0.019 | 0.035 | INCONCLUSIVE |
| uma-s-1p2-v1.1 | omol | AlAl | 0.252 | 0.488 | INCONCLUSIVE |
| uma-s-1p2-v1.1 | omol | NiNi | 5.820 | 13.1 | INCONCLUSIVE |

## Is Delta at 100 A equal to Delta at 8 A, within the tolerance?

| model | task | case | mean difference | flat |
|---|---|---|---|---|
| medium | notask | AlAl | 0.000 | yes |
| medium | notask | AlNi | 0.000 | yes |
| medium | notask | NiNi | -0.000 | yes |
| orb-v3 | notask | AlAl | -0.000 | yes |
| orb-v3 | notask | AlNi | 0.000 | yes |
| orb-v3 | notask | NiNi | 0.000 | yes |
| uma-m-1p1 | omat | AlAl | -0.002 | yes |
| uma-m-1p1 | omat | AlNi | -0.005 | yes |
| uma-m-1p1 | omat | NiNi | 0.049 | NO |
| uma-m-1p1 | omol | AlAl | 0.274 | yes |
| uma-m-1p1 | omol | AlNi | -0.650 | yes |
| uma-m-1p1 | omol | NiNi | 0.738 | yes |
| uma-s-1p1 | omat | AlAl | 0.000 | yes |
| uma-s-1p1 | omat | AlNi | 0.000 | yes |
| uma-s-1p1 | omat | NiNi | 0.000 | yes |
| uma-s-1p1 | omol | AlAl | 0.000 | yes |
| uma-s-1p1 | omol | AlNi | 0.000 | yes |
| uma-s-1p1 | omol | NiNi | 0.000 | yes |
| uma-s-1p1-v1 | omat | AlAl | -0.000 | yes |
| uma-s-1p1-v1 | omat | AlNi | 0.000 | yes |
| uma-s-1p1-v1 | omat | NiNi | 0.000 | yes |
| uma-s-1p1-v1 | omol | AlAl | 0.000 | yes |
| uma-s-1p1-v1 | omol | AlNi | 0.000 | yes |
| uma-s-1p1-v1 | omol | NiNi | 0.000 | yes |
| uma-s-1p2 | omat | AlAl | -0.000 | yes |
| uma-s-1p2 | omat | AlNi | -0.000 | yes |
| uma-s-1p2 | omat | NiNi | 0.000 | yes |
| uma-s-1p2 | omol | AlAl | 0.000 | yes |
| uma-s-1p2 | omol | AlNi | 0.000 | yes |
| uma-s-1p2 | omol | NiNi | 0.000 | yes |
| uma-s-1p2-v1.1 | omat | AlAl | 0.000 | yes |
| uma-s-1p2-v1.1 | omat | AlNi | 0.000 | yes |
| uma-s-1p2-v1.1 | omat | NiNi | -0.000 | yes |
| uma-s-1p2-v1.1 | omol | AlAl | 0.000 | yes |
| uma-s-1p2-v1.1 | omol | AlNi | 0.000 | yes |
| uma-s-1p2-v1.1 | omol | NiNi | 0.000 | yes |

## Ours against the paper's printed value (meV, Al + Ni)

| model | task | ours (mean) | paper |
|---|---|---|---|
| medium | notask | 0.000 | - |
| orb-v3 | notask | -501.1 | -709.7 |
| uma-m-1p1 | omat | 3,524.2 | 1,436.9 |
| uma-m-1p1 | omol | -338,720.4 | -385,824.5 |
| uma-s-1p1 | omat | -2,154.2 | -453.8 |
| uma-s-1p1 | omol | 109,367.1 | 88,670.6 |
| uma-s-1p1-v1 | omat | -2,075.3 | - |
| uma-s-1p1-v1 | omol | 106,079.0 | - |
| uma-s-1p2 | omat | -2,163.3 | - |
| uma-s-1p2 | omol | 125,427.7 | - |
| uma-s-1p2-v1.1 | omat | -2,117.0 | - |
| uma-s-1p2-v1.1 | omol | 123,103.7 | - |
