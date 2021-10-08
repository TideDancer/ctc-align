# A Comprehensive Study on CTC Loss for Timing Alignment

This repo contains the essential code for the ICASSP 2022 submission:
"A Comprehensive Study on CTC Loss for Timing Alignment", Xingyu Cai, Jiahong Yuan, Renjie Zheng, Kenneth Church.

## Minimum effort to run:
```bash
bash script.sh standard no regular l2
```
The script uses the CTC with standard update scheme, no priors, regular vocabulary, l2 distance for DTW.
It runs on the TIMIT dataset and obtain both phoneme-error-rate (PER) and alignment error (MSE).
