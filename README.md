# A Comprehensive Study on CTC Loss for Timing Alignment

This repo contains the essential code for the ICASSP 2022 submission:
"A Comprehensive Study on CTC Loss for Timing Alignment", Xingyu Cai, Jiahong Yuan, Renjie Zheng, Kenneth Church.

## Credits
The code is largely based on the following repositories and corresponding references:

* Phoneme recognition: <https://github.com/huggingface/transformers/tree/master/examples/research_projects/wav2vec2>
* Python CTC: <https://github.com/vadimkantorov/ctc>
* Entropy regularized CTC [1]: <https://github.com/liuhu-bigeye/enctc.crnn>

## Install required packages
```bash
pip install requirements.txt
```

## Minimum effort to run
```bash
bash script.sh standard no regular
```
The script runs for CTC using a combination of standard update scheme, no prior and regular vocabulary
It runs on the TIMIT dataset and obtain phoneme-error-rate (PER) and alignment error (MSE and MAE).

## Reference
1. Connectionist Temporal Classification with Maximum Entropy Regularization Hu Liu, Sheng Jin and Changshui Zhang. Neural Information Processing Systems (NeurIPS), 2018.
