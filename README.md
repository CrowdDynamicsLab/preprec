Implementation for the PREPRec paper, accepted at Recsys 2024. Our method enables cross-domain, cross-user zero-shot transfer competitive with in-domain SOTA models.

Quick start: download `Tools and Home Improvement` and `Office Products` datasets from "Small" subsets for experimentation section of [here](https://nijianmo.github.io/amazon/index.html) and rename as `amazon_tool.csv` and `amazon_office.csv` under `data/amazon` directory. then run `data/preprocess.sh` for preprocessing steps that need to be run before training. then after creating filler folders, `sample.sh` has some examples for running and evaluating models

Coming soon: env dependencies; exact replication scripts for each of the five datasets we evaluate on. 

Credits: Code is based off [this](https://github.com/pmixer/SASRec.pytorch) pytorch SASRec implementation, with code also taken/repurposed from [here](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), [here](https://github.com/pmixer/TiSASRec.pytorch/), [here](https://github.com/guoyang9/BPR-pytorch/) and [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch).
