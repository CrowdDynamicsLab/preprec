Implementation for the PREPRec paper, accepted at Recsys 2024. Our method enables cross-domain, cross-user zero-shot transfer competitive with in-domain SOTA models.

Quick start: Install packages from `requirements.txt`. Then follow the instructions in `data` folder for getting dataset and preprocessing. Then create a `res` folder to hold trained models and logs of results, and see `sample.sh` for examples for running and evaluating models.

Code credits: Original code is based off [this](https://github.com/pmixer/SASRec.pytorch) pytorch SASRec implementation, with code also taken/repurposed from [here](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), [here](https://github.com/pmixer/TiSASRec.pytorch/), [here](https://github.com/guoyang9/BPR-pytorch/) and [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

Please cite our work if you use it: 
```bibtex
@misc{wang2024pretrainedsequentialrecommendationframework,
      title={A Pre-trained Sequential Recommendation Framework: Popularity Dynamics for Zero-shot Transfer}, 
      author={Junting Wang and Praneet Rathi and Hari Sundaram},
      year={2024},
      eprint={2401.01497},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2401.01497}, 
}
