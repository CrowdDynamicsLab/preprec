based off [this](https://github.com/pmixer/SASRec.pytorch) pytorch SASRec implementation, with code also taken/repurposed from [here](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), [here](https://github.com/pmixer/TiSASRec.pytorch/), [here](https://github.com/guoyang9/BPR-pytorch/) and [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

keeping runs and model results local for now.

download `Tools and Home Improvement` and `Office Products` datasets from "Small" subsets for experimentation section of [here](https://nijianmo.github.io/amazon/index.html) and rename as `amazon_tool.csv` and `amazon_office.csv` under `data/amazon` directory
`pop.ipynb` has preprocessing steps and needs to be run before models, should be good to run on both datasets in amazon folder.


after, might need a few local folder creations/path changes to run code.
`sample.sh` has some examples for running and evaluating models