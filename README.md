based off [this](https://github.com/kang205/SASRec) pytorch SASRec implementation

keeping runs and model results local for now.

download `Tools and Home Improvement` and `Office Products` datasets from "Small" subsets for experimentation section of [here](https://nijianmo.github.io/amazon/index.html) and rename as `amazon_tool.csv` and `amazon_office.csv` under `data/amazon` directory
`pop.ipynb` has preprocessing steps and needs to be run before models, should be good to run on both datasets in amazon folder
after, `sample.sh` has an example for running our model, might need a few local folder creations/path changes