# Datasets

This README contains information about the datasets used in our RecSys 2024 paper to ensure reproducibility. This setup can also be used when testing our method on new datasets. 

A csv file must be supplied to run the `preprocess.sh` script for preprocessing, which is an offline, CPU-only step with no learning involved. For the five datasets in our paper, please refer [here](https://drive.google.com/drive/u/0/folders/1Jqyu615pUTnPkg0RFZ3MLQF7b8nCk-pE) for the CSVs. Note that these are filters of original datasets from:
1. [Amazon](https://nijianmo.github.io/amazon/index.html)
2. [Douban](https://www.dropbox.com/scl/fi/9zoykjl7km4wlrddscqrf/Douban.tar.gz?rlkey=i6w593rb3m8p8u13znp9mq1t3&e=2&dl=0)
3. [Epinions](https://snap.stanford.edu/data/soc-Epinions1.html)

Create a folder in the `data` folder titled each of these groups (`amazon`, `douban`, or `epinions`) and place the csv inside (refer to the `preprocess.sh` script for how it's called). Then run dataset-specific parts of `preprocess.sh`, navigate back to root, and check `sample.sh` for example training/evaluation scripts.

Note that `data_full.py` isn't used in `preprocess.sh`, it's there as an uncommented artifact of preprocessing variations, some of which are in ablations section of paper.
