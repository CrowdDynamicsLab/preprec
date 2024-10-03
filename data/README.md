# Datasets

This README contains information about the datasets used in our RecSys 2024 paper to ensure reproducibility. This setup can also be used when testing our method on new datasets. 

Prior to running the `./data.sh` script for preprocessing, you must supply a csv file. For the five datasets in our paper, please refer [here](https://drive.google.com/drive/u/0/folders/1Jqyu615pUTnPkg0RFZ3MLQF7b8nCk-pE). Note that these are filters of original datasets from:
1. [Amazon](https://nijianmo.github.io/amazon/index.html)
2. [Douban](https://www.dropbox.com/scl/fi/9zoykjl7km4wlrddscqrf/Douban.tar.gz?rlkey=i6w593rb3m8p8u13znp9mq1t3&e=2&dl=0)
3. [Epinions](https://snap.stanford.edu/data/soc-Epinions1.html)

Create a folder in the `data` folder titled each of these groups (`amazon`, `douban`, or `epinions`) and place the csv inside (refer to the `data.sh` script for how it's called).
