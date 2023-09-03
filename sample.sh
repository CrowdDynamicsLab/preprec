# examples for training model from scratch
python -u main.py --dataset amazon/amazon_office --model newrec --train_dir newrec_base
python -u main.py --dataset amazon/amazon_office --model sasrec --train_dir sasrec
python -u main.py --dataset amazon/amazon_office --model bert4rec --mask_prob 0.5 --train_dir bert4rec

# example for evaluating trained model, corresponds to first model trained above
python -u main.py --dataset amazon/amazon_office --model newrec --state_dict_path res/amazon/amazon_office/newrec_base/newrec.epoch=201.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth --inference_only --mode test --topk 10 5 1

# example for evaluating trained model by groups, corresponds to above
python -u main.py --dataset amazon/amazon_office --model newrec --state_dict_path res/amazon/amazon_office/newrec_base/newrec.epoch=201.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth --inference_only --mode test --topk 10 --eval_quality --quality_size 25

# example for transferring to another dataset, corresponds to above
python -u main.py --dataset amazon/amazon_tool --transfer --model newrec --state_dict_path res/amazon/amazon_office/newrec_base/newrec.epoch=201.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth --inference_only --mode test --topk 10 5 1