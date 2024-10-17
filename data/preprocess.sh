# Preprocessing with desired parameters for each dataset evaluated in the paper

python3 data.py --dataset ./douban/douban_music --t1_cutoff 10 --t2_cutoff 2
python3 data_2.py --dataset ./douban/douban_music --t2_cutoff 2 --userneg

python3 data.py --dataset ./douban/douban_movie --t1_cutoff 10 --t2_cutoff 2
python3 data_2.py --dataset ./douban/douban_movie --t2_cutoff 2 --userneg

python3 data.py --dataset ./epinions/epinions --t1_cutoff 10 --t2_cutoff 2
python3 data_2.py --dataset ./epinions/epinions --t2_cutoff 2 --userneg

python3 data.py --dataset ./amazon/amazon_office --t1_cutoff 10 --t2_cutoff 2
python3 data_2.py --dataset ./amazon/amazon_office --t2_cutoff 2 --userneg

# omission of cutoff overrides is intentional
python3 data.py --dataset ./amazon/amazon_tool
python3 data_2.py --dataset ./amazon/amazon_tool --userneg