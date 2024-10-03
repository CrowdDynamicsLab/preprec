# after previous steps in README have been followed, you must first set up negative samples for testing:
python main.py --dataset douban/douban_music --save_neg
python main.py --dataset douban/douban_movie --save_neg
python main.py --dataset epinions/epinions --save_neg
python main.py --dataset amazon/amazon_office --save_neg
python main.py --dataset amazon/amazon_tool --save_neg

# then to train on datasets
python main.py --dataset douban/douban_music --train_dir train --time_embed --monthpop wtembed --weekpop week_embed2
python main.py --dataset douban/douban_movie --train_dir train --time_embed --monthpop wtembed --weekpop week_embed2
python main.py --dataset epinions/epinions --train_dir train --time_embed --monthpop wtembed --weekpop week_embed2
python main.py --dataset amazon/amazon_office --train_dir train --time_embed --monthpop wtembed --weekpop week_embed2
python main.py --dataset amazon/amazon_tool --train_dir train --time_embed --monthpop wtembed --weekpop week_embed2

# then to evaluate trained models
python main.py --dataset douban/douban_music --train_dir test --state_dict_path res/douban/douban_music/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python main.py --dataset douban/douban_movie --train_dir test --state_dict_path res/douban/douban_movie/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python main.py --dataset epinions/epinions --train_dir test --state_dict_path res/epinions/epinions/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python main.py --dataset amazon/amazon_office --train_dir test --state_dict_path res/amazon/amazon_office/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python main.py --dataset amazon/amazon_tool --train_dir test --state_dict_path res/amazon/amazon_tool/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --inference_only

# example for zero-shot transfer to another dataset
python main.py --dataset douban/douban_music --train_dir movie_zs_test --state_dict_path res/douban/douban_movie/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --transfer --inference_only

# example for finetuning on subset (by # users) of another dataset
python main.py --dataset douban/douban_music --train_dir movie_fs_test --state_dict_path res/douban/douban_movie/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --fs_transfer --fs_num_epochs 5 --fs_prop 0.5
