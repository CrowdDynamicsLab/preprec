# first follow previous steps from README
# train_dir is directory inside `res/{dataset}` folder where checkpoints/metrics are saved

# then to train on datasets
python3 main.py --dataset douban/douban_music --train_dir train_music --time_embed --monthpop wtembed --weekpop week_embed2
python3 main.py --dataset douban/douban_movie --train_dir train_movie --time_embed --monthpop wtembed --weekpop week_embed2
python3 main.py --dataset epinions/epinions --train_dir train_epinions --time_embed --monthpop wtembed --weekpop week_embed2
python3 main.py --dataset amazon/amazon_office --train_dir train_office --time_embed --monthpop wtembed --weekpop week_embed2
python3 main.py --dataset amazon/amazon_tool --train_dir train_tool --time_embed --monthpop wtembed --weekpop week_embed2

# then to evaluate trained models
python3 main.py --dataset douban/douban_music --train_dir test_music --state_dict_path res/douban/douban_music/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python3 main.py --dataset douban/douban_movie --train_dir test_movie --state_dict_path res/douban/douban_movie/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python3 main.py --dataset epinions/epinions --train_dir test_epinions --state_dict_path res/epinions/epinions/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python3 main.py --dataset amazon/amazon_office --train_dir test_office --state_dict_path res/amazon/amazon_office/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --inference_only
python3 main.py --dataset amazon/amazon_tool --train_dir test_tool --state_dict_path res/amazon/amazon_tool/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --inference_only

# example for zero-shot transfer to another dataset
python3 main.py --dataset douban/douban_music --train_dir movie_zs_music --state_dict_path res/douban/douban_movie/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --transfer --inference_only

# example for finetuning on subset (by # users) of another dataset
python3 main.py --dataset douban/douban_music --train_dir movie_fs_music --state_dict_path res/douban/douban_movie/train/best.pth --time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --fs_transfer --fs_num_epochs 5 --fs_prop 0.5
