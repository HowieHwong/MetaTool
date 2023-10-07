export CUDA_VISIBLE_DEVICES=0
python run.py --test_type plugin_test_action --model_path 'baichuan-inc/Baichuan2-13B-chat' --temperature 0.0 --num_gpus 1
python run.py --test_type plugin_test_thought --model_path 'baichuan-inc/Baichuan2-13B-chat' --temperature 0.0 --num_gpus 1