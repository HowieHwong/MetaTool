"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import os,time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import requests
from fastchat.model import load_model, get_conversation_template, add_model_args
import logging
import argparse
import json
from tqdm import tqdm
from utils import *
from dotenv import load_dotenv
import openai
import traceback

os.environ['OPENAI_API_KEY'] = '<Your API Key>'
openai.api_key = '<Your API Key>'


load_dotenv()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                         
def generation(input,tokenizer,model):
    msg = input
    conv = get_conversation_template(args.model_path)
    conv.set_system_message('')
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(args.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs



def tool_test_thought(model_name, model, tokenizer):
    model_name = model_name.lower()
    # 判断文件夹是否存在
    assert os.path.exists('tool/new_test_res/{}'.format(model_name))
    file_list = os.listdir('tool/test_data')
        
    for file in file_list:
        if file.endswith('.json'):
            if os.path.isfile(os.path.join('tool/test_data', file)):
                all_data = []
                with open(os.path.join('tool/test_data', file), 'r') as f:
                    data = json.load(f)
                    for el in data:
                        res = generation(model=model, tokenizer=tokenizer, input=el['thought_prompt'])
                        print(res)
                        el['res'] = res
                        all_data.append(el)
                        save_json(all_data, os.path.join('tool/new_test_res/{}'.format(model_name), file))


def tool_test_action(model_name, model, tokenizer):
    model_name = model_name.lower()
    file_list = os.listdir('tool/test_data/{}'.format(model_name))
    for file in file_list:
        #if file.endswith('general_test.json'):
        if 1:
            all_data = []
            with open(os.path.join('tool/test_data', model_name, file), 'r') as f:
                data = json.load(f)
                for el in data:
                    res = generation(model=model, tokenizer=tokenizer, input=el['action_prompt'])
                    print(res)
                    el['action_res'] = res
                    all_data.append(el)
                    save_json(all_data, os.path.join('tool/new_test_res/{}'.format(model_name), file))



    

def get_res_chatgpt(string, gpt_model):
    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "user",
             "content": string}
        ]
    )
    print(completion.choices[0].message['content'])
    return completion.choices[0].message['content']


def run_single_test(args):
    model_mapping = {"baichuan-inc/Baichuan-13B-Chat": "baichuan-13b",
                     "baichuan-inc/Baichuan2-13B-chat": "baichuan2-13b",  
                     "THUDM/chatglm2-6b": "chatglm2",
                     "lmsys/vicuna-13b-v1.3": "vicuna-13b",
                     "lmsys/vicuna-7b-v1.3": "vicuna-7b",
                     "lmsys/vicuna-33b-v1.3": "vicuna-33b",
                     "meta-llama/Llama-2-7b-chat-hf": "llama2-7b",
                     "meta-llama/Llama-2-13b-chat-hf": "llama2-13b",
                     'TheBloke/koala-13B-HF': "koala-13b",
                     "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "oasst-12b",
                     "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b",
                     'ernie': "ernie",
                     "chatgpt": 'chatgpt',
                     'gpt-4': 'gpt-4'}
    if args.model_path != 'ernie' and args.model_path != 'chatgpt' and args.model_path != 'gpt-4':
        model, tokenizer = load_model(
            args.model_path,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            revision=args.revision,
            debug=args.debug,
        )
    else:
        model = None
        tokenizer = None
    test_type = args.test_type
    model_name=model_mapping[args.model_path]
    print(test_type)
    if test_type == 'tool_test_thought':
        tool_test_thought(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'tool_test_action':
        tool_test_action(model_name=model_name, model=model, tokenizer=tokenizer)
    else:
        print("Invalid test_type. Please provide a valid test_type.")
        return None
    return "OK"





@torch.inference_mode()
def main(args,max_retries = 20,retry_interval = 3):


    for attempt in range(max_retries):
        try:
            state = run_single_test(args)
            message = f"Test function successful on attempt {attempt + 1}"
            logging.info(message)
            print(message)
            return state  
        except Exception as e:
            traceback.print_exc()
            message = f"Test function failed on attempt {attempt + 1}:{e}"
            logging.error(message)
            print(message)
            print("Retrying in {} seconds...".format(retry_interval))
            time.sleep(retry_interval)

    return None  




# Generate a unique timestamp for the log file
timestamp = time.strftime("%Y%m%d%H%M%S")
log_filename = f"test_log_{timestamp}.txt"


logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--test_type", type=str, default='tool')
    args = parser.parse_args()
    state = main(args,)
    print(state)
    
    

