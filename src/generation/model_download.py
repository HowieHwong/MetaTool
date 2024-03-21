from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download Hugging Face model')
parser.add_argument('--model_path', type=str, help='Hugging Face model name (e.g., meta-llama/Llama-2-13b-chat)')
args = parser.parse_args()

load_dotenv()
import os
os.environ['CURL_CA_BUNDLE'] = ''
print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")
import time
import logging

logging.basicConfig(filename='download_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_with_retry(repo_id, max_retries=20, retry_interval=1):
    for attempt in range(max_retries):
        try:
            x = snapshot_download(repo_id=repo_id,local_files_only=False,resume_download=True)
            message = f"Download successful on attempt {attempt + 1}: {x}"
            logging.info(message)
            print(message)
            return x  # Download successful, return the result
        except Exception as e:
            message = f"Download failed on attempt {attempt + 1}: {e}"
            logging.error(message)
            print(message)
            print("Retrying in {} seconds...".format(retry_interval))
            time.sleep(retry_interval)
    
    return None  # Download failed after all retries



model_name = args.model_path
message = f"Starting download of model: {model_name}"
logging.info(message)
print(message,model_name)

x = download_with_retry(repo_id=model_name,)

if x is not None:
    message = "Download successful: {}".format(x)
    logging.info(message)
    print(message)
else:
    message = "Download failed after multiple attempts."
    logging.error(message)
    print(message)

message = "Download process completed."
logging.info(message)
print(message)

    
    