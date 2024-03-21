#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -m|--model-path <model_path> -t|--task <task>"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model-path)
        model_path="$2"
        shift # past argument
        shift # past value
        ;;
        -t|--task)
        task="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        usage
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$model_path" ] || [ -z "$task" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Add sys path
src_path="$(pwd)/src"
export PYTHONPATH="$PYTHONPATH:$src_path"

# Run model download script
python -m src.generation.model_download --model_path "$model_path"

wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
bash standalone_embed.sh start

sudo docker ps -a

# Run embedding script
python -m src.embedding.milvus_database

# Run prompt construction script
python -m src.prompt.prompt_construction "$task"
