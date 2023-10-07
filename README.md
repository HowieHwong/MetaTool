<div align="center">
<img src="assets/MetaTool_icon.png" alt="示例图片" width="300" height="300">
</div>

# MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use

<p align="center">
   <a href="https://atlas.nomic.ai/map/a43a6a84-4453-428a-8738-2534d7bf0b89/b2b8134b-a37e-45d2-a0d9-765911f27df6" target="_blank">🌐 Dataset Website</a> | <a href="https://arxiv.org/abs/2310.03128" target="_blank">📃 Paper </a>
</p>

[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/HowieHwong/MetaTool/issues) 
[![Language Python](https://img.shields.io/badge/Language-Python-red.svg?style=flat-square)](https://github.com/HowieHwong/MetaTool) 
[![License MIT](https://img.shields.io/badge/Lisence-MIT-blue.svg?style=flat-square)](https://github.com/HowieHwong/MetaTool) 

We introduce **MetaTool**, a benchmark designed to evaluate whether LLMs have tool usage awareness and can correctly choose tools. It includes:

- **ToolE Dataset**: This dataset contains various types of user queries in the form of prompts that trigger LLMs to use tools, including both single-tool and multi-tool scenarios.
- **Various Tasks**: we set the tasks for both tool usage awareness and tool selection. 
- **Results on nine LLMs**: We conduct experiments involving nine popular LLMs and find that the majority of them still struggle to effectively select tools, highlighting the existing gaps between LLMs and genuine intelligent agents.



<div align="center">
<img src="assets/benchmark_architecture_00.jpg" alt="示例图片">
</div>


### Quick Start
####Install the packages:
```shell
pip install --upgrade pip
pip install -r requirements.txt
```

#### Download the models:
```shell
python src/generation/model_download.py
```

#### Generate the results:
```shell
sh src/generation/run.sh
```

