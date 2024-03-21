import os
import json
import pickle
import random

import argparse
from ..embedding import milvus_database
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db,
    MilvusClient
)
import numpy as np
import pandas as pd
random_seed = 48


random.seed(random_seed)

TOOL_REASON_PATH = 'src/prompt/prompt_template/tool_reason_prompt'
PREFIX_PROMPT_PATH = 'src/prompt/prompt_template/Action_prompt_single_tool'
PREFIX_PROMPT_PATH2 = 'src/prompt/prompt_template/Thought_prompt_single_tool'
TOOL_INFO_PATH = 'dataset/plugin_info.json'
SCENARIO_PATH = 'dataset/scenario'
CLEAN_DATA_PATH = 'dataset/data/all_clean_data.csv'
BIGTOOLDES_PATH = 'dataset/big_tool_des.json'
SAMLLTOOLDES_PATH = 'dataset/plugin_info.json'
DESCRIPTION_PATH = 'dataset/plugin_des.json'
MULTI_TOOL_GOLDEN = 'dataset/data/multi_tool_query_golden.json'



class PromptConstructor:
    def __init__(self, tool_reason_path=TOOL_REASON_PATH, prefix_prompt_path=PREFIX_PROMPT_PATH,
                 tool_info_path=TOOL_INFO_PATH, scenario_path=SCENARIO_PATH, clean_data_path=CLEAN_DATA_PATH,
                 description_path=DESCRIPTION_PATH, multi_tool_golden=MULTI_TOOL_GOLDEN):
        self.tool_reason_path = tool_reason_path
        self.prefix_prompt_path = prefix_prompt_path
        self.tool_info_path = tool_info_path
        self.scenario_path = scenario_path
        self.clean_data_path = clean_data_path
        self.description_path = description_path
        self.multi_tool_golden = multi_tool_golden

    @staticmethod
    def read_file(filename, readlines=False):
        with open(filename, 'r') as f:
            if readlines:
                return f.readlines()
            else:
                return f.read()

    @staticmethod
    def read_tool_info(filename):
        with open(filename, 'rb') as f:
            return json.load(f)

    def construct_single_prompt(self, query, des, prefix_file=PREFIX_PROMPT_PATH):
        prefix_prompt = self.read_file(filename=prefix_file)
        prefix_prompt = prefix_prompt.replace('''{user_query}''', query)
        prefix_prompt = prefix_prompt.replace('''{tool_list}''', des)
        return prefix_prompt

    def construct_thought_prompt(self, query, prefix_file=PREFIX_PROMPT_PATH2):
        tool_reason = self.read_file(self.tool_reason_path)
        prefix_prompt_path = self.read_file(filename=prefix_file)
        prefix_prompt_path = prefix_prompt_path.replace('''{user_query}''', query)
        prefix_prompt_path = prefix_prompt_path.replace('''{tool_reason}''', tool_reason)
        return prefix_prompt_path

    def get_scenario_tools(self, scenario):
        with open(os.path.join(self.scenario_path, scenario + '.json'), 'r') as f:
            data = json.load(f)["Tools"]
        return data

    def get_query_by_tool(self, tool, is_sample=True, sample_number=20, random_seed=48):
        if random_seed is not None:
            np.random.seed(random_seed)
        query_tool_data = pd.read_csv(self.clean_data_path)
        query_tool_data = query_tool_data[query_tool_data['Tool'] == tool]
        print(tool)
        print(len(query_tool_data['Query'].values))
        assert len(query_tool_data['Query'].values) > 0

        if len(query_tool_data['Query'].values) > sample_number and is_sample:
            query_tool_data = query_tool_data.sample(n=sample_number)
        return query_tool_data['Query'].values

    def get_tool_description(self, tool):
        all_big_tool_des = self.read_tool_info(filename=BIGTOOLDES_PATH)
        all_small_tool_des = self.read_tool_info(filename=SAMLLTOOLDES_PATH)
        try:
            # tool_des = all_big_tool_des[tool]
            if isinstance(tool, list):
                tool_des = [all_big_tool_des[el] for el in tool]
            else:
                tool_des = all_big_tool_des[tool]

        except:
            tool_des = [el['description_for_human'] for el in all_small_tool_des if el['name_for_model'] == tool]
            if tool_des == []:
                raise ValueError
        return tool_des

    def get_scenario_tool_description(self, scenario):
        tools = self.get_scenario_tools(scenario)
        string = ""
        for index, tool in enumerate(tools):
            string += f"{str(index + 1)}. tool name: {tool}, tool description: {self.get_tool_description(tool)}\n"
        return string

    def select_random_query_by_tool(self, dataframe, tool_value, random_seed=48):
        random.seed(random_seed)
        filtered_rows = dataframe[dataframe['Tool'] == tool_value]
        random_row = random.choice(filtered_rows.index)
        random_query_value = dataframe.loc[random_row, 'Query']
        return random_query_value

    def select_10_tools_with_exclusion(self, tool_list, excluded_tool, random_seed=48, ramdom_sample=10):
        available_tools = [tool for tool in tool_list if tool not in excluded_tool]
        if len(available_tools) < 10:
            return []
        random.seed(random_seed)
        selected_tools = random.sample(available_tools, ramdom_sample)
        return selected_tools

    def reliability_tool_selection(self):
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(name='tool_embedding', using='default')
        collection.load()
        all_tools = list(json.load(open(self.description_path, 'r')).keys())
        all_data = []
        index = 0
        for tool_item in all_tools:
            queries = self.get_query_by_tool(tool_item, sample_number=5)
            excluded_list = milvus_database.get_excluded_tool_list(tool_item)
            tools_list = self.select_10_tools_with_exclusion(all_tools, excluded_list)
            for query in queries:
                des = ""
                for index, el in enumerate(tools_list):
                    des += f"{str(index + 1)}. tool name: {el}, tool description: {self.get_tool_description(el)}\n"
                action_prompt = self.construct_single_prompt(query, des)
                thought_prompt = self.construct_thought_prompt(query)
                all_data.append({'action_prompt': action_prompt, 'thought_prompt': thought_prompt, 'tool': tool_item, 'query': query, 'index': index})
                index += 1
        json.dump(all_data, open('prompt_data/hallucination_prompt_new.json', 'w'))

    def get_10_most_sim(self, tool):
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(name='tool_embedding', using='default')
        collection.load()
        client = MilvusClient(url='http://localhost:19530')
        embedding = client.get(collection_name='tool_embedding', ids=[tool])[0]['embedding']
        top10_tools = milvus_database.search([embedding], limit_num=10)
        top10_tools = [el.to_dict()['id'] for el in top10_tools]
        print("Top 10 tools {}".format(top10_tools))
        return top10_tools

    def get_tool_list_des(self, tools: list):
        string = ""
        for index, tool in enumerate(tools):
            string += f"{str(index + 1)}. tool name: {tool}, tool description: {self.get_tool_description(tool)}\n"
        return string

    def similarity_pipeline(self):
        all_data = []
        with open(self.description_path, 'r') as f:
            index = 0
            data = json.load(f)
            for k, v in data.items():
                top10_tools = self.get_10_most_sim(k)
                querys = self.get_query_by_tool(k, sample_number=5)
                description = self.get_tool_list_des(top10_tools)
                for query in querys:
                    action_prompt = self.construct_single_prompt(query, description)
                    thought_prompt = self.construct_thought_prompt(query)
                    all_data.append(
                        {'action_prompt': action_prompt, 'thought_prompt': thought_prompt, 'tool': k,
                         'query': query, 'index': index})
                    index += 1
            with open('prompt_data/general_test.json', 'w') as f2:
                print(len(all_data))
                json.dump(all_data, f2)

    def scenario_pipeline(self, scenario):
        promptconstructor = PromptConstructor()
        scenario_tools = promptconstructor.get_scenario_tools(scenario)
        all_data = []
        if not os.path.exists('prompt_data/scenario'):
            os.mkdir('prompt_data/scenario')
        with open('prompt_data/scenario/{}.json'.format(scenario), 'w') as f:
            index = 0
            for tool in scenario_tools:
                query_list = promptconstructor.get_query_by_tool(tool)
                for query in query_list:
                    des = promptconstructor.get_scenario_tool_description(scenario)
                    action_prompt = self.construct_single_prompt(query, des)
                    thought_prompt = self.construct_thought_prompt(query)
                    all_data.append(
                        {'action_prompt': action_prompt, 'thought_prompt': thought_prompt, 'tool': tool,
                         'query': query, 'index': index})
                    index += 1
            json.dump(all_data, f)

    def combine_json(self, folder_path):
        file_list = os.listdir(folder_path)
        all_data = []
        for file in file_list:
            with open(os.path.join(folder_path, file), 'r') as f:
                data = json.load(f)
                for el in data:
                    el['scenario'] = file.split('.')[0]
                    all_data.append(el)
        with open('prompt_data/scenario.json', 'w') as f:
            json.dump(all_data, f)
        return all_data

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")

    def get_multi_tool_prompt(self, multi_tool_file):
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(name='tool_embedding', using='default')
        collection.load()
        print(json.load(open(self.multi_tool_golden, 'r')))
        all_tools = [el2['tool'] for el2 in json.load(open(self.multi_tool_golden, 'r'))]# for item in el2[:2]
        all_data = []
        with open(multi_tool_file, 'r') as f:
            data = json.load(f)
            for el in data:
                thought_prompt = self.construct_thought_prompt(el['query'])
                excluded_list_1 = milvus_database.get_excluded_tool_list(el['tool'][0])
                excluded_list_2 = milvus_database.get_excluded_tool_list(el['tool'][1])
                # merge excluded_list_1 and excluded_list_2
                excluded_list = excluded_list_1 + excluded_list_2
                tools_list = self.select_10_tools_with_exclusion(all_tools, excluded_list, ramdom_sample=8)
                tools_list.append(el['tool'][0])
                tools_list.append(el['tool'][1])
                # shuffle tool_list
                random.shuffle(tools_list)
                des = ""
                for index, el2 in enumerate(tools_list):
                    print(el2)
                    des += f"{str(index + 1)}. tool name: {el2}, tool description: {self.get_tool_description(el2)}\n"
                action_prompt = self.construct_single_prompt(el['query'], des, prefix_file='src/prompt/prompt_template/Action_prompt_multi_tool')
                all_data.append(
                    {'action_prompt': action_prompt, 'thought_prompt': thought_prompt, 'tool': el['tool'], 'query': el['query']}
                )
        with open('prompt_data/multi_tool_prompt.json', 'w') as f2:
            json.dump(all_data, f2)


def remove_tool_rows_and_save(input_filename, output_filename):
    data = pd.read_csv(input_filename)
    filtered_data = data[data['Tool'] != 'AIComprehensiveTool']
    filtered_data.to_csv(output_filename, index=False)


def run_task(task):
    prompt_construction = PromptConstructor()
    if task == 'all':
        prompt_construction.create_folder_if_not_exists('prompt_data')
        prompt_construction.get_multi_tool_prompt('dataset/data/multi_tool_query_golden.json')
        prompt_construction.reliability_tool_selection()
        prompt_construction.similarity_pipeline()
        file_list = os.listdir('dataset/scenario')
        for file in file_list:
            scenario = file.split('.')[0]
            prompt_construction.scenario_pipeline(scenario)
        prompt_construction.combine_json('prompt_data/scenario')
    elif task == 'similar':
        prompt_construction.similarity_pipeline()
    elif task == 'scenario':
        file_list = os.listdir('dataset/scenario')
        for file in file_list:
            scenario = file.split('.')[0]
            prompt_construction.scenario_pipeline(scenario)
        prompt_construction.combine_json('prompt_data/scenario')
    elif task == 'reliable':
        prompt_construction.reliability_tool_selection()
    elif task == 'multi':
        prompt_construction.get_multi_tool_prompt('dataset/data/multi_tool_query_golden.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a task to run.")
    parser.add_argument("task", choices=["similar", "scenario", "reliable", "multi", "all"], default='all',
                        help="Select a task to run.")
    args = parser.parse_args()
    run_task(args.task)
