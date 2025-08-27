import itertools

from tqdm.auto import tqdm
import pandas as pd


total_datapoints =
max_judge_queries =

def judge_query(p_r_pair):
    """
    Queries a prompt response pair to the judge LLM
    :param p_a_pair:
    :return:
    """

def data_reader(filename):
    """
    Reads the data from the json file into a pandas dataset
    :return:
    """
    return pd.read_json(filename, lines=True)

df = data_reader("helpsteer2.jsonl")
