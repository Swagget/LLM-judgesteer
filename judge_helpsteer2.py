from config import *
import itertools
import joblib
from joblib import Parallel, delayed
import json
import numpy as np
from openai import OpenAI
import os
import sys
import time
from tqdm.auto import tqdm

import platform
print("python %s" % platform.python_version())
print("%d joblib CPUs" % joblib.cpu_count())


system_prompt = """You are a fair judge assistant tasked with providing clear
and objective feedback based on specific criteria, ensuring each assessment
reflects the absolute standards set for performance."""

main_prompt = """Task Description: An instruction (might include an input
inside it), a response to evaluate, and a score rubric representing evaluation
criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly
based on the given score rubric, not evaluating in general.
2. After writing the feedback, write a score that is an integer between
{min_score} and {max_score}. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria)
[RESULT] (an integer number between {min_score} and {max_score})".
4. Please do not generate any other opening, closing, and explanations.

Instruction: %s

Response: %s

Score Rubrics: %s

Feedback:"""

score_rubrics = """Helpfulness can be measured by how useful and helpful the
overall response is. While giving score, you can refer the following scoring
rubrics. You can only give a single value for the resulting score.
Score of 0: The response is not useful or helpful at all. The response
completely missed the essence of what the user wanted.
Score of 1: The response is borderline unhelpful and mostly does not capture
what the user was looking for, but is still usable and helpful in a small way.
Score of 2: The response is partially helpful but misses the overall goal of
the user's query/input in some way. The response did not fully satisfy what the
user was looking for.
Score of 3: The response is mostly helpful and mainly aligned with what the
user was looking for, but there is still some room for improvement.
Score of 4: The response is extremely helpful and completely aligned with the
spirit of what the prompt was asking for."""

default_score = 2


def get_response(prompt):
  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}]
  try:
    chat_response = agent.chat.completions.create(
      model=agent_model,
      messages=messages,
      max_tokens=1024)
    response = chat_response.choices[0].message.content
    try:
      pos = response.rfind("[RESULT]")
      if pos > -1:
        score = int(response[pos + 8 :].strip())
      else:
        score = default_score
    except:
      score = default_score
  except:
    response = ""
    score = default_score
  return messages, response, score


def run_inference(task, response, true_score):
  prompt = main_prompt % (task, response, score_rubrics)
  messages, response, score = get_response(prompt)

  return messages, response, score


def initialize_agent():
  global agent, agent_model

  agent = OpenAI(
    api_key=LLAMA_CONFIG["api_key"],
    base_url=LLAMA_CONFIG["url"],
  )
  agent_model = LLAMA_CONFIG["model"]


def load_training_set():
  # load data
  from datasets import load_dataset
  D = load_dataset("nvidia/HelpSteer2")
  D = D["train"]
  n = D.shape[0]
  print("Number of prompts: %d" % D.shape[0])

  D = D.select(np.repeat(np.arange(0, n, 10), 10))  # 10% of data points, each repeated 10 times
  # This step could be a bit better randomized to avoid more errors
  # this seems like a passive approach right now. We need to find an active approach that chooses dynamically rather than repeating 10 time what's going on.
  n = D.shape[0]
  print("Number of chosen prompts: %d" % D.shape[0])

  # parse data
  tasks = []
  responses = []
  scores = np.zeros(n, dtype=int)
  for i in range(n):
    tasks.append(D[i]["prompt"])
    responses.append(D[i]["response"])
    try:
      score = int(D[i]["helpfulness"])
    except:
      score = default_score
    scores[i] = score

  return tasks, responses, scores, n


# load data
tasks, responses, scores, n = load_training_set()

# load model
initialize_agent()

num_parallel_jobs = 32

# inference
start = time.time()
output = Parallel(n_jobs=num_parallel_jobs, backend="threading") \
  (delayed(run_inference)(tasks[run], responses[run], scores[run]) for run in tqdm(range(n)))
print("%.3f seconds" % (time.time() - start))

# evaluation
with open("helpsteer2.jsonl", "w") as f:
  for run in range(n):
    rationale = output[run][1]
    predicted_score = output[run][2]

    # log in json
    json_log = {"prompt": tasks[run],
      "response": responses[run],
      "human_score" : int(scores[run]),
      "rationale": rationale,
      "predicted_score": int(predicted_score)}
    f.write(json.dumps(json_log, default=str) + "\n")
