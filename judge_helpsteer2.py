from config import *
import copy
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


judged_score = "helpfulness"

system_prompt = """You are a fair judge tasked with providing clear and objective feedback based on specific criteria."""

score_rubrics_helpfulness = """Helpfulness:
Score 0: The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
Score 1: The response is borderline unhelpful and mostly does not capture what the user was looking for, but it is still usable and helpful in a small way.
Score 2: The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
Score 3: The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
Score 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for."""

score_rubrics_correctness = """Correctness:
Score 0: The response is completely incorrect. All information provided is wrong, false or hallucinated. If the prompt asks the assistant to do a task, the task is not at all attempted, or the wrong task was attempted in the response. The response is completely irrelevant to the prompt.
Score 1: The response has some correct elements but is mostly wrong or incomplete. The response may contain multiple instances of hallucinations, false information, misleading information, or irrelevant information. If the prompt asks the assistant to do a task, the task was attempted with a small amount of success.
Score 2: The response contains a mix of correct and incorrect information. The response may miss some details, contain misleading information, or minor hallucinations, but is more or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, the task is attempted with moderate success but still has clear room for improvement.
Score 3: The response is mostly accurate and correct with a small amount of missing information. It contains no misleading information or hallucinations. If the prompt asks the assistant to perform a task, the task is mostly successfully attempted.
Score 4: The response is completely correct and accurate to what is requested by the prompt with no necessary details missing and without false, misleading, or hallucinated information. If the prompt asks the assistant to do a task, the task is completely done and addressed in the response."""

score_rubrics_complexity = """Complexity:
Score 0: (Basic) – The response uses very easy to understand language that is clear and completely interpretable by children, adults, and anyone with a functional command of the language.
Score 1: (Simple) – The response uses relatively straightforward language and wording, but some schooling through elementary or a middle school in the language might be required to understand the response.
Score 2: (Intermediate) – People who have completed up through a high school education will probably be able to understand the vocabulary and sentence structure used, but those at the basic level or children might struggle to understand the response.
Score 3: (Advanced) – The response uses a fairly sophisticated vocabulary and terminology. Someone majoring in this subject at a college or university could have written it and would understand the response. An average adult who does not work or study in this area could not have written the response.
Score 4: (Expert) – An expert in the field or area could have written the response. It uses specific and technically relevant vocabulary. Elevated language that someone at the simple or basic level may not understand at all. The professional language of a lawyer, scientist, engineer, or doctor falls into this category."""

score_rubrics_verbosity = """Verbosity:
Score 0: (Succinct) – The response is short, to the point, and the most concise it can be. No additional information is provided outside of what is requested by the prompt (regardless of if the information or response itself is incorrect, hallucinated, or misleading. A response that gives an incorrect answer can still be succinct.).
Score 1: (Pretty Short) – The response is on the shorter side but could still have words, details, and/or text removed before it's at a bare minimum of what the response is trying to convey.
Score 2: (Average Length) – The response isn't especially long or short given what the prompt is asking of the model. The length is adequate for conveying a full response but isn't particularly wordy nor particularly concise.
Score 3: (Moderately Long) – The response is on the longer side but could still have more added to it before it is considered fully detailed or rambling.
Score 4: (Verbose) – The response is particularly lengthy, wordy, and/or extensive with extra details given what the prompt requested from the assistant model. The response can be verbose regardless of if the length is due to repetition and incoherency or if it is due to rich and insightful detail."""

default_score = -1


def get_main_prompt(task, response, score_rubrics):
  prompt = f"""###Task description:
An instruction (might include an input inside it), a response to evaluate, and a score rubric representing an evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing the feedback, write a score that is an integer between 0 and 4. You should refer to the score rubric.
3. The output should contain two lines in exactly this order:
- Feedback: Detailed feedback that assesses the quality of the response
- Rating: Score of the response
4. Please do not generate any other opening, closing, and explanations.

###Instruction to evaluate:
{task}

###Response to evaluate:
{response}

###Score rubrics:
{score_rubrics}

###Output:"""
  return prompt


def get_response(prompt):
  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}]
  try:
    chat_response = agent.chat.completions.create(
      model=agent_model,
      messages=messages,
      max_tokens=2048
    )
    response = chat_response.choices[0].message.content.strip()
    try:
      pos = response.find("Feedback:")
      if pos > -1:
        response = response[pos :]
      pos = response.rfind("Rating:")
      if pos > -1:
        score = int(response[pos + 7 :].strip())
      else:
        score = default_score
    except:
      score = default_score
  except:
    response = ""
    score = default_score
  return messages, response, score


def run_inference(task, response, true_score):
  prompt = get_main_prompt(task, response, score_rubrics)
  messages, output, score = get_response(prompt)

  return messages, output, score, []


def initialize_agent(model):
  global agent, agent_model, score_rubrics

  score_rubrics = eval("score_rubrics_%s" % judged_score)

  agent = OpenAI(
    api_key=LLAMA_CORP_CONFIG["api_key"],
    base_url=LLAMA_CORP_CONFIG["url"],
  )
  agent_model = model


def load_training_set():
  # load data
  from datasets import load_dataset
  D = load_dataset("nvidia/HelpSteer2")
  D = D["train"]
  n = D.shape[0]
  print("Number of prompts: %d" % D.shape[0])

  D = D.select(np.repeat(np.arange(0, n, 20), 30))  # 5% of data points, each repeated 10 times
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
      score = int(D[i][judged_score])
    except:
      score = default_score
    scores[i] = score

  return tasks, responses, scores, n


for model in ["gpt-4.1-nano", "llama-3-1-8b"]:
  for judged_score in ["helpfulness", "correctness", "complexity", "verbosity"]:
    filename = "helpsteer2_%s_%s.jsonl" % (judged_score, model)

    # load data
    tasks, responses, scores, n = load_training_set()

    # load model
    initialize_agent(model)

    with open("log.txt", "a") as f:
      print("Logging %s..." % filename, file=f)

    # inference
    start = time.time()
    num_parallel_jobs = 64
    output = Parallel(n_jobs=num_parallel_jobs, backend="threading") \
      (delayed(run_inference)(tasks[run], responses[run], scores[run]) for run in tqdm(range(n)))

    with open("log.txt", "a") as f:
      print("%.3f seconds" % (time.time() - start), file=f)

    predicted_scores = np.asarray([output[i][2] for i in range(n)])
    predicted_scores = np.reshape(predicted_scores, (n // 30, 30))
    print("Scores: ", end="")
    for i in range(min(n, 10)):
      vals = predicted_scores[i, :]
      vals = vals[vals != -1]
      print("%.3f (%.3f), " % (vals.mean(), vals.var()), end="")
    print()

    # evaluation
    with open(filename, "w") as f:
      for run in tqdm(range(n)):
        rationale = output[run][1]
        predicted_score = output[run][2]

        # log in json
        json_log = {"prompt": tasks[run],
          "response": responses[run],
          "human_score" : int(scores[run]),
          "rationale": rationale,
          "predicted_score": int(predicted_score)}
        f.write(json.dumps(json_log, default=str) + "\n")
