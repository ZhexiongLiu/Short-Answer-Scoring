import pandas as pd
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score
import yaml
import nltk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
nltk.download('punkt_tab')


def load_config_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def compute_metrics(pred):
    labels = pred.label_ids
    if labels is None or np.all(labels == -100): return {}
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    return {
        'Accuracy': acc,
        'P_Avg': avg_precision,
        'R_Avg': avg_recall,
        'F1_Avg': avg_f1,
        'WQK': qwk,
    }



def get_2way_data(data):
    id_list = []
    question_id_list = []
    prompt_list = []
    label_list = []
    ctrl_tokens_dict = {"Incorrect": 0, "Correct": 1}
    for line in data.to_dict(orient="records"):
        id = line["id"]
        question_id = line["question_id"]
        question = line["question"]
        incorrect_rubric = line["rubric"]["Incorrect"][0]
        partial_correct_rubric = line["rubric"]["Incorrect"][1]
        correct_rubric = line["rubric"]["Correct"]
        answer = line["answer"]

        prompt = f"""
        ## Frage: {question} 
        ## Falsch: {incorrect_rubric} 
        ## Falsch: {partial_correct_rubric} 
        ## Richtig: {correct_rubric} 
        ## Antwort: {answer}
        """

        id_list.append(id)
        question_id_list.append(question_id)
        prompt_list.append(prompt)

        if "score" in line:
            score = line["score"]
            label = ctrl_tokens_dict[score]
            label_list.append(label)
        else:
            label_list.append(-100)

    return id_list, question_id_list, prompt_list, label_list


def get_3way_data(data):
    id_list = []
    question_id_list = []
    prompt_list = []
    label_list = []
    ctrl_tokens_dict = {"Incorrect": 0, "Partially correct": 1, "Correct": 2}
    for line in data.to_dict(orient="records"):
        id = line["id"]
        question_id = line["question_id"]
        question = line["question"]
        incorrect_rubric = line["rubric"]["Incorrect"]
        partial_correct_rubric = line["rubric"]["Partially correct"]
        correct_rubric = line["rubric"]["Correct"]
        answer = line["answer"]
        prompt = f"""
        ## Frage: {question} 
        ## Falsch: {incorrect_rubric} 
        ## Teilweise Richtig: {partial_correct_rubric} 
        ## Richtig: {correct_rubric} 
        ## Antwort: {answer}
        """

        id_list.append(id)
        question_id_list.append(question_id)
        prompt_list.append(prompt)

        if "score" in line:
            score = line["score"]
            label = ctrl_tokens_dict[score]
            label_list.append(label)
        else:
            label_list.append(-100)


    return id_list, question_id_list, prompt_list, label_list

def get_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        df = pd.read_json(file_path)
    return df
