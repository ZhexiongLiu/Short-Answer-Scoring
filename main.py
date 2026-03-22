import argparse
import asyncio
import logging
import os
import yaml
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from utilities import get_json_data


def parse_score(raw_score):
    if 'rationale' in raw_score.lower():
        raw_score = raw_score.split("rationale")[0].strip()

    if 'incorrect' in raw_score.lower():
        return 0
    elif 'correct' in raw_score.lower():
        return 1
    else:
        logging.info(f"Unrecognized score format: {raw_score}")
        return 0


def evaluate_with_sklearn(score, pred_score):
    precision = precision_score(score, pred_score, average="macro", zero_division=0)
    recall = recall_score(score, pred_score, average="macro", zero_division=0)
    f1 = f1_score(score, pred_score, average="macro", zero_division=0)
    wqk = cohen_kappa_score(score, pred_score, weights="quadratic")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "wqk": wqk,
    }


class BatchGeneration:
    def __init__(self, args):

        self.args = args

        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
        api_key = config["openai_api_key"]

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-06-01",
            azure_endpoint="https://erevise-api.openai.azure.com/"
        )

        if self.args.prompt_template == "baseline":
            prompt_file = os.path.join(os.path.dirname(__file__), f"prompts/baseline.md")
        elif self.args.prompt_template == "rationale":
            prompt_file = os.path.join(os.path.dirname(__file__), f"prompts/rationale.md")
        else:
            raise ValueError(f"Unsupported prompt template: {self.args.prompt_template}")

        with open(prompt_file, "r", encoding="utf-8") as reader:
            self.prompt_template = reader.read()

    def build_prompt(self, question: str, rubric: str, answer: str):
        return self.prompt_template.format(question=question, rubric=rubric, answer=answer)

    async def _one_completion(self, prompt: str, sem: asyncio.Semaphore):

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a helpful STEAM teacher."
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=prompt
            ),
        ]

        async with sem:
            try:
                if self.args.model.startswith("gpt-5"):
                    resp = await self.client.chat.completions.create(
                        model=self.args.model,
                        messages=messages,
                        temperature=self.args.temperature,
                    )
                else:
                    resp = await self.client.chat.completions.create(
                        model=self.args.model,
                        messages=messages,
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                    )
                return resp.choices[0].message.content
            except Exception as e:
                logging.info(f"OpenAI request failed: {e}")
                return "0"

    def chunked(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    async def run_in_batches(self, prompts):
        sem = asyncio.Semaphore(self.args.max_concurrent)
        results = []

        total = len(prompts)
        batch_iter = list(self.chunked(prompts, self.args.batch_size))

        for b_idx, batch in enumerate(batch_iter, start=1):
            tasks = [self._one_completion(p, sem) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)

            if self.args.show_progress:
                done = min(b_idx * self.args.batch_size, total)
                logging.info(f"Finished batch {b_idx}/{len(batch_iter)} — {done}/{total} prompts")

        return results

    async def generate(self, prompts):
        try:
            tasks = self.run_in_batches(prompts)
            results = await asyncio.gather(tasks)
            return results[0]

        finally:
            pass

    def run(self, prompts):
        results = asyncio.run(self.generate(prompts))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1", type=str)
    parser.add_argument("--dataset", default="2way", type=str)
    parser.add_argument("--output_path", default="experiments/debug", type=str)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--max_concurrent", default=120, type=int)
    parser.add_argument("--prompt_template", default="baseline", type=str)
    parser.add_argument("--show_progress", default=True, type=bool)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
    logging.info("Running async evaluator with batching")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.dataset == "2way":
        file_path_trial = os.path.join(os.path.dirname(__file__), "data/2way/ALICE_LP_trial_2way__v2.json")
        file_path_train = os.path.join(os.path.dirname(__file__), "data/2way/ALICE_LP_train_2way__v2.json")
    elif args.dataset == "3way":
        file_path_trial = os.path.join(os.path.dirname(__file__), "data/3way/ALICE_LP_trial_3way__v2.json")
        file_path_train = os.path.join(os.path.dirname(__file__), "data/3way/ALICE_LP_train_3way__v2.json")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    df_trial = get_json_data(file_path_trial)
    df_train = get_json_data(file_path_train)

    questions = df_trial["question"].tolist()
    answers = df_trial["answer"].tolist()
    rubrics = df_trial["rubric"].tolist()
    gold_scores = df_trial["score"].tolist()

    evaluator = BatchGeneration(args)
    prompts = [evaluator.build_prompt(q, r, a) for q, r, a in zip(questions, rubrics, answers)]
    pred_scores = evaluator.run(prompts)

    df_trial["pred_score"] = pred_scores
    print(df_trial[["score", "pred_score"]])
    df_trial.to_csv(os.path.join(args.output_path, f"results.csv"), index=False)

    parsed_gold_scores = [parse_score(s) for s in gold_scores]
    parsed_pred_scores = [parse_score(s) for s in pred_scores]

    metrics = evaluate_with_sklearn(parsed_gold_scores, parsed_pred_scores)
    print(
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"WQK: {metrics['wqk']:.4f}"
    )
