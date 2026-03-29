import transformers
import json
import pandas as pd
import argparse
import os
import wandb
import torch
from ir import IRCallback, CosineCallback, CosineSimilarityTracker, WeightCallback, GradientTracker
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback, BitsAndBytesConfig
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSequenceClassification
from utilities import load_config_file, get_2way_data, get_3way_data, compute_metrics, get_json_data
from peft.src.peft import LoraConfig, DoraConfig, BottleneckConfig
from peft.src.peft import get_peft_model, prepare_model_for_int8_training


class TestSetEvaluationCallback(TrainerCallback):
    def __init__(self, trainer, test_dataset, compute_metrics=None, main_args=None, file_name=None):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.trainer = trainer
        self.main_args = main_args
        self.file_name = file_name

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.main_args.logging_steps == 0 and state.global_step > 800:
            trainer = self.trainer
            logits = trainer.predict(test_dataset=self.test_dataset.remove_columns("label"))
            preds = logits.predictions.argmax(-1)

            if self.main_args.dataset_name == '2way':
                label_map = {0: "Incorrect", 1: "Correct"}
            elif self.main_args.dataset_name == '3way':
                label_map = {0: "Incorrect", 1: "Partially Correct", 2: "Correct"}
            else:
                raise ValueError(f"Unknown dataset: {self.main_args.dataset_name}")

            output_list = []

            for i, pred_idx in enumerate(preds):
                example = self.test_dataset[i]
                pred_label = label_map[pred_idx]

                output_dict = {
                    "id": example["id"],
                    "question_id": example["question_id"],
                    "score": pred_label
                }
                output_list.append(output_dict)


            with open(os.path.join(self.main_args.output_dir, f"{self.file_name}_step{state.global_step}.json"), "w") as f:
                json.dump(output_list, f, indent=2)

            print(f"Saved predictions for {len(output_list)} examples.")


def main():
    if args.dataset_name == '2way':
        file_path_trial = os.path.join(os.path.dirname(__file__), "data/2way/ALICE_LP_trial_2way__v2.json")
        file_path_train = os.path.join(os.path.dirname(__file__), "data/2way/ALICE_LP_train_2way__v2.json")
        file_path_test_answer = os.path.join(os.path.dirname(__file__), "data/2way/unlabelled_eval/2way_unseen_answers_eval.json")
        file_path_test_question = os.path.join(os.path.dirname(__file__), "data/2way/unlabelled_eval/2way_unseen_questions_eval.json")

        df_trial = get_json_data(file_path_trial)
        df_train = get_json_data(file_path_train)
        df_test_answer = get_json_data(file_path_test_answer)
        df_test_question = get_json_data(file_path_test_question)

        id_list_train, question_id_list_train, prompt_list_train, label_list_train = get_2way_data(df_train)
        id_list_trial, question_id_list_trial, prompt_list_trial, label_list_trial = get_2way_data(df_trial)
        id_list_test_answer, question_id_list_test_answer, prompt_list_test_answer, label_list_test_answer = get_2way_data(df_test_answer)
        id_list_test_question, question_id_list_test_question, prompt_list_test_question, label_list_test_question = get_2way_data(df_test_question)

        args.num_labels = 2

    elif args.dataset_name == '3way':
        file_path_trial = os.path.join(os.path.dirname(__file__), "data/3way/ALICE_LP_trial_3way__v2.json")
        file_path_train = os.path.join(os.path.dirname(__file__), "data/3way/ALICE_LP_train_3way__v2.json")
        file_path_test_answer = os.path.join(os.path.dirname(__file__), "data/3way/unlabelled_eval/3way_unseen_answers_eval.json")
        file_path_test_question = os.path.join(os.path.dirname(__file__), "data/3way/unlabelled_eval/3way_unseen_questions_eval.json")

        df_trial = get_json_data(file_path_trial)
        df_train = get_json_data(file_path_train)
        df_test_answer = get_json_data(file_path_test_answer)
        df_test_question = get_json_data(file_path_test_question)

        id_list_train, question_id_list_train, prompt_list_train, label_list_train = get_3way_data(df_train)
        id_list_trial, question_id_list_trial, prompt_list_trial, label_list_trial = get_3way_data(df_trial)
        id_list_test_answer, question_id_list_test_answer, prompt_list_test_answer, label_list_test_answer = get_3way_data(df_test_answer)
        id_list_test_question, question_id_list_test_question, prompt_list_test_question, label_list_test_question = get_3way_data(df_test_question)

        args.num_labels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            num_labels=args.num_labels
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            num_labels=args.num_labels
        )

    if args.model_name == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_int8_training(model)

    if args.adapter_name == "lora":
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.adapter_name == "dora":
        config = DoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=True,
            Wdecompose_target_modules=None
        )
    elif args.adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=256,
            non_linearity='tanh',
            adapter_dropout=0.0,
            use_parallel_adapter=True,
            use_adapterp=False,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            scaling=1.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        raise ValueError(f"Unknown adapter: {args.adapter_name}")

    model = get_peft_model(model, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    print(model)
    model.print_trainable_parameters()

    dataset_train = Dataset.from_dict({"id": id_list_train,
                                       "question_id": question_id_list_train,
                                       "prompt": prompt_list_train,
                                       "label": label_list_train}).shuffle(seed=171)

    dataset_val = Dataset.from_dict({"id": id_list_trial,
                                     "question_id": question_id_list_trial,
                                     "prompt": prompt_list_trial,
                                     "label": label_list_trial})

    dataset_test_answer = Dataset.from_dict({"id": id_list_test_answer,
                                            "question_id": question_id_list_test_answer,
                                            "prompt": prompt_list_test_answer,
                                            "label": label_list_test_answer})

    dataset_test_question = Dataset.from_dict({"id": id_list_test_question,
                                            "question_id": question_id_list_test_question,
                                            "prompt": prompt_list_test_question,
                                            "label": label_list_test_question})

    dataset = DatasetDict({'train': dataset_train, 'validation': dataset_val, 'test_answer': dataset_test_answer, 'test_question': dataset_test_question})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=args.max_length)


    encoded_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer, return_tensors="pt", padding="max_length", max_length=args.max_length)

    trainer_callbacks = []
    if args.tuning_method == "ir":
        print('IR Tuning activated')
        if args.importance_metric_name == "gradient":
            tracker = GradientTracker(model)
            ir_callback = IRCallback(model, encoded_dataset["train"].remove_columns(['id', 'question_id']), data_collator, tracker, args.batch_size, args.split_num)
        elif args.importance_metric_name == "cosine":
            tracker = CosineSimilarityTracker(model)
            ir_callback = CosineCallback(model, tracker, args.split_num)
        elif args.importance_metric_name == "weight":
            ir_callback = WeightCallback(model, args.split_num)
        else:
            raise NotImplemented
        trainer_callbacks.append(ir_callback)
    elif args.tuning_method == "full":
        print('Full Tuning activated')
        pass
    else:
        raise ValueError(f"Unknown tuning method: {args.tuning_method}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="wandb",
        max_steps=args.max_steps
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=data_collator,
        callbacks=trainer_callbacks,
        compute_metrics=compute_metrics,
    )

    test_callback_answer = TestSetEvaluationCallback(
        trainer=trainer,
        test_dataset=encoded_dataset["test_answer"],
        main_args=args,
        file_name="answer_prediction"
    )

    test_callback_question = TestSetEvaluationCallback(
        trainer=trainer,
        test_dataset=encoded_dataset["test_question"],
        main_args=args,
        file_name="question_prediction"
    )

    trainer.add_callback(test_callback_answer)
    trainer.add_callback(test_callback_question)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='llama3.1-8b', choices=['llama3-german', 'llama3.1-8b', 'llama2-13b', 'mistral-7b', 'deepseek-r1-distil-8b', 'qwen3-8b', 'phi-2'], help='model name')
    parser.add_argument('--dataset-name', default='3way', choices=['2way', '3way'], help='dataset name')
    parser.add_argument('--adapter-name', default='dora', choices=['lora', 'dora', 'bottleneck'], help='model adapter name')
    parser.add_argument('--importance-metric-name', default='gradient', choices=['gradient', 'weight', 'cosine'], help='importance score metric name')
    parser.add_argument('--tuning-method', default='full', choices=['ir', 'full'], help='layer-wise tuning method')
    parser.add_argument('--output-dir', default='debug', help='experiment directory')
    parser.add_argument('--use-instruction', default=True, action='store_true', help='whether to use instruction')
    parser.add_argument('--load-8bit', default=False, action='store_true', help='whether to load 8bit model')

    parser.add_argument('--gpu', type=str, default=0, help='default gpu id')
    parser.add_argument('--seed', type=int, default=171, help='seed number')
    parser.add_argument('--num-labels', type=int, default=5, help='number of labels')
    parser.add_argument('--max-length', type=int, default=512, help='max length of token sequence')

    parser.add_argument('--lora-r', type=int, default=32, help='lora rank')
    parser.add_argument('--lora-alpha', type=int, default=64, help='lora alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='lora dropout')

    parser.add_argument('--warmup-steps', type=int, default=100, help='warmup steps for learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--per-device-train-batch-size', type=int, default=16, help='device batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='gradient accumulation steps')

    parser.add_argument('--num-train-epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--max-steps', type=int, default=-1, help='number of maximum steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.00, help='weight decay')
    parser.add_argument('--logging-steps', type=int, default=100, help='log result every n steps')
    parser.add_argument('--save-strategy', type=str, default='no', help='save strategy')
    parser.add_argument('--save-steps', type=int, default=0, help='save model every n steps')
    parser.add_argument('--split-num', type=int, default=1, help='number of splits')

    args = parser.parse_args()
    config = load_config_file('config.yaml')
    huggingface_key = config['huggingface_key']
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)

    prefix = "ir-tuning-short-answer"
    if args.tuning_method == 'ir':
        wandb_name = f"{prefix}" + '-' + args.model_name + '-' + args.adapter_name + '-' + args.tuning_method + '-' + args.importance_metric_name + f'-split{args.split_num}'
    else:
        wandb_name = f"{prefix}" + '-' + args.model_name + '-' + args.adapter_name + '-' + args.tuning_method
    wandb_project = f"{prefix}" + '-' + args.dataset_name
    args.output_dir = os.path.join('experiments', wandb_project, wandb_name)

    os.environ["WANDB_NAME"] = wandb_name
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["HUGGING_FACE_HUB_TOKEN"] = huggingface_key
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.use_instruction:
        os.environ["WANDB_NAME"] += '-instruction'
        args.output_dir += '-instruction'

    if args.model_name == 'llama3.1-8b' and args.use_instruction:
        args.base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model_name == 'llama3.1-8b' and not args.use_instruction:
        args.base_model = 'meta-llama/Llama-3.1-8B'
    elif args.model_name == 'llama2-13b':
        args.base_model = 'meta-llama/Llama-2-13b-hf'
        args.batch_size = 8
        args.per_device_train_batch_size = 8
        args.logging_steps = 800
    elif args.model_name == 'mistral-7b' and args.use_instruction:
        args.base_model = 'mistralai/Mistral-7B-Instruct-v0.3'
    elif args.model_name == 'mistral-7b' and not args.use_instruction:
        args.base_model = 'mistralai/Mistral-7B-v0.3'
    elif args.model_name == 'deepseek-r1-distil-8b':
        args.base_model = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    elif args.model_name == 'qwen3-8b' and args.use_instruction:
        args.base_model = 'Qwen/Qwen3-8B'
    elif args.model_name == 'qwen3-8b' and not args.use_instruction:
        args.base_model = 'Qwen/Qwen3-8B'
    elif args.model_name == 'phi-2':
        args.base_model = 'microsoft/phi-2'
    elif args.model_name == 'llama3-german':
        args.base_model = "DiscoResearch/Llama3-German-8B"
    else:
        raise ValueError('Unknown model name')

    args.gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "parameter.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # sys.stdout = open(os.path.join(args.output_dir, "output.log"), "w")
    # sys.stderr = open(os.path.join(args.output_dir, "output.log"), "w")

    main()
    print("Task completed successfully!")