import itertools
import os

default_params = " --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --optim adamw_torch --output_dir tmp"


def main():
    models = ["bert-base-cased", "xlm-roberta-base", "roberta-base"]
    tasks = [
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "stsb",
        "wnli",
    ]

    for model, task in itertools.product(models, tasks):
        os.system(f"python train_no_reduce.py --model_name_or_path {model} --task_name {task}" + default_params)
        os.system(f"python train_smaller_vocab.py --model_name_or_path {model} --task_name {task}" + default_params)


if __name__ == "__main__":
    main()
