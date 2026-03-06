import argparse
import inspect
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


@dataclass
class TrainConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path: str = "dataset/hf_devops_chat"
    output_dir: str = "training/artifacts/tinyllama-devops"
    max_length: int = 512
    validation_split: float = 0.1
    seed: int = 42
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TinyLlama training config and Hugging Face Trainer."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/tinyllama_train_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--do-train",
        action="store_true",
        help="Run trainer.train() after initialization.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_config(config_path: Path) -> TrainConfig:
    cfg = TrainConfig()
    if not config_path.exists():
        return cfg

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    valid_fields = set(asdict(cfg).keys())
    unknown_keys = sorted(set(data.keys()) - valid_fields)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")

    for key, value in data.items():
        setattr(cfg, key, value)
    return cfg


def ensure_dataset_splits(
    dataset_obj: Dataset | DatasetDict,
    validation_split: float,
    seed: int,
) -> DatasetDict:
    if isinstance(dataset_obj, Dataset):
        if validation_split > 0:
            split = dataset_obj.train_test_split(test_size=validation_split, seed=seed)
            return DatasetDict({"train": split["train"], "validation": split["test"]})
        return DatasetDict({"train": dataset_obj})

    if "train" not in dataset_obj:
        raise ValueError("DatasetDict must contain a `train` split.")

    if "validation" in dataset_obj:
        return dataset_obj

    if "test" in dataset_obj:
        return DatasetDict({"train": dataset_obj["train"], "validation": dataset_obj["test"]})

    if validation_split > 0:
        split_train = dataset_obj["train"].train_test_split(test_size=validation_split, seed=seed)
        return DatasetDict({"train": split_train["train"], "validation": split_train["test"]})

    return DatasetDict({"train": dataset_obj["train"]})


def tokenize_or_validate(
    splits: DatasetDict,
    tokenizer: Any,
    max_length: int,
) -> DatasetDict:
    train_columns = splits["train"].column_names

    if "text" in train_columns:
        def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_token_type_ids=False,
            )

        tokenized = splits.map(
            tokenize_batch,
            batched=True,
            remove_columns=train_columns,
        )
    else:
        required = {"input_ids", "attention_mask"}
        if not required.issubset(set(train_columns)):
            raise ValueError(
                "Dataset needs either a `text` column or both `input_ids` and `attention_mask`."
            )
        tokenized = splits

    if "labels" not in tokenized["train"].column_names:
        tokenized = tokenized.map(
            lambda batch: {"labels": batch["input_ids"]},
            batched=True,
        )

    if "token_type_ids" in tokenized["train"].column_names:
        tokenized = tokenized.remove_columns(["token_type_ids"])

    return tokenized


def build_training_arguments(
    config: TrainConfig,
    has_eval: bool,
    output_dir: Path,
    train_size: int,
) -> TrainingArguments:
    effective_batch = (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch))
    total_steps = max(1, int(steps_per_epoch * config.num_train_epochs))
    warmup_steps = int(total_steps * config.warmup_ratio)

    args_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "seed": config.seed,
        "report_to": "none",
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "logging_strategy": "steps",
    }
    if warmup_steps > 0:
        args_kwargs["warmup_steps"] = warmup_steps

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if has_eval:
        args_kwargs["eval_steps"] = config.eval_steps
        if "eval_strategy" in ta_params:
            args_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in ta_params:
            args_kwargs["evaluation_strategy"] = "steps"
    else:
        if "eval_strategy" in ta_params:
            args_kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in ta_params:
            args_kwargs["evaluation_strategy"] = "no"

    return TrainingArguments(**args_kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
) -> Trainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": default_data_collator,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    set_seed(config.seed)

    dataset_path = resolve_path(project_root, config.dataset_path)
    output_dir = resolve_path(project_root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading dataset from: {dataset_path}")
    raw_dataset = load_from_disk(str(dataset_path))
    dataset_splits = ensure_dataset_splits(raw_dataset, config.validation_split, config.seed)
    tokenized_splits = tokenize_or_validate(dataset_splits, tokenizer, config.max_length)

    has_eval = "validation" in tokenized_splits
    train_dataset = tokenized_splits["train"]
    eval_dataset = tokenized_splits["validation"] if has_eval else None

    training_args = build_training_arguments(
        config,
        has_eval=has_eval,
        output_dir=output_dir,
        train_size=len(train_dataset),
    )
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\nTrainer initialized successfully.")
    print(f"train_size={len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval_size={len(eval_dataset)}")
    print(f"output_dir={output_dir}")

    if args.do_train:
        print("\nStarting training...")
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("Training complete. Model and tokenizer saved.")


if __name__ == "__main__":
    main()
