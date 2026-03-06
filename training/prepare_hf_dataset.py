import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from datasets import Dataset, DatasetDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert chat JSON to a Hugging Face Dataset with prompt template text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/devops_dataset.json"),
        help="Path to source JSON dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/hf_devops_chat"),
        help="Directory to save Hugging Face dataset (`save_to_disk`).",
    )
    parser.add_argument(
        "--user-token",
        default="<|User|>",
        help="Template token for user turns.",
    )
    parser.add_argument(
        "--assistant-token",
        default="<|assistent|>",
        help="Template token for assistant turns.",
    )
    parser.add_argument(
        "--system-token",
        default="<|System|>",
        help="Template token for system turns.",
    )
    parser.add_argument(
        "--drop-system",
        action="store_true",
        help="If set, remove system turns from templated text.",
    )
    parser.add_argument(
        "--eval-size",
        type=float,
        default=0.0,
        help="Optional eval split ratio (for example: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/eval split.",
    )
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="If set, tokenize the `text` column and add `input_ids` and `attention_mask`.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="bert-base-uncased",
        help="Tokenizer model name or local path used when --tokenize is set.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Sequence length used for truncation and padding when tokenizing.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(data).__name__}.")
    return data


def render_chat_text(
    messages: List[Dict[str, Any]],
    role_tokens: Dict[str, str],
    drop_system: bool = False,
) -> str:
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        if drop_system and role == "system":
            continue
        if role not in role_tokens:
            continue

        content = str(msg.get("content", "")).strip()
        if not content:
            continue

        lines.append(f"{role_tokens[role]}\n{content}")

    return "\n".join(lines).strip()


def convert_to_dataset(
    raw_records: List[Dict[str, Any]],
    role_tokens: Dict[str, str],
    drop_system: bool = False,
) -> Dataset:
    converted: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Record index {idx} missing valid `messages` list.")

        text = render_chat_text(messages, role_tokens=role_tokens, drop_system=drop_system)
        row = dict(record)
        row["text"] = text
        converted.append(row)

    return Dataset.from_list(converted)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def tokenize_dataset(
    dataset_obj: Dataset | DatasetDict,
    tokenizer_name: str,
    max_length: int,
) -> Dataset | DatasetDict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ensure_pad_token(tokenizer)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_token_type_ids=False,
        )

    return dataset_obj.map(tokenize_batch, batched=True)


def main() -> None:
    args = parse_args()
    role_tokens = {
        "system": args.system_token,
        "user": args.user_token,
        "assistant": args.assistant_token,
    }

    raw_records = load_json_records(args.input)
    dataset = convert_to_dataset(
        raw_records=raw_records,
        role_tokens=role_tokens,
        drop_system=args.drop_system,
    )

    output_obj: Dataset | DatasetDict
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.eval_size and args.eval_size > 0:
        output_obj = dataset.train_test_split(test_size=args.eval_size, seed=args.seed)
    else:
        output_obj = dataset

    if args.tokenize:
        output_obj = tokenize_dataset(
            dataset_obj=output_obj,
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
        )

    output_obj.save_to_disk(str(args.output))
    print(f"Saved dataset object to: {args.output}")
    print(output_obj)

    print("\nExample templated sample:\n")
    print(dataset[0]["text"])
    if args.tokenize:
        if isinstance(output_obj, DatasetDict):
            sample = output_obj["train"][0]
        else:
            sample = output_obj[0]
        print("\nTokenized sample keys:")
        print([k for k in sample.keys() if k in {"input_ids", "attention_mask"}])
        print("input_ids[:20] =", sample["input_ids"][:20])
        print("attention_mask[:20] =", sample["attention_mask"][:20])


if __name__ == "__main__":
    main()
