import re
import yaml
import logging
import os
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=handlers,
    )
    return logging.getLogger("quant_llm")


def extract_number(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K model response."""
    # Look for patterns like "#### 42" or "the answer is 42" or just trailing number
    patterns = [
        r"####\s*([\-\d,\.]+)",
        r"(?:the answer is|answer:|=)\s*([\-\d,\.]+)",
        r"([\-\d,\.]+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").strip()
    return None


def extract_yes_no(text: str) -> str | None:
    """Extract yes/no answer from a BoolQ model response."""
    text_lower = text.lower().strip()
    # Check first token / first word first
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in ("yes", "no"):
        return first_word
    if re.search(r"\byes\b", text_lower):
        return "yes"
    if re.search(r"\bno\b", text_lower):
        return "no"
    return None


def format_gsm8k_prompt(question: str, few_shot_examples: list[dict]) -> str:
    parts = []
    for ex in few_shot_examples:
        parts.append(
            f"Question: {ex['question']}\n"
            f"Let's think step by step.\n"
            f"{ex['answer']}\n"
        )
    parts.append(
        f"Question: {question}\n"
        f"Let's think step by step.\n"
    )
    return "\n".join(parts)


def format_boolq_prompt(passage: str, question: str) -> str:
    return (
        f"Passage: {passage}\n\n"
        f"Question: {question}\n"
        f"Answer (yes or no):"
    )


def format_piqa_prompt(goal: str, sol1: str, sol2: str) -> str:
    return (
        f"Goal: {goal}\n"
        f"Solution 1: {sol1}\n"
        f"Solution 2: {sol2}\n"
        f"Which solution is better? Answer 1 or 2:"
    )


def get_nested_attr(obj, attr_path: str):
    """Get nested attribute using dot-separated path."""
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def set_nested_attr(obj, attr_path: str, value):
    """Set nested attribute using dot-separated path."""
    parts = attr_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
