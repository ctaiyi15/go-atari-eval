
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm
from peft import PeftModel
from sgfmill import sgf, sgf_moves
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = "YichuanMa/LoGos-7B"

ADAPTER_PATH = str(ROOT_DIR / "outputs/logo-qlora/final")

INPUT_DIR = ROOT_DIR / "go-eval-positions/sente-test"
OUTPUT_FILE = ROOT_DIR / "outputs/decision.json"

MAX_INPUT_LENGTH = 3072
MAX_NEW_TOKENS = 16
MAX_GENERATION_SECONDS = 60

COLUMNS = "ABCDEFGHJKLMNOPQRST"

INSTRUCTION = (
    "Given the following Go game state and candidate move, "
    "identify the Go term that describes the candidate move "
    "and provide commentary explaining the move."
)

def coordinate(row, col):
    return f"{COLUMNS[col]}{row + 1}"

def load_position(path):
    game = sgf.Sgf_game.from_bytes(path.read_bytes())
    board, moves = sgf_moves.get_setup_and_moves(game)

    if board.side != 19:
        raise ValueError("Expected a 19x19 board")

    history = []
    last_color = None

    for index, (color, move) in enumerate(moves, start=1):
        player = "X" if color == "b" else "O"
        last_color = color

        if move is None:
            history.append(f"{index}.{player}-PASS")
            continue

        row, col = move
        vertex = coordinate(row, col)

        history.append(f"{index}.{player}-{vertex}")
        board.play(row, col, color)

    board_rows = []

    for row in range(18, -1, -1):
        values = []

        for col in range(19):
            stone = board.get(row, col)

            if stone == "b":
                values.append("1")
            elif stone == "w":
                values.append("-1")
            else:
                values.append("0")

        board_rows.append(" ".join(values))

    next_player = (
        "O" if last_color == "b"
        else "X"
    )

    return {
        "move_history": " ".join(history),
        "board_state": "\n".join(board_rows),
        "history_length": len(moves),
        "next_player": next_player,
    }

def build_prompt(position, candidate_move):
    formatted_move = (
        f"{position['next_player']}-{candidate_move}"
    )

    example_input = (
        INSTRUCTION
        + "\n\nMove history:\n"
        + position["move_history"]
        + "\n\nBoard state:\n"
        + position["board_state"]
        + "\n\nCandidate move:\n"
        + formatted_move
    )

    return (
        "### Instruction:\n"
        + example_input
        + "\n\n### Response:\n"
    )

def extract_term(output):
    match = re.search(
        r"(?im)^\s*Term\s*:\s*\*?([^\n\r]+)",
        output,
    )

    if match is None:
        return None

    term = match.group(1).strip()
    term = term.rstrip(".:;, ")

    return term if term else None

def normalize_term(term):
    if term is None:
        return None

    return re.sub(
        r"\s+",
        " ",
        term.strip().lower(),
    )

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    return tokenizer, model

@torch.inference_mode()
def generate(tokenizer, model, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        max_time=MAX_GENERATION_SECONDS,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_tokens = outputs[0, input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return generated_text.strip(), input_length

def evaluate_example(example):
    metrics = {}

    all_rows = []

    for position_name in ("X", "Y"):
        rows = example["predictions"][position_name]
        all_rows.extend(rows)

        expected_count = len(example["candidate_moves"])

        if len(rows) < expected_count:
            status = "incomplete"
        elif any(row["error"] for row in rows):
            status = "generation_error"
        elif any(
            row["generated_term"] is None
            for row in rows
        ):
            status = "missing_term"
        else:
            status = "complete"

        atari_candidates = [
            row["candidate"]
            for row in rows
            if normalize_term(row["generated_term"]) == "atari"
        ]

        metrics[position_name] = {
            "status": status,
            "total_predictions": len(rows),
            "valid_terms": sum(
                row["generated_term"] is not None
                for row in rows
            ),
            "atari_candidates": atari_candidates,
            "atari_count": len(atari_candidates),
        }

    valid = sum(
        row["generated_term"] is not None
        for row in all_rows
    )

    metrics["overall"] = {
        "total_predictions": len(all_rows),
        "valid_terms": valid,
        "term_generation_rate": (
            valid / len(all_rows)
            if all_rows else None
        ),
        "atari_predictions": sum(
            normalize_term(row["generated_term"]) == "atari"
            for row in all_rows
        ),
    }

    return metrics

def evaluate_dataset(results):
    total_predictions = 0
    valid_terms = 0
    atari_predictions = 0

    completed_examples = 0
    completed_positions = 0
    total_positions = 0

    term_counts = {}

    a_x_atari = 0
    a_y_atari = 0
    paired_a_count = 0
    a_x_only_atari = 0

    for example in results["examples"]:
        example["metrics"] = evaluate_example(example)

        if all(
            example["metrics"][p]["status"] == "complete"
            for p in ("X", "Y")
        ):
            completed_examples += 1

        for position_name in ("X", "Y"):
            total_positions += 1

            if (
                example["metrics"][position_name]["status"]
                == "complete"
            ):
                completed_positions += 1

            for row in example["predictions"][position_name]:
                total_predictions += 1

                term = normalize_term(row["generated_term"])

                if term is not None:
                    valid_terms += 1
                    term_counts[term] = (
                        term_counts.get(term, 0) + 1
                    )

                if term == "atari":
                    atari_predictions += 1

        x_a = next(
            (
                row for row in example["predictions"]["X"]
                if row["candidate"] == "A"
            ),
            None,
        )

        y_a = next(
            (
                row for row in example["predictions"]["Y"]
                if row["candidate"] == "A"
            ),
            None,
        )

        if (
            x_a is not None
            and y_a is not None
            and x_a["generated_term"] is not None
            and y_a["generated_term"] is not None
        ):
            paired_a_count += 1

            x_is_atari = (
                normalize_term(x_a["generated_term"])
                == "atari"
            )

            y_is_atari = (
                normalize_term(y_a["generated_term"])
                == "atari"
            )

            a_x_atari += int(x_is_atari)
            a_y_atari += int(y_is_atari)

            if x_is_atari and not y_is_atari:
                a_x_only_atari += 1

    return {
        "total_examples": len(results["examples"]),
        "completed_examples": completed_examples,
        "completed_positions": completed_positions,
        "total_positions": total_positions,
        "total_predictions": total_predictions,
        "valid_terms": valid_terms,
        "term_generation_rate": (
            valid_terms / total_predictions
            if total_predictions else None
        ),
        "atari_predictions": atari_predictions,
        "term_counts": dict(
            sorted(
                term_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ),
        "paired_A": {
            "valid_pairs": paired_a_count,
            "X_A_atari": a_x_atari,
            "Y_A_atari": a_y_atari,
            "X_A_atari_rate": (
                a_x_atari / paired_a_count
                if paired_a_count else None
            ),
            "Y_A_atari_rate": (
                a_y_atari / paired_a_count
                if paired_a_count else None
            ),
            "X_atari_Y_not_atari": a_x_only_atari,
            "X_atari_Y_not_atari_rate": (
                a_x_only_atari / paired_a_count
                if paired_a_count else None
            ),
        },
    }

def save_results(results):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT_FILE.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(OUTPUT_FILE)

def prepare_example(sample_path):
    folder = sample_path.parent

    sample = json.loads(
        sample_path.read_text(encoding="utf-8")
    )

    moves = sample["moves"]

    if not all(k in moves for k in ("A", "B", "C")):
        raise ValueError("Missing A, B, or C candidate")

    positions = {
        "X": load_position(folder / "X.sgf"),
        "Y": load_position(folder / "Y.sgf"),
    }

    expected_length = sample["move_number"]

    for name, position in positions.items():
        if position["history_length"] != expected_length:
            raise ValueError(
                f"{name}: history has "
                f"{position['history_length']} moves, "
                f"expected {expected_length}"
            )

    for name in ("X", "Y"):
        for candidate in ("A", "B", "C"):
            sample["results"][name]["winrate"][candidate]

    return sample, positions

def main():
    sample_files = sorted(
        INPUT_DIR.rglob("sample.json")
    )

    if not sample_files:
        raise FileNotFoundError(
            f"No sample.json files found in {INPUT_DIR}"
        )

    if OUTPUT_FILE.exists():

        results = json.loads(
            OUTPUT_FILE.read_text(encoding="utf-8")
        )

        if (
            results.get("adapter_path") != ADAPTER_PATH
            or results.get("base_model") != BASE_MODEL
            or results.get("decoding") != "greedy"
            or results.get("instruction") != INSTRUCTION
        ):
            raise ValueError(
                "Existing output uses different model, "
                "decoding, or prompt settings. "
                "Use a new OUTPUT_FILE."
            )

    else:
        results = {
            "base_model": BASE_MODEL,
            "adapter_path": ADAPTER_PATH,
            "input_directory": str(INPUT_DIR),
            "decoding": "greedy",
            "instruction": INSTRUCTION,
            "examples": [],
            "errors": [],
            "metrics": {},
        }

    existing = {
        example["folder"]: example
        for example in results["examples"]
    }

    tokenizer, model = load_model()

    progress = tqdm(
        total=len(sample_files) * 6,
        desc="A/B/C term inference",
        unit="prompt",
        dynamic_ncols=True,
    )

    for sample_path in sample_files:
        folder_id = str(
            sample_path.parent.relative_to(INPUT_DIR)
        )

        try:
            sample, positions = prepare_example(sample_path)

        except Exception as exc:
            if not any(
                e["folder"] == folder_id
                for e in results["errors"]
            ):
                results["errors"].append({
                    "folder": folder_id,
                    "error": str(exc),
                })

            tqdm.write(f"Skipping {folder_id}: {exc}")
            progress.update(6)
            save_results(results)
            continue

        if folder_id in existing:
            example = existing[folder_id]
        else:
            example = {
                "folder": folder_id,
                "source_sha256": sample["source_sha256"],
                "move_number": sample["move_number"],
                "move_counts": {
                    name: positions[name]["history_length"]
                    for name in ("X", "Y")
                },
                "candidate_moves": sample["moves"],
                "reference_winrates": {
                    name: sample["results"][name]["winrate"]
                    for name in ("X", "Y")
                },
                "predictions": {
                    "X": [],
                    "Y": [],
                },
                "metrics": {},
            }

            results["examples"].append(example)
            existing[folder_id] = example

        for position_name in ("X", "Y"):
            position = positions[position_name]

            for candidate in ("A", "B", "C"):
                move = sample["moves"][candidate]

                completed = any(
                    row["candidate"] == candidate
                    for row in example["predictions"][position_name]
                )

                if completed:
                    progress.update(1)
                    continue

                progress.set_postfix(
                    example=folder_id,
                    position=position_name,
                    candidate=candidate,
                )

                prompt = build_prompt(position, move)

                try:
                    output, input_tokens = generate(
                        tokenizer,
                        model,
                        prompt,
                    )

                    term = extract_term(output)
                    error = None

                except Exception as exc:
                    output = ""
                    input_tokens = None
                    term = None
                    error = str(exc)

                row = {
                    "candidate": candidate,
                    "move": move,
                    "move_count": position["history_length"],
                    "input_tokens": input_tokens,
                    "generated_term": term,
                    "is_atari": (
                        normalize_term(term) == "atari"
                        if term is not None else None
                    ),
                    "reference_winrate": sample["results"][
                        position_name
                    ]["winrate"][candidate],
                    "generated_output": output,
                    "error": error,
                    "prompt": prompt,
                }

                example["predictions"][
                    position_name
                ].append(row)

                example["metrics"] = evaluate_example(example)
                save_results(results)

                progress.update(1)

    progress.close()

    results["metrics"] = evaluate_dataset(results)
    save_results(results)
    print(f"Saved {results['metrics']['total_predictions']} predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
