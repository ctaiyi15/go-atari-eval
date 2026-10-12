from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

import torch
from tqdm.auto import tqdm
from peft import PeftModel
from sgfmill import sgf, sgf_moves
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = "YichuanMa/LoGos-7B"
ADAPTER_PATH = ROOT_DIR / "outputs/logo-qlora/final"
INPUT_DIR = ROOT_DIR / "go-eval-positions/outputs-addition"
OUTPUT_FILE = ROOT_DIR / "outputs/addition.json"
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
        history.append(f"{index}.{player}-{coordinate(row, col)}")
        board.play(row, col, color)

    board_rows = []
    for row in range(18, -1, -1):
        values = []
        for col in range(19):
            stone = board.get(row, col)
            values.append("1" if stone == "b" else "-1" if stone == "w" else "0")
        board_rows.append(" ".join(values))
    root = game.get_root()
    player_to_play = root.get("PL") if root.has_property("PL") else None
    next_player = ("X" if player_to_play == "b" else "O") if player_to_play else (
        "O" if last_color == "b" else "X"
    )
    return {
        "move_history": " ".join(history),
        "board_state": "\n".join(board_rows),
        "history_length": len(moves),
        "next_player": next_player,
    }


def build_prompt(position, candidate_move):
    example_input = (
        INSTRUCTION
        + "\n\nMove history:\n" + position["move_history"]
        + "\n\nBoard state:\n" + position["board_state"]
        + "\n\nCandidate move:\n"
        + f"{position['next_player']}-{candidate_move}"
    )
    return "### Instruction:\n" + example_input + "\n\n### Response:\n"


def extract_term(output):
    match = re.search(r"(?im)^[ \t]*Term[ \t]*:[ \t]*\*?([^\n\r]+)", output)
    if not match:
        return None
    term = match.group(1).strip().rstrip(".:;, ")
    return term or None


def extract_comment(output):
    match = re.search(r"(?im)^[ \t]*Comment[ \t]*:[ \t]*", output)
    if not match:
        return None
    body = re.split(
        r"(?im)^\s*(?:Win\s*rate|Term)\s*:",
        output[match.end():], maxsplit=1,
    )[0].strip()
    return body or None


def extract_winrate(output):
    match = re.search(r"(?im)^[ \t]*Win[ \t]*rate[ \t]*:[ \t]*([0-9]*\.?[0-9]+)[ \t]*(%)?", output)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value /= 100
    return value if 0 <= value <= 1 else None


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto",
        trust_remote_code=True, torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate(tokenizer, model, prompt):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
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
    generated = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    return generated.strip(), input_length


def prepare_example(sample_path):
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    moves = sample["moves"]
    if not isinstance(moves, dict) or not all(k in moves for k in ("A", "B")):
        raise ValueError("Missing A or B candidate in moves")
    position = load_position(sample_path.parent / "X.sgf")
    expected_length = sample["move_number"]
    if position["history_length"] != expected_length:
        raise ValueError(
            f"X: SGF has {position['history_length']} moves, expected {expected_length}"
        )
    reference = sample["results"]["X"]["winrate"]
    if not all(k in reference for k in ("A", "B")):
        raise ValueError("Missing A/B reference winrates under results.X.winrate")
    return sample, position


def save_results(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUTPUT_FILE)


def summarize(results):
    rows = [row for ex in results["examples"] for row in ex["predictions"]["X"]]
    terms = [r for r in rows if r["generated_term"] is not None]
    comments = [r for r in rows if r["generated_comment"] is not None]
    pairs = [r for r in rows if r["generated_winrate"] is not None and r["reference_winrate"] is not None]
    errors = [abs(r["generated_winrate"] - r["reference_winrate"]) for r in pairs]
    return {
        "examples": len(results["examples"]),
        "expected_predictions": len(results["examples"]) * 2,
        "predictions": len(rows),
        "terms_generated": len(terms),
        "term_generation_rate": len(terms) / len(rows) if rows else None,
        "term_counts": dict(sorted(Counter(r["generated_term"].lower() for r in terms).items(), key=lambda item: -item[1])),
        "comments_generated": len(comments),
        "comment_generation_rate": len(comments) / len(rows) if rows else None,
        "average_comment_words": sum(len(r["generated_comment"].split()) for r in comments) / len(comments) if comments else None,
        "winrates_generated": len(pairs),
        "winrate_generation_rate": len(pairs) / len(rows) if rows else None,
        "winrate_mae": sum(errors) / len(errors) if errors else None,
        "generation_errors": sum(bool(r["error"]) for r in rows),
    }


def main():
    sample_files = sorted(INPUT_DIR.rglob("sample.json"))
    if not sample_files:
        raise FileNotFoundError(f"No sample.json files found under {INPUT_DIR}")

    settings = {
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
        "input_directory": str(INPUT_DIR),
        "instruction": INSTRUCTION,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_generation_seconds": MAX_GENERATION_SECONDS,
        "decoding": "greedy",
        "dataset_type": "addition_X_AB_term_winrate_comment",
        "output_order": "Term > Win rate > Comment",
    }
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        for key, value in settings.items():
            if results.get(key) != value:
                raise ValueError(f"Output settings differ for {key}; use another OUTPUT_FILE")
    else:
        results = {**settings, "examples": [], "errors": [], "metrics": {}}

    existing = {ex["folder"]: ex for ex in results["examples"]}
    tokenizer, model = load_model()
    progress = tqdm(total=len(sample_files) * 2, desc="Addition X A/B", unit="prompt")

    for sample_path in sample_files:
        folder_id = str(sample_path.parent.relative_to(INPUT_DIR))
        try:
            sample, position = prepare_example(sample_path)
        except Exception as exc:
            if not any(e["folder"] == folder_id for e in results["errors"]):
                results["errors"].append({"folder": folder_id, "error": str(exc)})
            tqdm.write(f"Skipping {folder_id}: {exc}")
            progress.update(2)
            save_results(results)
            continue

        if folder_id not in existing:
            example = {
                "folder": folder_id,
                "source_sha256": sample.get("source_sha256"),
                "move_number": sample["move_number"],
                "candidate_moves": {k: sample["moves"][k] for k in ("A", "B")},
                "reference_winrates": sample["results"]["X"]["winrate"],
                "reference_score_leads": sample["results"]["X"].get("score_lead"),
                "X_A_targets": sample.get("X_A_targets"),
                "predictions": {"X": []},
            }
            results["examples"].append(example)
            existing[folder_id] = example
        else:
            example = existing[folder_id]

        for candidate in ("A", "B"):
            if any(row["candidate"] == candidate for row in example["predictions"]["X"]):
                progress.update(1)
                continue
            move = sample["moves"][candidate]
            prompt = build_prompt(position, move)
            progress.set_postfix(folder=folder_id, candidate=candidate)
            try:
                output, input_tokens = generate(tokenizer, model, prompt)
                term = extract_term(output)
                comment = extract_comment(output)
                winrate = extract_winrate(output)
                error = None
            except Exception as exc:
                output, input_tokens, term, comment, winrate = "", None, None, None, None
                error = str(exc)
            row = {
                "candidate": candidate,
                "move": move,
                "move_count": position["history_length"],
                "input_tokens": input_tokens,
                "generated_term": term,
                "generated_winrate": winrate,
                "generated_comment": comment,
                "reference_winrate": sample["results"]["X"]["winrate"][candidate],
                "generated_output": output,
                "error": error,
                "prompt": prompt,
            }
            example["predictions"]["X"].append(row)
            results["metrics"] = summarize(results)
            save_results(results)
            progress.update(1)

    progress.close()
    results["metrics"] = summarize(results)
    save_results(results)
    print(json.dumps(results["metrics"], indent=2))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()