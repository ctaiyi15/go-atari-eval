from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parent.parent.parent
KATAGO_EXE = str(ROOT / 'katago/katago.exe')
KATAGO_MODEL = str(ROOT / 'katago/kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz')
KATAGO_CONFIG = str(ROOT / 'katago/analysis_example.cfg')
INPUT_FILES = [ROOT / 'data/intermediate/logo_format/train.json', ROOT / 'data/intermediate/logo_format/val.json', ROOT / 'data/intermediate/logo_format/test.json']
OUTPUT_DIR = ROOT / 'data/final/logo_format'
MAX_MOVES = 150
SAVE_EVERY = 100
# X = Black, O = White
SYMBOL_TO_COLOR = {'X': 'B', 'O': 'W'}
KATAGO_WINRATE_PERSPECTIVE = 'BLACK'

def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)

def extract_move_history(input_text: str) -> list[tuple[str, str]]:
    match = re.search('Move history:\\s*(.*?)\\s*Board state:', input_text, flags=re.DOTALL)
    if not match:
        raise ValueError('Could not find Move history.')
    history_text = match.group(1)
    moves = re.findall('\\d+\\.\\s*([OX])-([A-HJ-T]\\d+|pass)', history_text, flags=re.IGNORECASE)
    return [(symbol.upper(), coord.upper()) for symbol, coord in moves]

def extract_candidate_move(input_text: str) -> tuple[str, str]:
    match = re.search('Candidate move:\\s*([OX])-([A-HJ-T]\\d+|pass)', input_text, flags=re.IGNORECASE)
    if not match:
        raise ValueError('Could not find Candidate move.')
    symbol = match.group(1).upper()
    coord = match.group(2).upper()
    return (symbol, coord)

def add_winrate_to_output(output_text: str, winrate: float) -> str:
    winrate_percent = winrate * 100
    lines = output_text.splitlines()
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if line.startswith('Term:') and (not inserted):
            new_lines.append(f'Winrate: {winrate_percent:.2f}%')
            inserted = True
    if not inserted:
        raise ValueError("Could not find 'Term:' in output.")
    return '\n'.join(new_lines)

def start_katago() -> subprocess.Popen:
    command = [KATAGO_EXE, 'analysis', '-model', KATAGO_MODEL, '-config', KATAGO_CONFIG]
    print()
    print('Starting KataGo...')
    print(' '.join(command))
    print()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
    return process

def build_katago_query(query_id: str, history: list[tuple[str, str]], candidate: tuple[str, str]) -> dict[str, Any]:
    candidate_symbol, candidate_coord = candidate
    moves = []
    for symbol, coord in history:
        color = SYMBOL_TO_COLOR[symbol]
        moves.append([color, coord])
    candidate_color = SYMBOL_TO_COLOR[candidate_symbol]
    moves.append([candidate_color, candidate_coord])
    query = {'id': query_id, 'moves': moves, 'initialStones': [], 'rules': 'chinese', 'komi': 7.5, 'boardXSize': 19, 'boardYSize': 19, 'analyzeTurns': [len(moves)], 'initialPlayer': 'B'}
    return query

def query_katago(process: subprocess.Popen, query: dict[str, Any]) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(query) + '\n')
    process.stdin.flush()
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('KataGo stopped unexpectedly.')
        line = line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if response.get('id') == query['id']:
            return response

def get_candidate_player_winrate(response: dict[str, Any], candidate_symbol: str) -> float:
    root_info = response.get('rootInfo')
    if root_info is None:
        raise ValueError('KataGo response does not contain rootInfo.')
    black_winrate = root_info.get('winrate')
    if black_winrate is None:
        raise ValueError('KataGo response does not contain rootInfo.winrate.')
    black_winrate = float(black_winrate)
    candidate_color = SYMBOL_TO_COLOR[candidate_symbol]
    if candidate_color == 'B':
        return black_winrate
    return 1.0 - black_winrate

def get_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + '.checkpoint.json')

def get_error_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + '.errors.json')

def save_checkpoint(output_path: Path, output_data: list[dict], next_index: int, skipped_long: int, failed: int, failures: list[dict]) -> None:
    checkpoint_path = get_checkpoint_path(output_path)
    atomic_write_json(output_path, output_data)
    checkpoint = {'next_index': next_index, 'skipped_long': skipped_long, 'failed': failed, 'failures': failures, 'output_data': output_data}
    atomic_write_json(checkpoint_path, checkpoint)
    print()
    print('--------------------------------------------------')
    print(f'CHECKPOINT SAVED')
    print(f'Next input index: {next_index}')
    print(f'Output examples:  {len(output_data)}')
    print(f'Skipped >150:     {skipped_long}')
    print(f'Failed:           {failed}')
    print(f'Checkpoint:       {checkpoint_path}')
    print('--------------------------------------------------')
    print()

def load_checkpoint(output_path: Path) -> tuple[int, list[dict], int, int, list[dict]]:
    checkpoint_path = get_checkpoint_path(output_path)
    if not checkpoint_path.exists():
        return (0, [], 0, 0, [])
    print()
    print(f'Found checkpoint:')
    print(f'  {checkpoint_path}')
    with checkpoint_path.open('r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    next_index = checkpoint.get('next_index', 0)
    output_data = checkpoint.get('output_data', [])
    skipped_long = checkpoint.get('skipped_long', 0)
    failed = checkpoint.get('failed', 0)
    failures = checkpoint.get('failures', [])
    print(f'Resuming from input example {next_index + 1}')
    print(f'Already generated: {len(output_data)}')
    print(f'Already skipped: {skipped_long}')
    print(f'Already failed: {failed}')
    print()
    return (next_index, output_data, skipped_long, failed, failures)

def process_dataset(input_path: Path, output_path: Path, katago: subprocess.Popen) -> None:
    print()
    print('=' * 70)
    print(f'Input : {input_path}')
    print(f'Output: {output_path}')
    print('=' * 70)
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Original examples: {len(data)}')
    start_index, output_data, skipped_long, failed, failures = load_checkpoint(output_path)
    next_index = start_index
    try:
        for index in range(start_index, len(data)):
            example = data[index]
            try:
                input_text = example['input']
                output_text = example['output']
                history = extract_move_history(input_text)
                move_count = len(history)
                if move_count > MAX_MOVES:
                    skipped_long += 1
                    next_index = index + 1
                    print(f'[{index + 1}/{len(data)}] SKIP: {move_count} moves')
                else:
                    candidate_symbol, candidate_coord = extract_candidate_move(input_text)
                    query = build_katago_query(query_id=str(index), history=history, candidate=(candidate_symbol, candidate_coord))
                    response = query_katago(katago, query)
                    winrate = get_candidate_player_winrate(response, candidate_symbol)
                    new_example = dict(example)
                    new_example['output'] = add_winrate_to_output(output_text, winrate)
                    output_data.append(new_example)
                    next_index = index + 1
                    print(f'[{index + 1}/{len(data)}] {candidate_symbol}-{candidate_coord} | after move #{move_count + 1} | winrate={winrate * 100:.2f}%')
            except Exception as e:
                failed += 1
                failures.append({'index': index, 'error': str(e)})
                next_index = index + 1
                print(f'[{index + 1}/{len(data)}] ERROR: {e}')
            processed_since_start = index - start_index + 1
            if processed_since_start % SAVE_EVERY == 0:
                save_checkpoint(output_path=output_path, output_data=output_data, next_index=next_index, skipped_long=skipped_long, failed=failed, failures=failures)
    except KeyboardInterrupt:
        print()
        print()
        print('Keyboard interrupt detected.')
        print('Saving current progress before exiting...')
        save_checkpoint(output_path=output_path, output_data=output_data, next_index=next_index, skipped_long=skipped_long, failed=failed, failures=failures)
        raise
    atomic_write_json(output_path, output_data)
    error_path = get_error_path(output_path)
    if failures:
        atomic_write_json(error_path, failures)
    elif error_path.exists():
        error_path.unlink()
    checkpoint_path = get_checkpoint_path(output_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print()
    print('=' * 70)
    print(f'FINISHED: {input_path.name}')
    print('=' * 70)
    print(f'Original:     {len(data)}')
    print(f'Saved:        {len(output_data)}')
    print(f'Skipped >150: {skipped_long}')
    print(f'Failed:       {failed}')
    print(f'Output:       {output_path}')
    if failures:
        print(f'Errors:       {error_path}')
    print('=' * 70)
    print()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    katago = start_katago()
    try:
        for input_path in INPUT_FILES:
            output_path = OUTPUT_DIR / input_path.name
            process_dataset(input_path=input_path, output_path=output_path, katago=katago)
    finally:
        print()
        print('Stopping KataGo...')
        katago.terminate()
        try:
            katago.wait(timeout=10)
        except subprocess.TimeoutExpired:
            katago.kill()
        print('KataGo stopped.')
if __name__ == '__main__':
    main()
