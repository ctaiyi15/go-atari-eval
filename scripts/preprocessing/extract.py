from __future__ import annotations
import json
import random
import os
from collections import defaultdict
from pathlib import Path
from sgfmill import sgf, boards
ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))
from term_matcher import name_move, load_library
ANNOTATIONS = ROOT / 'annotations'
INTERMEDIATE_DIR = ROOT / 'data/intermediate/logo_format'
TERMS_FILE = ROOT / 'src/terms.json'
MAX_FILES = 1500
SAMPLES_PER_TERM = 3000
TARGET_TERMS = ['Stretch', 'Atari', 'Connect', 'Hane', 'Tiger’s Mouth']
GO_COLUMNS = 'ABCDEFGHJKLMNOPQRST'

def sgfmill_move_to_string(color: str, move: tuple[int, int] | None) -> str:
    color = color.upper()
    if move is None:
        return f'{color}[]'
    row, col = move
    sgf_x = chr(ord('a') + col)
    sgf_y = chr(ord('a') + row)
    return f'{color}[{sgf_x}{sgf_y}]'

def board_to_matrix(board: boards.Board) -> list[list[int]]:
    size = board.side
    result = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            stone = board.get(row, col)
            if stone == 'b':
                result[row][col] = 1
            elif stone == 'w':
                result[row][col] = -1
    return result

def get_comment(node) -> str | None:
    try:
        comment = node.get('C')
    except KeyError:
        return None
    if not comment:
        return None
    comment = comment.strip()
    return comment if comment else None

def traverse_node(node, board: boards.Board, move_history: list[str], examples: list[dict], game_id: str):
    current_board = board.copy()
    current_history = list(move_history)
    try:
        color, move = node.get_move()
    except Exception:
        color, move = (None, None)
    if color is not None:
        move_string = sgfmill_move_to_string(color, move)
        comment = get_comment(node)
        board_state = board_to_matrix(current_board)
        example = {'game_id': game_id, 'move_number': len(current_history) + 1, 'position_before_move': list(current_history), 'board_state': board_state, 'move': move_string, 'comment': comment, 'has_comment': comment is not None}
        examples.append(example)
        if move is not None:
            row, col = move
            try:
                current_board.play(row, col, color)
            except ValueError as e:
                raise ValueError(f'Illegal move {move_string}: {e}')
        current_history.append(move_string)
    for child in node:
        traverse_node(child, current_board, current_history, examples, game_id)

def process_sgf(path: str) -> list[dict]:
    with open(path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    board = boards.Board(size)
    root = game.get_root()
    examples = []
    game_id = Path(path).stem
    traverse_node(node=root, board=board, move_history=[], examples=examples, game_id=game_id)
    return examples

def parse_move(move_string: str) -> tuple[int, tuple[int, int] | None]:
    color = move_string[0]
    sign = 1 if color == 'B' else -1
    start = move_string.find('[') + 1
    end = move_string.find(']')
    coord = move_string[start:end]
    if not coord:
        return (sign, None)
    x = ord(coord[0]) - ord('a')
    y = ord(coord[1]) - ord('a')
    return (sign, (x, y))

def has_valid_comment(example: dict) -> bool:
    comment = example.get('comment')
    if comment is None:
        return False
    if not isinstance(comment, str):
        return False
    return bool(comment.strip())

def sgf_to_go_coord(move_string: str) -> tuple[str, str]:
    color = move_string[0]
    symbol = 'X' if color == 'B' else 'O'
    start = move_string.find('[') + 1
    end = move_string.find(']')
    sgf_coord = move_string[start:end]
    if not sgf_coord:
        return (symbol, 'pass')
    sgf_x = ord(sgf_coord[0]) - ord('a')
    sgf_y = ord(sgf_coord[1]) - ord('a')
    column = GO_COLUMNS[sgf_x]
    row = 19 - sgf_y
    return (symbol, f'{column}{row}')

def format_move_history(history: list[str]) -> str:
    formatted = []
    for i, move in enumerate(history, start=1):
        symbol, coord = sgf_to_go_coord(move)
        formatted.append(f'{i}.{symbol}-{coord}')
    return ' '.join(formatted)

def format_candidate_move(move_string: str) -> str:
    symbol, coord = sgf_to_go_coord(move_string)
    return f'{symbol}-{coord}'

def format_board(board: list[list[int]]) -> str:
    lines = []
    for row in board:
        lines.append(' '.join((str(value) for value in row)))
    return '\n'.join(lines)

def make_input(example: dict) -> str:
    history = format_move_history(example['position_before_move'])
    candidate = format_candidate_move(example['move'])
    board = format_board(example['board_state'])
    prompt = f'Given the following Go game state and candidate move, identify the Go term that describes the candidate move and provide commentary explaining the move.\n\nMove history:\n{history}\n\nBoard state:\n{board}\n\nCandidate move:\n{candidate}'
    return prompt

def make_output(example: dict) -> str:
    term = example['term']
    comment = example['comment'].strip()
    return f'Term: {term}\nComment: {comment}'

def convert_example(example: dict) -> dict:
    return {'input': make_input(example), 'output': make_output(example)}

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    paths = []
    for directory, _, files in os.walk(ANNOTATIONS):
        for filename in files:
            if filename.lower().endswith('.sgf'):
                paths.append(os.path.join(directory, filename))
    paths.sort()
    if MAX_FILES is not None:
        paths = paths[:MAX_FILES]
    examples = []
    for i, path in enumerate(paths, 1):
        try:
            examples.extend(process_sgf(path))
        except Exception as e:
            print(f'SGF error: {path}: {e}')
        if i % 10 == 0:
            print(f'Collected {i}/{len(paths)} games')
    patterns = load_library(str(TERMS_FILE))
    matched = []
    for i, example in enumerate(examples):
        sign, vertex = parse_move(example['move'])
        try:
            term = name_move(example['board_state'], sign, vertex, patterns)
        except Exception as e:
            print(f'Term error {i}: {e}')
            term = None
        if term is not None:
            matched.append({**example, 'term': term})
    random.seed(42)
    by_term = defaultdict(list)
    for example in matched:
        term = example.get('term')
        if term in TARGET_TERMS and has_valid_comment(example):
            by_term[term].append(example)
    sampled_by_term = {}
    for term in TARGET_TERMS:
        candidates = by_term[term]
        random.shuffle(candidates)
        sampled_by_term[term] = candidates[:min(SAMPLES_PER_TERM, len(candidates))]
        print(f'{term}: {len(sampled_by_term[term])}')
    balanced = []
    for term in TARGET_TERMS:
        balanced.extend(sampled_by_term[term])
    random.shuffle(balanced)
    splits = {'train': [], 'val': [], 'test': []}
    for term in TARGET_TERMS:
        term_examples = sampled_by_term[term]
        random.shuffle(term_examples)
        n = len(term_examples)
        n_train, n_val = (int(n * 0.8), int(n * 0.1))
        splits['train'].extend(term_examples[:n_train])
        splits['val'].extend(term_examples[n_train:n_train + n_val])
        splits['test'].extend(term_examples[n_train + n_val:])
    for split, items in splits.items():
        random.shuffle(items)
        converted = []
        for i, example in enumerate(items):
            try:
                converted.append(convert_example(example))
            except Exception as e:
                print(f'Conversion error {split}/{i}: {e}')
        save_json(INTERMEDIATE_DIR / f'{split}.json', converted)
        print(f'{split}: {len(converted)} examples')
if __name__ == '__main__':
    main()
