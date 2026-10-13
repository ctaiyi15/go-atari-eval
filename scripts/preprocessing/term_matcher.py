from __future__ import annotations

import json
from pathlib import Path


def load_library(path: str | Path = "src/terms.json") -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        library = json.load(f)
    if not isinstance(library, list):
        raise ValueError("Term library must be a JSON list")
    return library


def neighbors(x: int, y: int, width: int, height: int):
    return [(a, b) for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            if 0 <= a < width and 0 <= b < height]


def liberties(board: list[list[int]], vertex: tuple[int, int],
              visited: set | None = None, empty: set | None = None) -> int:
    if visited is None:
        visited = set()
    if empty is None:
        empty = set()
    height, width = len(board), len(board[0])
    x, y = vertex
    sign = board[y][x]
    visited.add(vertex)
    for nx, ny in neighbors(x, y, width, height):
        if len(empty) >= 3:
            break
        if board[ny][nx] == -sign or (nx, ny) in visited:
            continue
        if board[ny][nx] == 0:
            empty.add((nx, ny))
        else:
            liberties(board, (nx, ny), visited, empty)
    return len(empty)


def symmetries(x: int, y: int):
    return ((x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x))


def matches(board: list[list[int]], move: tuple[int, int], pattern: dict) -> bool:
    height, width = len(board), len(board[0])
    x, y = move
    size = pattern.get("size")
    if size is not None and (width != height or width != int(size)):
        return False
    played_sign = board[y][x]
    if played_sign == 0:
        return False

    for anchor, anchor_sign in pattern.get("anchors", []):
        ax, ay = map(int, anchor)
        anchor_sign = int(anchor_sign)
        if pattern.get("type") == "corner":
            positions = ((ax, ay), (width - 1 - ax, ay),
                         (ax, height - 1 - ay), (width - 1 - ax, height - 1 - ay),
                         (ay, ax), (width - 1 - ay, ax),
                         (ay, height - 1 - ax), (width - 1 - ay, height - 1 - ax))
            if move not in positions:
                continue

        possible = [True] * 8
        for vertex, vertex_sign in pattern.get("vertices", []):
            vx, vy = map(int, vertex)
            expected = int(vertex_sign) * played_sign * anchor_sign
            for i, (dx, dy) in enumerate(symmetries(vx - ax, vy - ay)):
                px, py = x + dx, y + dy
                if possible[i] and not (0 <= px < width and 0 <= py < height
                                        and board[py][px] == expected):
                    possible[i] = False
            if not any(possible):
                break
        if any(possible):
            return True
    return False


def name_move(board: list[list[int]], sign: int,
              vertex: tuple[int, int] | None, library: list[dict]) -> str | None:
    height, width = len(board), len(board[0])
    if sign == 0 or vertex is None:
        return "Pass"
    x, y = vertex
    if not (0 <= x < width and 0 <= y < height):
        return "Pass"
    if board[y][x] != 0:
        return None

    adjacent = neighbors(x, y, width, height)
    for nx, ny in adjacent:
        if board[ny][nx] == -sign:
            count = liberties(board, (nx, ny))
            if count == 1:
                return "Take"
            if count == 2:
                return "Atari"

    after = [row[:] for row in board]
    after[y][x] = sign
    if liberties(after, vertex) == 0:
        return "Suicide"

    friendly = sum(board[ny][nx] == sign for nx, ny in adjacent)
    if friendly == len(adjacent):
        return "Fill"
    if friendly >= 2:
        return "Connect"

    for pattern in library:
        if matches(after, vertex, pattern):
            return pattern.get("name")

    if width % 2 and height % 2 and vertex == (width // 2, height // 2):
        return "Tengen"

    if min(width, height) > 7:
        near_x, near_y = (3 if width >= 13 else 2), (3 if height >= 13 else 2)
        far_x, far_y = width - near_x - 1, height - near_y - 1
        hoshi = []
        if width % 2:
            hoshi.extend(((width // 2, near_y), (width // 2, far_y)))
        if height % 2:
            hoshi.extend(((near_x, height // 2), (far_x, height // 2)))
        if vertex in hoshi:
            return "Hoshi"
    return None
