import requests
from pathlib import Path

out_dir = Path('annotations')
out_dir.mkdir(exist_ok=True)

base_url = 'https://raw.githubusercontent.com/AndreHe02/go/master/data/annotations'

for i in range(1, 9001):
    filename = f'{i:03d}.sgf'
    url = f'{base_url}/{filename}'

    r = requests.get(url)
    if r.status_code == 200:
        (out_dir / filename).write_bytes(r.content)
        print(f'Downloaded {filename}')
    else:
        print(f'Missing: {filename}')