from main import RESULT_PATH
import json

for path in RESULT_PATH.iterdir():
    if not path.is_file():
        continue

    save_dir = RESULT_PATH / path.stem
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(path) as f:
        data = json.load(f)

    for category, cat_data in data.items():
        with open(save_dir / f"{category}.json", "w") as f:
            json.dump(cat_data, f)
