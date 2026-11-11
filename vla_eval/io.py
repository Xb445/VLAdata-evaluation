"""File and protocol utilities shared by the evaluation stages."""

import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("expected a JSON object")
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                yield item


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temp = Path(file.name)
        try:
            file.write(text)
            file.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    write_text(path, "".join(encode(row) + "\n" for row in rows))


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def tree_hash(root, paths):
    root = Path(root).resolve()
    return digest([(str(p.relative_to(root)), file_hash(p)) for p in sorted(set(paths))])


def distinct(rows, key):
    index = {}
    for row in rows:
        value = row[key]
        if value in index:
            raise ValueError(f"Duplicate {key}: {value}")
        index[value] = row
    return index


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def positive(value, name):
    if not number(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def weights(values, name):
    if not values or any(not number(v) or v <= 0 for v in values.values()):
        raise ValueError(f"{name} must have finite positive weights")
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def require_same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} changed or does not match the frozen protocol")
