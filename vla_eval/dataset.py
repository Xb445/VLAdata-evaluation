"""Read the v3 metadata and low-dimensional columns without loading torch."""

from pathlib import Path
from string import Formatter

from .io import distinct, positive, read_json, tree_hash


def _parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValueError("Install pyarrow in the evaluation environment") from exc
    return parquet


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def _shards(root, template):
    pattern = "".join(
        literal + ("*" if key is not None else "") for literal, key, _, _ in Formatter().parse(template)
    )
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError(f"Invalid dataset path pattern: {template}")
    return sorted({_inside(root, p.relative_to(root)) for p in root.glob(pattern) if p.is_file()})


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = read_json(self.root / "meta/info.json")
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("Expected LeRobotDataset v3.0")
        self.fps = positive(self.info["fps"], "dataset FPS")
        self.features = self.info["features"]
        self.cameras = {k: v["dtype"] for k, v in self.features.items() if v["dtype"] in {"video", "image"}}
        self.data = _shards(self.root, self.info["data_path"])
        self.meta_files = _shards(self.root, "meta/episodes/chunk-*/file-*.parquet")
        if not self.data or not self.meta_files:
            raise ValueError("Data or episode parquet shards are missing")
        pq = _parquet()
        self.episodes = distinct(
            (row for path in self.meta_files for row in pq.ParquetFile(path).read().to_pylist()),
            "episode_index",
        )
        if sorted(self.episodes) != list(range(self.info["total_episodes"])):
            raise ValueError("Episode indices must match meta/info.json")
        task_rows = pq.ParquetFile(self.root / "meta/tasks.parquet").read().to_pylist()
        self.tasks = distinct(task_rows, "task_index")
        self.data_map = {}
        self.video_map = {}
        self._video_error = {}
        for index, meta in self.episodes.items():
            path = self.data_file(meta)
            self.data_map.setdefault(path, set()).add(index)
            for key, dtype in self.cameras.items():
                if dtype == "video":
                    try:
                        self.video_map[index, key] = self.video_file(meta, key)
                    except (KeyError, TypeError, ValueError):
                        self._video_error[index, key] = "invalid_video_metadata"
        if not set(self.data_map).issubset(self.data):
            raise ValueError("Episode metadata references a missing data shard")

    def data_file(self, meta):
        return _inside(
            self.root,
            self.info["data_path"].format(
                chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
            ),
        )

    def video_file(self, meta, key):
        prefix = f"videos/{key}"
        return _inside(
            self.root,
            self.info["video_path"].format(
                video_key=key,
                chunk_index=meta[f"{prefix}/chunk_index"],
                file_index=meta[f"{prefix}/file_index"],
            ),
        )

    def fingerprint(self):
        meta = [p for p in (self.root / "meta").glob("*.json") if p.is_file()]
        videos = [p for p in self.video_map.values() if p.is_file()]
        return tree_hash(
            self.root, [*self.data, *self.meta_files, *meta, *videos, self.root / "meta/tasks.parquet"]
        )

    def frames(self, extra_columns=()):
        """Yield each episode once; a parquet file can contain several episodes."""
        pq = _parquet()
        core = {
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "task_index",
            "action",
            "observation.state",
        }
        seen = set()
        for path in self.data:
            file = pq.ParquetFile(path)
            columns = set(file.schema_arrow.names)
            if not core <= columns:
                raise ValueError(f"{path.name} is missing columns {sorted(core - columns)}")
            selected = sorted(core | (set(extra_columns) & columns))
            rows = []
            active = None
            for batch in file.iter_batches(columns=selected):
                for row in batch.to_pylist():
                    index = row["episode_index"]
                    if index != active:
                        if rows:
                            yield active, rows
                        if index not in self.data_map.get(path, ()) or index in seen:
                            raise ValueError(f"Unexpected or discontiguous episode {index} in {path}")
                        seen.add(index)
                        active, rows = index, []
                    rows.append(row)
            if rows:
                yield active, rows
        if seen != set(self.episodes):
            raise ValueError(f"Missing episode rows: {sorted(set(self.episodes) - seen)}")

    def task_text(self, index):
        row = self.tasks.get(index, {})
        return row.get("task", row.get("__index_level_0__", ""))

    def check_dimensions(self, contract):
        for key, dims in (("action", "action_dimensions"), ("observation.state", "state_dimensions")):
            feature = self.features.get(key, {})
            names = [d["name"] for d in contract[dims]]
            if list(feature.get("shape", ())) != [len(names)]:
                raise ValueError(f"{key} shape does not match the declared dimensions")
            declared_names = feature.get("names")
            if declared_names is not None and declared_names != names:
                raise ValueError(f"{key} names or order differ from the dataset contract")
        cameras = set(contract["camera_keys"])
        if not cameras or not cameras <= self.cameras.keys():
            raise ValueError("Required cameras do not match LeRobot feature keys")
