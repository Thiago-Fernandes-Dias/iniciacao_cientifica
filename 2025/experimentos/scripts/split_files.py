import re
import shutil
from pathlib import Path


SOURCES: list[tuple[Path, list[tuple[str, Path]]]] = [
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_a/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB1_A/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB1_A/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_b/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB1_B/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB1_B/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db2_a/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB2_A/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB2_A/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db2_b/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB2_B/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB2_B/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db3_a/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB3_A/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB3_A/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db3_b/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB3_B/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB3_B/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db4_a/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB4_A/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB4_A/image/query")),
        ],
    ),
    (
        Path("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db4_b/bmp"),
        [
            (r"_[1-4]", Path("/mnt/d/Datasets/FVC_2000_DB4_B/image/gallery")),
            (r"_[5-8]", Path("/mnt/d/Datasets/FVC_2000_DB4_B/image/query")),
        ],
    ),
]


def copy_file(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / src.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.copy2(src, dest)
    return dest


def process_source(source: Path, patterns: list[tuple[str, Path]]) -> None:
    source = source.resolve()
    if not source.is_dir():
        print(f"Error: {source} is not a directory")
        return

    for pattern, dest_dir in patterns:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(source.iterdir()):
        if not f.is_file():
            continue
        for pattern, dest in patterns:
            if re.search(pattern, f.name):
                dest_file = copy_file(f, dest)
                print(f"{f.name} -> {dest_file}")
                break
        else:
            print(f"{f.name} -> (no match, skipped)")


def main():
    for source, patterns in SOURCES:
        process_source(source, patterns)


if __name__ == "__main__":
    main()
