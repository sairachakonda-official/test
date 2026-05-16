import gc
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb


RAW_A = "project/raw_a"
RAW_B = "project/raw_b"

MAPPING_DIR = "project/mapping"

PROCESSED_A = "project/processed_a"
PROCESSED_B = "project/processed_b"

JOINED_DIR = "project/joined"

TEMP_DIR = "project/temp"

JOIN_COLUMN = "id"

JOIN_TYPE = "INNER"

BUCKETS = 256

THREADS = max(1, os.cpu_count() // 2)

MEMORY_LIMIT = "120GB"

ROW_GROUP_SIZE = 100000

COMPRESSION_INTERMEDIATE = "snappy"

COMPRESSION_FINAL = "zstd"

SKIP_EXISTING = True

MAX_JOIN_WORKERS = min(16, THREADS)

MAX_PROCESS_WORKERS = min(8, THREADS)

CLEAN_AFTER_JOIN = True

CLEAN_AFTER_MERGE = False


# KEEP ONLY REQUIRED COLUMNS
# JOIN_COLUMN automatically included

PROCESS_COLUMNS_A = [
    "id",
    "name",
    "amount",
    "date"
]

PROCESS_COLUMNS_B = [
    "id",
    "score",
    "status"
]


# FINAL JOIN OUTPUT COLUMNS
# None means all processed columns

LEFT_COLUMNS = None

RIGHT_COLUMNS = None


os.makedirs(MAPPING_DIR, exist_ok=True)
os.makedirs(PROCESSED_A, exist_ok=True)
os.makedirs(PROCESSED_B, exist_ok=True)
os.makedirs(JOINED_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def parquet_exists(path):
    return (
        os.path.exists(path)
        and any(Path(path).rglob("*.parquet"))
    )


def file_exists(path):
    return (
        os.path.exists(path)
        and os.path.getsize(path) > 0
    )


def safe_remove(path):
    try:
        if os.path.isfile(path):
            os.remove(path)

        elif os.path.isdir(path):
            shutil.rmtree(path)

    except Exception as e:
        log(f"Cleanup Failed : {path}")
        log(str(e))


def clean_memory():
    gc.collect()


def connect():
    con = duckdb.connect()

    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("PRAGMA enable_object_cache")

    return con


def parquet_files(folder):
    return sorted(str(p) for p in Path(folder).glob("*.parquet"))


def build_process_select(columns):
    if not columns:
        return "t.*"

    unique_cols = []

    seen = set()

    for col in [JOIN_COLUMN] + columns:
        if col not in seen:
            seen.add(col)
            unique_cols.append(f"t.{col}")

    return ",\n".join(unique_cols)


def build_join_select(columns, alias):
    if not columns:
        return f"{alias}.* EXCLUDE(join_id, bucket)"

    cleaned = []

    seen = set()

    for col in columns:
        if col in ["join_id", "bucket"]:
            continue

        if col not in seen:
            seen.add(col)
            cleaned.append(f"{alias}.{col}")

    return ",\n".join(cleaned)


PROCESS_SELECT_A = build_process_select(
    PROCESS_COLUMNS_A
)

PROCESS_SELECT_B = build_process_select(
    PROCESS_COLUMNS_B
)

LEFT_SELECT = build_join_select(
    LEFT_COLUMNS,
    "a"
)

RIGHT_SELECT = build_join_select(
    RIGHT_COLUMNS,
    "b"
)


def create_distinct_ids():
    log("=" * 80)
    log("STEP 1 -> CREATE DISTINCT IDS")
    log("=" * 80)

    start = time.time()

    output = f"{MAPPING_DIR}/distinct_ids.parquet"

    if SKIP_EXISTING and file_exists(output):
        log("Distinct IDs Already Exist -> Skipping")
        return

    con = connect()

    query = f"""
    COPY (
        SELECT DISTINCT {JOIN_COLUMN}
        FROM (
            SELECT {JOIN_COLUMN}
            FROM read_parquet('{RAW_A}/*.parquet')

            UNION ALL

            SELECT {JOIN_COLUMN}
            FROM read_parquet('{RAW_B}/*.parquet')
        )
        WHERE {JOIN_COLUMN} IS NOT NULL
    )
    TO '{output}'
    (
        FORMAT PARQUET,
        COMPRESSION '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output}')
        """
    ).fetchone()[0]

    con.close()

    clean_memory()

    log(f"Distinct IDs : {total:,}")
    log(f"Completed In : {time.time() - start:.2f} sec")


def create_id_map():
    log("=" * 80)
    log("STEP 2 -> CREATE ID MAP")
    log("=" * 80)

    start = time.time()

    output = f"{MAPPING_DIR}/id_map.parquet"

    if SKIP_EXISTING and file_exists(output):
        log("ID Map Already Exists -> Skipping")
        return

    con = connect()

    query = f"""
    COPY (
        SELECT
            {JOIN_COLUMN},

            row_number() OVER (
                ORDER BY {JOIN_COLUMN}
            )::BIGINT AS join_id

        FROM read_parquet(
            '{MAPPING_DIR}/distinct_ids.parquet'
        )
    )
    TO '{output}'
    (
        FORMAT PARQUET,
        COMPRESSION '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output}')
        """
    ).fetchone()[0]

    con.close()

    clean_memory()

    log(f"Global IDs  : {total:,}")
    log(f"Completed In: {time.time() - start:.2f} sec")


def validate_mapping():
    log("=" * 80)
    log("STEP 3 -> VALIDATE MAPPING")
    log("=" * 80)

    con = connect()

    raw_count = con.execute(
        f"""
        SELECT COUNT(DISTINCT {JOIN_COLUMN})
        FROM (
            SELECT {JOIN_COLUMN}
            FROM read_parquet('{RAW_A}/*.parquet')

            UNION ALL

            SELECT {JOIN_COLUMN}
            FROM read_parquet('{RAW_B}/*.parquet')
        )
        WHERE {JOIN_COLUMN} IS NOT NULL
        """
    ).fetchone()[0]

    mapped_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{MAPPING_DIR}/id_map.parquet'
        )
        """
    ).fetchone()[0]

    con.close()

    clean_memory()

    log(f"Distinct Raw IDs : {raw_count:,}")
    log(f"Mapped IDs       : {mapped_count:,}")

    if raw_count != mapped_count:
        raise RuntimeError("ID Mapping Validation Failed")

    log("Validation Passed")


def process_file(
    file_path,
    output_dir,
    select_columns
):
    start = time.time()

    filename = Path(file_path).stem

    output_path = f"{output_dir}/{filename}"

    if (
        SKIP_EXISTING
        and parquet_exists(output_path)
    ):
        log(f"Skipping Existing : {filename}")
        return

    log(f"Processing : {filename}")

    con = connect()

    query = f"""
    COPY (
        SELECT

            {select_columns},

            m.join_id,

            CASE
                WHEN m.join_id IS NULL
                THEN -1
                ELSE abs(hash(m.join_id)) % {BUCKETS}
            END AS bucket

        FROM read_parquet('{file_path}') t

        LEFT JOIN read_parquet(
            '{MAPPING_DIR}/id_map.parquet'
        ) m

        USING({JOIN_COLUMN})
    )
    TO '{output_path}'
    (
        FORMAT PARQUET,
        PARTITION_BY(bucket),
        COMPRESSION '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE},
        OVERWRITE_OR_IGNORE
    )
    """

    con.execute(query)

    rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{file_path}')
        """
    ).fetchone()[0]

    con.close()

    clean_memory()

    log(f"Completed : {filename}")
    log(f"Rows      : {rows:,}")
    log(f"Time      : {time.time() - start:.2f} sec")


def process_folder(
    input_folder,
    output_folder,
    select_columns
):
    log("=" * 80)
    log(f"STEP 4 -> PROCESS {input_folder}")
    log("=" * 80)

    start = time.time()

    files = parquet_files(input_folder)

    log(f"Files Found : {len(files)}")

    with ThreadPoolExecutor(
        max_workers=MAX_PROCESS_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_file,
                file,
                output_folder,
                select_columns
            )
            for file in files
        ]

        completed = 0

        for future in as_completed(futures):
            future.result()

            completed += 1

            log(
                f"Progress : "
                f"{completed}/{len(files)}"
            )

            clean_memory()

    log(
        f"Folder Completed In : "
        f"{time.time() - start:.2f} sec"
    )


def analyze_bucket_distribution(folder):
    log("=" * 80)
    log(f"STEP 5 -> ANALYZE {folder}")
    log("=" * 80)

    con = connect()

    stats = con.execute(
        f"""
        SELECT
            bucket,
            COUNT(*) AS rows
        FROM read_parquet(
            '{folder}/**/*.parquet'
        )
        GROUP BY bucket
        ORDER BY rows DESC
        """
    ).fetchall()

    con.close()

    clean_memory()

    for bucket, rows in stats[:10]:
        log(f"Bucket {bucket} -> {rows:,} rows")


def join_bucket(bucket):
    start = time.time()

    con = connect()

    output = f"{JOINED_DIR}/bucket_{bucket}.parquet"

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log(f"Skipping Bucket : {bucket}")
        return

    path_a = f"{PROCESSED_A}/bucket={bucket}"

    path_b = f"{PROCESSED_B}/bucket={bucket}"

    if not os.path.exists(path_a):
        return

    if not os.path.exists(path_b):
        return

    log(f"Joining Bucket : {bucket}")

    query = f"""
    COPY (
        SELECT

            a.join_id,

            a.{JOIN_COLUMN} AS left_{JOIN_COLUMN},

            b.{JOIN_COLUMN} AS right_{JOIN_COLUMN},

            {LEFT_SELECT},

            {RIGHT_SELECT}

        FROM read_parquet(
            '{path_a}/*.parquet'
        ) a

        {JOIN_TYPE} JOIN read_parquet(
            '{path_b}/*.parquet'
        ) b

        ON a.join_id = b.join_id
    )
    TO '{output}'
    (
        FORMAT PARQUET,
        COMPRESSION '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output}')
        """
    ).fetchone()[0]

    con.close()

    clean_memory()

    if CLEAN_AFTER_JOIN:
        safe_remove(path_a)
        safe_remove(path_b)

    log(f"Bucket Done : {bucket}")
    log(f"Rows        : {rows:,}")
    log(f"Time        : {time.time() - start:.2f} sec")


def join_all_buckets():
    log("=" * 80)
    log(f"STEP 6 -> PARALLEL {JOIN_TYPE} JOINS")
    log("=" * 80)

    start = time.time()

    valid_buckets = []

    for bucket in range(BUCKETS):
        path_a = f"{PROCESSED_A}/bucket={bucket}"

        path_b = f"{PROCESSED_B}/bucket={bucket}"

        if os.path.exists(path_a) and os.path.exists(path_b):
            valid_buckets.append(bucket)

    log(f"Buckets To Join : {len(valid_buckets)}")

    with ThreadPoolExecutor(
        max_workers=MAX_JOIN_WORKERS
    ) as executor:

        futures = [
            executor.submit(join_bucket, bucket)
            for bucket in valid_buckets
        ]

        completed = 0

        for future in as_completed(futures):
            future.result()

            completed += 1

            log(
                f"Join Progress : "
                f"{completed}/{len(valid_buckets)}"
            )

            clean_memory()

    log(
        f"All Joins Completed In : "
        f"{time.time() - start:.2f} sec"
    )


def merge_outputs():
    log("=" * 80)
    log("STEP 7 -> MERGE FINAL OUTPUT")
    log("=" * 80)

    start = time.time()

    output = f"{JOINED_DIR}/final_output.parquet"

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log("Final Output Exists -> Skipping")
        return

    con = connect()

    query = f"""
    COPY (
        SELECT *
        FROM read_parquet(
            '{JOINED_DIR}/bucket_*.parquet'
        )
    )
    TO '{output}'
    (
        FORMAT PARQUET,
        COMPRESSION '{COMPRESSION_FINAL}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output}')
        """
    ).fetchone()[0]

    con.close()

    clean_memory()

    if CLEAN_AFTER_MERGE:
        for file in Path(JOINED_DIR).glob("bucket_*.parquet"):
            safe_remove(str(file))

    log(f"Final Rows  : {rows:,}")
    log(f"Output File : {output}")
    log(f"Completed   : {time.time() - start:.2f} sec")


def cleanup_temp():
    log("=" * 80)
    log("STEP 8 -> CLEAN TEMP")
    log("=" * 80)

    safe_remove(TEMP_DIR)

    os.makedirs(TEMP_DIR, exist_ok=True)

    clean_memory()

    log("Temp Cleaned")


def main():
    total_start = time.time()

    log("=" * 80)
    log("LARGE SCALE PARQUET JOIN PIPELINE")
    log("=" * 80)

    log(f"Join Type      : {JOIN_TYPE}")

    log(f"Buckets        : {BUCKETS}")

    log(f"Threads        : {THREADS}")

    create_distinct_ids()

    create_id_map()

    validate_mapping()

    process_folder(
        RAW_A,
        PROCESSED_A,
        PROCESS_SELECT_A
    )

    process_folder(
        RAW_B,
        PROCESSED_B,
        PROCESS_SELECT_B
    )

    analyze_bucket_distribution(PROCESSED_A)

    analyze_bucket_distribution(PROCESSED_B)

    join_all_buckets()

    merge_outputs()

    cleanup_temp()

    log("=" * 80)
    log("PIPELINE COMPLETED")
    log("=" * 80)

    log(
        f"TOTAL EXECUTION TIME : "
        f"{time.time() - total_start:.2f} sec"
    )


if __name__ == "__main__":
    main()
