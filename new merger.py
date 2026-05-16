import gc
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb


# =========================================================
# PATHS
# =========================================================

RAW_A = "project/raw_a"

RAW_B = "project/raw_b"

MAPPING_DIR = "project/mapping"

PROCESSED_A = "project/processed_a"

PROCESSED_B = "project/processed_b"

JOINED_DIR = "project/joined"

TEMP_DIR = "project/temp"


# =========================================================
# JOIN CONFIG
# =========================================================

JOIN_COLUMN = "id"

JOIN_TYPE = "INNER"

BUCKETS = 256


# =========================================================
# PERFORMANCE CONFIG
# =========================================================

THREADS = max(1, os.cpu_count() // 2)

MEMORY_LIMIT = "120GB"

ROW_GROUP_SIZE = 100000

COMPRESSION_INTERMEDIATE = "snappy"

COMPRESSION_FINAL = "zstd"

MAX_JOIN_WORKERS = min(16, THREADS)

MAX_PROCESS_WORKERS = min(8, THREADS)

SKIP_EXISTING = True

CLEAN_AFTER_JOIN = True

CLEAN_AFTER_MERGE = False


# =========================================================
# KEEP COLUMNS
# None = keep all
# =========================================================

PROCESS_COLUMNS_A = None

PROCESS_COLUMNS_B = None

LEFT_COLUMNS = None

RIGHT_COLUMNS = None


# =========================================================
# SINGLE DROP CONFIG
# SUPPORTS:
# - EXACT MATCH
# - PREFIX MATCH
# =========================================================

DROP_COLUMNS = [
    "left_id",
    "right_id",
    "a_meta_",
    "b_raw_",
    "a_temp_col",
    "b_unused_col",
    "tmp_",
    "test_"
]


# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(MAPPING_DIR, exist_ok=True)

os.makedirs(PROCESSED_A, exist_ok=True)

os.makedirs(PROCESSED_B, exist_ok=True)

os.makedirs(JOINED_DIR, exist_ok=True)

os.makedirs(TEMP_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def parquet_exists(path):
    if not os.path.exists(path):
        return False

    for _, _, files in os.walk(path):

        for file in files:

            if file.endswith(".parquet"):
                return True

    return False


def parquet_files(folder):
    return sorted(
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.endswith(".parquet")
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

    con.execute(
        f"SET memory_limit='{MEMORY_LIMIT}'"
    )

    con.execute(
        f"SET threads={THREADS}"
    )

    con.execute(
        f"SET temp_directory='{TEMP_DIR}'"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    con.execute(
        "PRAGMA enable_object_cache"
    )

    return con


# =========================================================
# COLUMN DROP CHECKER
# =========================================================

def should_drop_column(column_name):

    for value in DROP_COLUMNS:

        # EXACT MATCH

        if column_name == value:
            return True

        # PREFIX MATCH

        if column_name.startswith(value):
            return True

    return False


# =========================================================
# FILTER SOURCE COLUMNS
# =========================================================

def get_filtered_columns(
    file_path,
    keep_columns=None
):
    con = connect()

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{file_path}')
        """
    ).fetchall()

    con.close()

    all_columns = [row[0] for row in schema]

    filtered = []

    for col in all_columns:

        if should_drop_column(col):
            continue

        filtered.append(col)

    if keep_columns:

        keep_set = set(keep_columns)

        filtered = [
            col
            for col in filtered
            if col in keep_set
        ]

    if JOIN_COLUMN not in filtered:
        filtered.insert(0, JOIN_COLUMN)

    return filtered


# =========================================================
# BUILD JOIN SELECT
# =========================================================

def build_join_select(
    columns,
    alias,
    prefix=None
):
    cleaned = []

    seen = set()

    for col in columns:

        if col in ["join_id", "bucket"]:
            continue

        mapped_name = (
            f"{prefix}{col}"
            if prefix
            else col
        )

        if should_drop_column(mapped_name):
            continue

        if mapped_name not in seen:

            seen.add(mapped_name)

            cleaned.append(
                f"{alias}.{col} "
                f"AS {mapped_name}"
            )

    return ",\n".join(cleaned)


# =========================================================
# STEP 1 -> DISTINCT IDS
# =========================================================

def create_distinct_ids():
    log("=" * 80)

    log("STEP 1 -> CREATE DISTINCT IDS")

    log("=" * 80)

    output = (
        f"{MAPPING_DIR}/distinct_ids.parquet"
    )

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log("Distinct IDs Exist -> Skipping")

        return

    con = connect()

    query = f"""
    COPY (
        SELECT DISTINCT {JOIN_COLUMN}
        FROM (
            SELECT {JOIN_COLUMN}
            FROM read_parquet(
                '{RAW_A}/*.parquet'
            )

            UNION ALL

            SELECT {JOIN_COLUMN}
            FROM read_parquet(
                '{RAW_B}/*.parquet'
            )
        )
        WHERE {JOIN_COLUMN} IS NOT NULL
    )
    TO '{output}'
    (
        FORMAT PARQUET,
        COMPRESSION
        '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    con.close()

    clean_memory()

    log("Distinct IDs Completed")


# =========================================================
# STEP 2 -> ID MAP
# =========================================================

def create_id_map():
    log("=" * 80)

    log("STEP 2 -> CREATE ID MAP")

    log("=" * 80)

    output = f"{MAPPING_DIR}/id_map.parquet"

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log("ID Map Exists -> Skipping")

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
        COMPRESSION
        '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    con.close()

    clean_memory()

    log("ID Map Completed")


# =========================================================
# STEP 3 -> PROCESS FILE
# =========================================================

def process_file(
    file_path,
    output_dir,
    keep_columns=None
):
    start = time.time()

    filename = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    output_path = (
        f"{output_dir}/{filename}"
    )

    if (
        SKIP_EXISTING
        and parquet_exists(output_path)
    ):
        log(f"Skipping : {filename}")

        return

    log(f"Processing : {filename}")

    selected_columns = get_filtered_columns(
        file_path=file_path,
        keep_columns=keep_columns
    )

    select_sql = ",\n".join(
        f"t.{col}"
        for col in selected_columns
    )

    con = connect()

    query = f"""
    COPY (
        SELECT

            {select_sql},

            m.join_id,

            CASE
                WHEN m.join_id IS NULL
                THEN -1
                ELSE abs(hash(m.join_id))
                     % {BUCKETS}
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
        COMPRESSION
        '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE},
        OVERWRITE_OR_IGNORE
    )
    """

    con.execute(query)

    con.close()

    clean_memory()

    log(
        f"Completed : {filename} "
        f"({time.time() - start:.2f} sec)"
    )


# =========================================================
# STEP 4 -> PROCESS FOLDER
# =========================================================

def process_folder(
    input_folder,
    output_folder,
    keep_columns=None
):
    log("=" * 80)

    log(f"PROCESSING : {input_folder}")

    log("=" * 80)

    files = parquet_files(input_folder)

    with ThreadPoolExecutor(
        max_workers=MAX_PROCESS_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_file,
                file,
                output_folder,
                keep_columns
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


# =========================================================
# STEP 5 -> JOIN BUCKET
# =========================================================

def join_bucket(bucket):
    start = time.time()

    output = (
        f"{JOINED_DIR}/bucket_{bucket}.parquet"
    )

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        return

    path_a = (
        f"{PROCESSED_A}/bucket={bucket}"
    )

    path_b = (
        f"{PROCESSED_B}/bucket={bucket}"
    )

    if not os.path.exists(path_a):
        return

    if not os.path.exists(path_b):
        return

    sample_a = os.path.join(
        path_a,
        os.listdir(path_a)[0]
    )

    sample_b = os.path.join(
        path_b,
        os.listdir(path_b)[0]
    )

    left_columns = get_filtered_columns(
        sample_a,
        LEFT_COLUMNS
    )

    right_columns = get_filtered_columns(
        sample_b,
        RIGHT_COLUMNS
    )

    left_select = build_join_select(
        left_columns,
        "a",
        prefix="a_"
    )

    right_select = build_join_select(
        right_columns,
        "b",
        prefix="b_"
    )

    con = connect()

    query = f"""
    COPY (
        SELECT

            a.join_id,

            {left_select},

            {right_select}

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
        COMPRESSION
        '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    con.close()

    clean_memory()

    if CLEAN_AFTER_JOIN:

        safe_remove(path_a)

        safe_remove(path_b)

    log(
        f"Bucket {bucket} Done "
        f"({time.time() - start:.2f} sec)"
    )


# =========================================================
# STEP 6 -> JOIN ALL BUCKETS
# =========================================================

def join_all_buckets():
    log("=" * 80)

    log("STEP 6 -> JOIN ALL BUCKETS")

    log("=" * 80)

    valid_buckets = []

    for bucket in range(BUCKETS):

        path_a = (
            f"{PROCESSED_A}/bucket={bucket}"
        )

        path_b = (
            f"{PROCESSED_B}/bucket={bucket}"
        )

        if (
            os.path.exists(path_a)
            and os.path.exists(path_b)
        ):
            valid_buckets.append(bucket)

    with ThreadPoolExecutor(
        max_workers=MAX_JOIN_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                join_bucket,
                bucket
            )
            for bucket in valid_buckets
        ]

        completed = 0

        for future in as_completed(futures):

            future.result()

            completed += 1

            log(
                f"Join Progress : "
                f"{completed}/"
                f"{len(valid_buckets)}"
            )


# =========================================================
# STEP 7 -> MERGE OUTPUTS
# =========================================================

def merge_outputs():
    log("=" * 80)

    log("STEP 7 -> MERGE OUTPUTS")

    log("=" * 80)

    output = (
        f"{JOINED_DIR}/final_output.parquet"
    )

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log("Final Output Exists")

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
        COMPRESSION
        '{COMPRESSION_FINAL}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    con.close()

    clean_memory()

    if CLEAN_AFTER_MERGE:

        for file in os.listdir(JOINED_DIR):

            if (
                file.startswith("bucket_")
                and file.endswith(".parquet")
            ):
                safe_remove(
                    os.path.join(
                        JOINED_DIR,
                        file
                    )
                )

    log("Final Merge Completed")


# =========================================================
# STEP 8 -> CLEAN TEMP
# =========================================================

def cleanup_temp():
    safe_remove(TEMP_DIR)

    os.makedirs(TEMP_DIR, exist_ok=True)

    clean_memory()

    log("Temp Cleaned")


# =========================================================
# MAIN
# =========================================================

def main():
    total_start = time.time()

    log("=" * 80)

    log("LARGE SCALE PARQUET JOIN PIPELINE")

    log("=" * 80)

    create_distinct_ids()

    create_id_map()

    process_folder(
        RAW_A,
        PROCESSED_A,
        PROCESS_COLUMNS_A
    )

    process_folder(
        RAW_B,
        PROCESSED_B,
        PROCESS_COLUMNS_B
    )

    join_all_buckets()

    merge_outputs()

    cleanup_temp()

    log("=" * 80)

    log("PIPELINE COMPLETED")

    log("=" * 80)

    log(
        f"TOTAL TIME : "
        f"{time.time() - total_start:.2f} sec"
    )


if __name__ == "__main__":
    main()
