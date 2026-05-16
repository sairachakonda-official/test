import gc
import os
import shutil
import threading
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

BUCKETS = max(32, os.cpu_count() * 4)


# =========================================================
# PERFORMANCE CONFIG
# =========================================================

THREADS = max(1, os.cpu_count() // 2)

MEMORY_LIMIT = "120GB"

ROW_GROUP_SIZE = 250000

COMPRESSION_INTERMEDIATE = "snappy"

COMPRESSION_FINAL = "zstd"

MAX_PROCESS_WORKERS = 2

MAX_JOIN_WORKERS = 2

SKIP_EXISTING = True

CLEAN_AFTER_JOIN = True

CLEAN_AFTER_MERGE = False

GC_INTERVAL = 20


# =========================================================
# COLUMN FILTERS
# =========================================================

PROCESS_COLUMNS_A = None
PROCESS_COLUMNS_B = None

LEFT_COLUMNS = None
RIGHT_COLUMNS = None


# =========================================================
# DROP CONFIG
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
# CREATE DIRS
# =========================================================

for path in [
    MAPPING_DIR,
    PROCESSED_A,
    PROCESSED_B,
    JOINED_DIR,
    TEMP_DIR
]:
    os.makedirs(path, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================

def log(message):

    print(
        f"[{time.strftime('%H:%M:%S')}] {message}"
    )


# =========================================================
# THREAD LOCAL CONNECTION
# =========================================================

thread_local = threading.local()


def get_connection():

    if hasattr(thread_local, "con"):
        return thread_local.con

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
        "PRAGMA disable_object_cache"
    )

    thread_local.con = con

    return con


def close_thread_connection():

    try:

        if hasattr(thread_local, "con"):

            try:
                thread_local.con.close()
            except:
                pass

            del thread_local.con

    except:
        pass


# =========================================================
# HELPERS
# =========================================================

def parquet_files(folder):

    return sorted(
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.endswith(".parquet")
    )


def parquet_exists(path):

    if not os.path.exists(path):
        return False

    for _, _, files in os.walk(path):

        for file in files:

            if file.endswith(".parquet"):
                return True

    return False


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


# =========================================================
# MEMORY
# =========================================================

def clean_memory(*variables):

    try:

        for var in variables:

            try:
                del var
            except:
                pass

        gc.collect()

    except:
        pass


# =========================================================
# SCHEMA CACHE
# =========================================================

SCHEMA_CACHE = {}


def get_schema(file_path):

    if file_path in SCHEMA_CACHE:
        return SCHEMA_CACHE[file_path]

    con = get_connection()

    columns = con.execute(
        f"""
        SELECT *
        FROM read_parquet('{file_path}')
        LIMIT 0
        """
    ).df().columns.tolist()

    SCHEMA_CACHE[file_path] = columns

    return columns


# =========================================================
# DROP CHECK
# =========================================================

def should_drop_column(column_name):

    for value in DROP_COLUMNS:

        if column_name == value:
            return True

        if column_name.startswith(value):
            return True

    return False


# =========================================================
# FILTER COLUMNS
# =========================================================

def get_filtered_columns(
    file_path,
    keep_columns=None
):

    all_columns = get_schema(file_path)

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
# BUILD SELECT
# =========================================================

def build_join_select(
    columns,
    alias,
    prefix=None
):

    result = []

    seen = set()

    for col in columns:

        if col in ["join_id", "bucket"]:
            continue

        mapped = (
            f"{prefix}{col}"
            if prefix
            else col
        )

        if should_drop_column(mapped):
            continue

        if mapped in seen:
            continue

        seen.add(mapped)

        result.append(
            f"{alias}.{col} AS {mapped}"
        )

    return ",\n".join(result)


# =========================================================
# STEP 1
# =========================================================

def create_distinct_ids():

    log("=" * 80)
    log("STEP 1 -> DISTINCT IDS")
    log("=" * 80)

    output = (
        f"{MAPPING_DIR}/distinct_ids.parquet"
    )

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log("Distinct IDs Exist")
        return

    con = get_connection()

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
        COMPRESSION '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    log("Distinct IDs Completed")


# =========================================================
# STEP 2
# =========================================================

def create_id_map():

    log("=" * 80)
    log("STEP 2 -> ID MAP")
    log("=" * 80)

    output = f"{MAPPING_DIR}/id_map.parquet"

    if (
        SKIP_EXISTING
        and file_exists(output)
    ):
        log("ID Map Exists")
        return

    con = get_connection()

    query = f"""
    COPY (
        SELECT

            {JOIN_COLUMN},

            abs(hash({JOIN_COLUMN}))
            ::BIGINT AS join_id

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

    log("ID Map Completed")


# =========================================================
# STEP 3
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

    columns = get_filtered_columns(
        file_path=file_path,
        keep_columns=keep_columns
    )

    select_sql = ",\n".join(
        f"t.{col}"
        for col in columns
    )

    con = get_connection()

    query = f"""
    COPY (
        SELECT

            {select_sql},

            m.join_id,

            abs(hash(m.join_id))
            % {BUCKETS} AS bucket

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

    clean_memory(
        columns,
        select_sql,
        query
    )

    log(
        f"Completed : {filename} "
        f"({time.time() - start:.2f} sec)"
    )


# =========================================================
# STEP 4
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

            if completed % GC_INTERVAL == 0:
                gc.collect()

            log(
                f"Progress : "
                f"{completed}/{len(files)}"
            )

    clean_memory(
        files,
        futures
    )


# =========================================================
# STEP 5
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

    files_a = parquet_files(path_a)
    files_b = parquet_files(path_b)

    if not files_a or not files_b:
        return

    left_columns = get_filtered_columns(
        files_a[0],
        LEFT_COLUMNS
    )

    right_columns = get_filtered_columns(
        files_b[0],
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

    con = get_connection()

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
        COMPRESSION '{COMPRESSION_INTERMEDIATE}',
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    )
    """

    con.execute(query)

    if CLEAN_AFTER_JOIN:

        safe_remove(path_a)

        safe_remove(path_b)

    clean_memory(
        left_columns,
        right_columns,
        left_select,
        right_select,
        query,
        files_a,
        files_b
    )

    log(
        f"Bucket {bucket} Done "
        f"({time.time() - start:.2f} sec)"
    )


# =========================================================
# STEP 6
# =========================================================

def join_all_buckets():

    log("=" * 80)
    log("STEP 6 -> JOIN BUCKETS")
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

            if completed % GC_INTERVAL == 0:
                gc.collect()

            log(
                f"Join Progress : "
                f"{completed}/"
                f"{len(valid_buckets)}"
            )

    clean_memory(
        valid_buckets,
        futures
    )


# =========================================================
# STEP 7
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

    con = get_connection()

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
# CLEANUP
# =========================================================

def cleanup():

    log("=" * 80)
    log("FINAL CLEANUP")
    log("=" * 80)

    try:

        if os.path.exists(TEMP_DIR):

            for item in os.listdir(TEMP_DIR):

                safe_remove(
                    os.path.join(TEMP_DIR, item)
                )

    except Exception as e:

        log(str(e))

    close_thread_connection()

    SCHEMA_CACHE.clear()

    gc.collect()

    log("Cleanup Completed")


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

    cleanup()

    log("=" * 80)
    log("PIPELINE COMPLETED")
    log("=" * 80)

    log(
        f"TOTAL TIME : "
        f"{time.time() - total_start:.2f} sec"
    )


if __name__ == "__main__":

    main()
