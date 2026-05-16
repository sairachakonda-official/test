import os
import gc
import uuid
import shutil
import duckdb
import psutil
import time
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FOLDER = "input_parquet"

# ID MAP PARQUET
ID_MAP_FILE = "id_map/id_map.parquet"

OUTPUT_FOLDER = "partitioned_output"

# JOIN CONFIG
JOIN_KEY = "client_data1"
BUCKET_COLUMN = "bucket"

# PERFORMANCE
COMPRESSION = "SNAPPY"
ROW_GROUP_SIZE = 250_000
MEMORY_LIMIT = "4GB"

# OPTIONAL COLUMN FILTERING
SELECT_COLUMNS = "*"

# TEMP
TEMP_BASE = "tmp_partition"

# LOGS
LOG_FOLDER = "logs"

# =============================================================================
# SETUP
# =============================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_BASE, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# =============================================================================
# LOG FILE
# =============================================================================

LOG_FILE = os.path.join(
    LOG_FOLDER,
    f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# =============================================================================
# LOGGING
# =============================================================================

def log(msg):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    final_msg = f"[{timestamp}] {msg}"

    print(final_msg, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:

        f.write(final_msg + "\n")

# =============================================================================
# MEMORY
# =============================================================================

def print_memory():

    mem = psutil.virtual_memory()

    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)

    log(
        f"RAM -> "
        f"Used: {used_gb:.2f} GB | "
        f"Available: {available_gb:.2f} GB | "
        f"Total: {total_gb:.2f} GB"
    )

# =============================================================================
# CLEANUP
# =============================================================================

def cleanup(con=None):

    log("Starting cleanup")

    try:

        if con:

            con.close()

            log("DuckDB connection closed")

    except Exception as e:

        log(f"Cleanup failed: {str(e)}")

    gc.collect()

    log("Garbage collection completed")

# =============================================================================
# GET FILES
# =============================================================================

def get_parquet_files(folder):

    log(f"Scanning parquet files in: {folder}")

    parquet_files = []

    for root, _, files in os.walk(folder):

        for file in files:

            if file.endswith(".parquet"):

                parquet_files.append(
                    os.path.join(root, file)
                )

    parquet_files.sort()

    log(f"Total parquet files found: {len(parquet_files)}")

    return parquet_files

# =============================================================================
# MOVE GENERATED FILES
# =============================================================================

def move_partition_files(temp_output_dir):

    log("Starting partition file move")

    total_moved = 0

    if not os.path.exists(temp_output_dir):

        log("Temp output directory missing")

        return

    for bucket_name in os.listdir(temp_output_dir):

        bucket_dir = os.path.join(
            temp_output_dir,
            bucket_name
        )

        if not os.path.isdir(bucket_dir):
            continue

        final_bucket_dir = os.path.join(
            OUTPUT_FOLDER,
            bucket_name
        )

        os.makedirs(
            final_bucket_dir,
            exist_ok=True
        )

        moved_files = 0

        for file in os.listdir(bucket_dir):

            if not file.endswith(".parquet"):
                continue

            src_file = os.path.join(
                bucket_dir,
                file
            )

            dst_file = os.path.join(
                final_bucket_dir,
                f"part_{uuid.uuid4().hex}.parquet"
            )

            shutil.move(
                src_file,
                dst_file
            )

            moved_files += 1
            total_moved += 1

        log(
            f"Moved {moved_files} parquet files -> "
            f"{final_bucket_dir}"
        )

    log(f"Total parquet files moved: {total_moved}")

# =============================================================================
# PROCESS FILE
# =============================================================================

def process_file(file_path, idx, total_files):

    file_name = os.path.basename(file_path)

    log("\n" + "=" * 80)
    log(f"[{idx}/{total_files}] START FILE")
    log(f"FILE: {file_name}")
    log("=" * 80)

    file_start = time.time()

    print_memory()

    # -------------------------------------------------------------------------
    # FILE SIZE
    # -------------------------------------------------------------------------

    try:

        file_size_gb = (
            os.path.getsize(file_path)
            / (1024 ** 3)
        )

        log(f"Input File Size: {file_size_gb:.2f} GB")

    except Exception as e:

        log(f"File size check failed: {str(e)}")

    # -------------------------------------------------------------------------
    # TEMP DIR
    # -------------------------------------------------------------------------

    worker_temp_dir = os.path.join(
        TEMP_BASE,
        uuid.uuid4().hex
    )

    log(f"Creating temp dir: {worker_temp_dir}")

    os.makedirs(
        worker_temp_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # DUCKDB
    # -------------------------------------------------------------------------

    log("Opening DuckDB connection")

    con = duckdb.connect()

    try:

        # ---------------------------------------------------------------------
        # SETTINGS
        # ---------------------------------------------------------------------

        log("Applying DuckDB settings")

        con.execute("PRAGMA threads=1")

        con.execute(
            f"PRAGMA memory_limit='{MEMORY_LIMIT}'"
        )

        con.execute(
            "PRAGMA preserve_insertion_order=false"
        )

        con.execute(
            f"PRAGMA temp_directory='{worker_temp_dir}'"
        )

        log("DuckDB settings applied")

        # ---------------------------------------------------------------------
        # VALIDATE INPUT
        # ---------------------------------------------------------------------

        log("Validating input parquet")

        validate_query = f"""
        SELECT
            COUNT(*) AS total_rows
        FROM read_parquet('{file_path}')
        """

        total_rows = con.execute(
            validate_query
        ).fetchone()[0]

        log(f"Rows Found: {total_rows:,}")

        # ---------------------------------------------------------------------
        # VALIDATE ID MAP
        # ---------------------------------------------------------------------

        log("Checking ID map")

        id_map_query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT {BUCKET_COLUMN}) AS total_buckets
        FROM read_parquet('{ID_MAP_FILE}')
        """

        id_map_result = con.execute(
            id_map_query
        ).fetchone()

        log(
            f"ID Map Rows: "
            f"{id_map_result[0]:,}"
        )

        log(
            f"ID Map Buckets: "
            f"{id_map_result[1]:,}"
        )

        # ---------------------------------------------------------------------
        # JOIN + PARTITION
        # ---------------------------------------------------------------------

        log("Starting JOIN + PARTITION")

        partition_start = time.time()

        query = f"""
        COPY (
            SELECT
                src.*,
                map.{BUCKET_COLUMN}
            FROM read_parquet('{file_path}') src
            INNER JOIN read_parquet('{ID_MAP_FILE}') map
            ON src.{JOIN_KEY} = map.{JOIN_KEY}
        )
        TO '{worker_temp_dir}'
        (
            FORMAT PARQUET,
            PARTITION_BY ({BUCKET_COLUMN}),
            PER_THREAD_OUTPUT FALSE,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE},
            COMPRESSION {COMPRESSION}
        )
        """

        con.execute(query)

        partition_elapsed = (
            time.time() - partition_start
        )

        log(
            f"JOIN + PARTITION completed in "
            f"{partition_elapsed:.2f} sec"
        )

        print_memory()

        # ---------------------------------------------------------------------
        # MOVE FILES
        # ---------------------------------------------------------------------

        move_start = time.time()

        move_partition_files(worker_temp_dir)

        move_elapsed = time.time() - move_start

        log(
            f"Move completed in "
            f"{move_elapsed:.2f} sec"
        )

        print_memory()

        # ---------------------------------------------------------------------
        # DONE
        # ---------------------------------------------------------------------

        total_elapsed = time.time() - file_start

        log(f"Completed File: {file_name}")

        log(
            f"Total File Time: "
            f"{total_elapsed:.2f} sec"
        )

    except Exception as e:

        log(f"ERROR PROCESSING FILE: {file_name}")

        log(str(e))

    finally:

        cleanup(con)

        # ---------------------------------------------------------------------
        # TEMP CLEANUP
        # ---------------------------------------------------------------------

        log("Removing temp directory")

        try:

            shutil.rmtree(worker_temp_dir)

            log(
                f"Removed temp directory: "
                f"{worker_temp_dir}"
            )

        except Exception as e:

            log(f"Temp cleanup failed: {str(e)}")

        gc.collect()

        print_memory()

        log("=" * 80)
        log(f"END FILE: {file_name}")
        log("=" * 80)

# =============================================================================
# MAIN
# =============================================================================

def main():

    total_start = time.time()

    log("=" * 80)
    log("JOIN + PARTITION PIPELINE")
    log("=" * 80)

    print_memory()

    # -------------------------------------------------------------------------
    # VALIDATE ID MAP EXISTS
    # -------------------------------------------------------------------------

    if not os.path.exists(ID_MAP_FILE):

        log(f"ID Map missing: {ID_MAP_FILE}")

        return

    log(f"Using ID Map: {ID_MAP_FILE}")

    # -------------------------------------------------------------------------
    # DISCOVER FILES
    # -------------------------------------------------------------------------

    files = get_parquet_files(INPUT_FOLDER)

    total_files = len(files)

    log(f"Files Found: {total_files}")

    if total_files == 0:

        log("No parquet files found")

        return

    # -------------------------------------------------------------------------
    # PROCESS FILES
    # -------------------------------------------------------------------------

    for idx, file_path in enumerate(files, 1):

        process_file(
            file_path=file_path,
            idx=idx,
            total_files=total_files
        )

    # -------------------------------------------------------------------------
    # FINAL STATS
    # -------------------------------------------------------------------------

    total_elapsed = time.time() - total_start

    log("\n" + "=" * 80)
    log("ALL FILES COMPLETED")
    log("=" * 80)

    log(
        f"Total Pipeline Time: "
        f"{total_elapsed:.2f} sec"
    )

    print_memory()

    log(f"Log File Saved: {LOG_FILE}")

# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":

    main()
