import gc
import json
import logging
import math
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
RASTER_PATH     = "./data/input/brazil_coverage_2024.tif"
OUTPUT_DIR      = "./data/output"
OUTPUT_FILE     = "brasil_coverage.gpkg"
CHECKPOINT_FILE = "checkpoint_brasil_coverage.json"

BLOCK_LENGTH   = 2048   # pixels por bloco (2 048 × 2 048 ~16 MB por banda uint8)
BATCH_BLOCKS   = 25     
NUM_CORES      = 4
MAX_IN_FLIGHT  = NUM_CORES * 3   # janela deslizante

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------
def process_block(args):
    raster_path, col_off, row_off, window_width, window_height = args
    results = []
    
    try:
        with rasterio.open(raster_path) as src:
            window = Window(col_off, row_off, window_width, window_height)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            nodata = src.nodata
            
            if nodata is not None:
                mask = (data != nodata).astype("uint8")
                if not mask.any():
                    return results
            
            else:
                mask = None
            
            for geom, value in shapes(data, mask=mask, transform=transform):
                results.append({"geometry": shape(geom), "value": int(value)})
            
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).error(
            "Erro no bloco (%d, %d): %s", col_off, row_off, e
        )
    
    return results

def write_batch(geometries, output_path, crs, first_write):
    if not geometries:
        return

    gdf = gpd.GeoDataFrame(geometries, crs=crs)
    mode = 'w' if first_write else 'a'
    gdf.to_file(output_path, driver="GPKG", mode=mode)
    del gdf
    gc.collect()

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()

def save_checkpoint(path, done):
    with open(path, "w") as f:
        json.dump(sorted(done), f)
    

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg = str(output_dir / OUTPUT_FILE)
    checkpoint_path = str(output_dir / CHECKPOINT_FILE)

    t0 = datetime.now()
    log.info("Iniciando — %s", t0.strftime("%Y-%m-%d %H:%M:%S"))
    
    with rasterio.open(RASTER_PATH) as src:
        width_total  = src.width
        height_total = src.height
        crs          = src.crs

    n_cols_blk = math.ceil(width_total  / BLOCK_LENGTH)
    n_rows_blk = math.ceil(height_total / BLOCK_LENGTH)
    total_blk  = n_cols_blk * n_rows_blk

    log.info("Raster: %d × %d px  |  grade: %d × %d  |  total: %d blocos", width_total, height_total, n_rows_blk, n_cols_blk, total_blk)

    # --- checkpoint: retoma onde parou ----------------------------------------
    done_ids = load_checkpoint(checkpoint_path)
    resuming = bool(done_ids)
    
    if resuming:
        log.info("Retomando do checkpoint: %d blocos já concluídos.", len(done_ids))
    else:
        if os.path.exists(output_gpkg):
            os.remove(output_gpkg)
            
    pending_tasks = []
    for row_off in range(0, height_total, BLOCK_LENGTH):
        for col_off in range(0, width_total, BLOCK_LENGTH):
            bid = f"{col_off}_{row_off}"
            if bid in done_ids:
                continue
            window_width = min(BLOCK_LENGTH, width_total  - col_off)
            window_height = min(BLOCK_LENGTH, height_total - row_off)
            pending_tasks.append((RASTER_PATH, col_off, row_off, window_width, window_height, bid))
    
    remaining = len(pending_tasks)
    log.info("Tarefas pendentes: %d  |  processos: %d  |  janela in-flight: %d", remaining, NUM_CORES, MAX_IN_FLIGHT)

    if remaining == 0:
        log.info("Nada a fazer — todos os blocos já foram processados.")
        return

    # --- processamento ----------------------------
    processed = len(done_ids)
    batch_geom = []
    batch_count = 0
    first_write = not remaining
    task_iter = iter(pending_tasks)
    
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {}
        for task in task_iter:
            if len(futures) >= MAX_IN_FLIGHT:
                break
            bid = task[-1]
            args = task[:-1]
            future = executor.submit(process_block, args)
            futures[future] = bid

        while futures:
            done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
            
            for future in done_set:
                bid = futures.pop(future)
                processed += 1
                batch_count += 1
                
                try:
                    block_geom = future.result()
                    if block_geom:
                        batch_geom.extend(block_geom)
                except Exception as e:
                    log.error("Future falhou (bloco %s): %s", bid, e)
                
                done_ids.add(bid)
                
                if batch_count >= BATCH_BLOCKS:
                    write_batch(batch_geom, output_gpkg, crs, first_write)
                    save_checkpoint(checkpoint_path, done_ids)
                    log.info(
                        "Escrita — %d/%d blocos  |  lote: %d geoms  |  "
                        "arquivo: %.1f MB",
                        processed, total_blk, len(batch_geoms),
                        os.path.getsize(output_gpkg) / 1_048_576
                        if os.path.exists(output_gpkg) else 0,
                    )
                    batch_geoms  = []
                    batch_count  = 0
                    first_write  = False
                
                # --- submete próxima tarefa -------------
                try:
                    next_task = next(task_iter)
                    next_bid  = next_task[-1]
                    next_args = next_task[:-1]
                    nf        = executor.submit(process_block, next_args)
                    futures[nf] = next_bid
                except StopIteration:
                    pass
    
    # --- flush do último lote ---------------------------------------------------
    if batch_geoms:
        write_batch(batch_geoms, output_gpkg, crs, first_write)
        save_checkpoint(checkpoint_path, done_ids)
        log.info("Último lote escrito: %d geoms.", len(batch_geoms))

    # --- remove checkpoint ao finalizar com sucesso -----------------------------
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        log.info("Checkpoint removido.")

    elapsed = datetime.now() - t0
    size_mb = os.path.getsize(output_gpkg) / 1_048_576 if os.path.exists(output_gpkg) else 0
    log.info("Concluído!  Tempo total: %s  |  Arquivo: %.1f MB  → %s", elapsed, size_mb, output_gpkg)


if __name__ == "__main__":
    main()
    