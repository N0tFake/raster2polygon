import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import os
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Desativa warnings desnecessários do shapely/geopandas durante o append
warnings.filterwarnings("ignore")

def process_block(args):
    raster_path, col_off, row_off, window_width, window_height = args
    
    geometries_data = []
    
    with rasterio.open(raster_path) as src:
        window = Window(col_off, row_off, window_width, window_height)
        block_matriz = src.read(1, window=window)
        transform_window = src.window_transform(window)
        nodata = src.nodata
        
        mask = (block_matriz != nodata) if nodata is not None else None
        
        if mask is not None and not mask.any():
            return geometries_data
            
        generate_feicoes = shapes(
            block_matriz,
            mask=mask,
            transform=transform_window
        )
        
        for geom, value in generate_feicoes:
            geometries_data.append({'geometry': shape(geom), 'value': value})
            
    return geometries_data

def main():
    raster_path = "./data/input/brazil_coverage_2024.tif"
    output_path = './data/output/'
    os.makedirs(output_path, exist_ok=True)

    name_output_file = os.path.join(output_path, "brasil_coverage.gpkg") 
    
    block_length = 8196
    now = datetime.now()
    print(f"Iniciando os trabalhos: {now}")
    
    with rasterio.open(raster_path) as src:
        width_total = src.width
        height_total = src.height
        crs = src.crs
    
    colunas_matriz = math.ceil(width_total / block_length)
    linhas_matriz = math.ceil(height_total / block_length)
    total_blocks = colunas_matriz * linhas_matriz
    
    print(f"Raster dimensions: {width_total}x{height_total} pixels")
    print(f"Matriz dimension: [{linhas_matriz}, {colunas_matriz}]")
    print(f"Total Blocks: {total_blocks}")

    tasks = []
    for row_off in range(0, height_total, block_length):
        for col_off in range(0, width_total, block_length):
            window_width = min(block_length, width_total - col_off)
            window_height = min(block_length, height_total - row_off)
            tasks.append((raster_path, col_off, row_off, window_width, window_height))

    blocos_processados = 0
    # num_cores = max(1, os.cpu_count() - 1)
    
    num_cores = 3
    
    print(f"Iniciando pool com {num_cores} processos...")
    
    if os.path.exists(name_output_file):
        os.remove(name_output_file)
        
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_block, task): task for task in tasks}
        
        for future in as_completed(futures):
            blocos_processados += 1
            resultados_bloco = future.result()
            
            if resultados_bloco:
                gdf = gpd.GeoDataFrame(resultados_bloco, crs=crs)
                
                gdf.to_file(name_output_file, driver="GPKG", mode="a")
            
            if blocos_processados % 10 == 0 or blocos_processados == total_blocks:
                print(f"Progresso: {blocos_processados}/{total_blocks} blocos concluídos.")

    print(f"Cabô!!! Tempo total: {datetime.now() - now}")

if __name__ == "__main__":
    main()