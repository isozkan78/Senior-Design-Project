import ee
import datetime

# Google Earth Engine Initialization
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print("GEE Initialization failed. Please authenticate.")

class DataIngestionPipeline:
    def __init__(self):
        # Coordinates for Antalya Region (Approximate Bounding Box)
        self.roi = ee.Geometry.Rectangle([29.2, 36.0, 32.5, 37.5])
        self.start_date = '2015-01-01'
        self.end_date = '2025-01-01'
        
        print(f"Region of Interest set to Antalya. Date Range: {self.start_date} - {self.end_date}")

    def get_dynamic_data(self):
        """
        Fetches dynamic variables: NDVI (Vegetation) and LST (Temperature)
        Source: MODIS (MOD13A1, MOD11A2)
        """
        print("Fetching Dynamic Data (MODIS)...")
        
        # NDVI Collection (Vegetation)
        ndvi_col = ee.ImageCollection('MODIS/006/MOD13A1') \
            .filterDate(self.start_date, self.end_date) \
            .filterBounds(self.roi) \
            .select('NDVI')
            
        # LST Collection (Temperature)
        lst_col = ee.ImageCollection('MODIS/006/MOD11A2') \
            .filterDate(self.start_date, self.end_date) \
            .filterBounds(self.roi) \
            .select('LST_Day_1km')
            
        print(f"NDVI Images Found: {ndvi_col.size().getInfo()}")
        print(f"LST Images Found: {lst_col.size().getInfo()}")
        return ndvi_col, lst_col

    def get_static_data(self):
        """
        Fetches static variables: Elevation, Slope
        Source: NASA SRTM
        """
        print("Fetching Static Data (SRTM)...")
        srtm = ee.Image('USGS/SRTMGL1_003').clip(self.roi)
        elevation = srtm.select('elevation')
        slope = ee.Terrain.slope(elevation)
        
        return elevation, slope

    def get_fire_labels(self):
        """
        Fetches Ground Truth: Active Fire Points
        Source: NASA FIRMS
        """
        print("Fetching Active Fire Data (FIRMS)...")
        firms = ee.ImageCollection('FIRMS') \
            .filterDate(self.start_date, self.end_date) \
            .filterBounds(self.roi)
            
        print(f"Fire Data Points Loaded.")
        return firms

if __name__ == "__main__":
    # Testing pipeline
    pipeline = DataIngestionPipeline()
    ndvi, lst = pipeline.get_dynamic_data()
    elev, slope = pipeline.get_static_data()
    fire_data = pipeline.get_fire_labels()
    
    print("Data Ingestion Pipeline Test Complete.")