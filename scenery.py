import utm
import os
import numpy as np
from osgeo import gdal, osr
from owslib.wms import WebMapService
import elevation
from pyproj import CRS
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import shapely
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, track
import yaml
import osmnx as ox

# TODO
# Check google and bing/tiles works well
# Color matching - use photopea
# Runways
# finish prog bars

def ii(x):
    return np.array(Image.open(x))

def make_bbox(long0, lat0, lat1, long1):
    return Polygon([[long0, lat0],
                    [long1,lat0],
                    [long1,lat1],
                    [long0, lat1]])

def stopPrint(func, *args, **kwargs):
    with open(os.devnull,"w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 


def calculate_utm_and_latlon_corners(lat, lon, x_tiles, y_tiles, tile_size):
    """
    This function takes a latitude/longitude pair and calculates the UTM coordinates
    and the corresponding latitude/longitude for the top left and bottom right corners
    of a box of x_tiles by y_tiles, with each tile being tile_size square meters.
    """
    # Convert the center point latitude and longitude to UTM coordinates
    center_easting, center_northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    
    # Calculate the total width and height of the box
    total_width = tile_size * x_tiles
    total_height = tile_size * y_tiles
    
    # Calculate half the width and half the height for the offsets
    half_width = total_width / 2
    half_height = total_height / 2
    
    # Calculate the top left corner UTM coordinates
    top_left_easting = center_easting - half_width
    top_left_northing = center_northing + half_height
    
    # Calculate the bottom right corner UTM coordinates
    bottom_right_easting = center_easting + half_width
    bottom_right_northing = center_northing - half_height

    # Convert the top left corner UTM coordinates back to lat/lon
    top_left_lat, top_left_lon = utm.to_latlon(top_left_easting, top_left_northing, zone_number, zone_letter)
    
    # Convert the bottom right corner UTM coordinates back to lat/lon
    bottom_right_lat, bottom_right_lon = utm.to_latlon(bottom_right_easting, bottom_right_northing, zone_number, zone_letter)
    
    return {
        'utm': {
            'top_left': {'northing': top_left_northing, 'easting': top_left_easting, 'zone': f"{zone_number}{zone_letter}"},
            'bottom_right': {'northing': bottom_right_northing, 'easting': bottom_right_easting, 'zone': f"{zone_number}{zone_letter}"}
        },
        'latlon': {
            'top_left': (top_left_lat, top_left_lon),
            'bottom_right': (bottom_right_lat, bottom_right_lon)
        }
    }


def lat_lon_to_tile_coords(lat, lon, zoom):
    """
    Convert latitude and longitude to OSM tile coordinates for a given zoom level.
    
    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        zoom (int): Zoom level.

    Returns:
        tuple: A tuple (x, y) representing tile coordinates.
    """
    # Number of tiles on the X axis at the given zoom level
    n = 2 ** zoom
    
    # X tile number
    x_tile = int((lon + 180) / 360 * n)
    
    # Y tile number
    lat_rad = math.radians(lat)
    y_tile = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
    
    return x_tile, y_tile

def tile_coords_to_lat_lon(x, y, zoom):
    """
    Convert OSM tile coordinates to the top-left corner latitude and longitude.
    
    Args:
        x (int): X tile coordinate.
        y (int): Y tile coordinate.
        zoom (int): Zoom level.

    Returns:
        tuple: A tuple (lat, lon) representing the latitude and longitude of the top-left corner of the tile.
    """
    n = 2 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    
    return (lat_deg, lon_deg)

def coords_translate(x, y, from_crs, to_crs):
    src = osr.SpatialReference()
    tgt = osr.SpatialReference()
    src.ImportFromEPSG(from_crs)
    tgt.ImportFromEPSG(to_crs)

    transform = osr.CoordinateTransformation(src, tgt)
    coords = transform.TransformPoint(y, x) # CHECK
    x,y = coords[0:2]
    return x, y

def stitch_tiles(tiles, tile_width, output_size=8192):
    rows = len(tiles)
    cols = len(tiles[0])
    
    # stitched_image = Image.new('RGB', (cols*tile_width, rows*tile_width))
    stitched_image = Image.new('RGB', (output_size, output_size))

    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            data = Image.open(tile)
            data.resize((tile_width//2, tile_width//2))
            stitched_image.paste(data, (i*tile_width//2, j*tile_width//2))
            data = None
            
    stitched_image.resize((output_size,output_size))
    return stitched_image

def save_nothing(name, size=8192):
    nothing = Image.new("RGB", (size, size))
    nothing.save(name)

class Scenery(object):
    def from_yaml(yaml_file):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)['scenery']
            scenery = Scenery(config['center'], config['tiles']['x'], config['tiles']['y'], output=os.getcwd())
            try:
                scenery.wms = config['wms']
            except:
                pass
            return scenery

    def __init__(self, center, tiles_x=4, tiles_y=4, output="."):
        self.output = output
        self.center = center
        self.tiles_x = tiles_x
        self.tiles_y = tiles_x
        self.tile_size = 23040
        self.corners = calculate_utm_and_latlon_corners(center['lat'], 
                                                        center['lon'], 
                                                        self.tiles_x, 
                                                        self.tiles_x, 
                                                        self.tile_size)
        self.raw_dem = f"{self.output}/dem.tif"
        self.reproj_dem = f"{self.output}/dem_reproj.tif"
        self.zone = int(self.corners['utm']['top_left']['zone'][:-1])
        self.heights = None
        self.wms = None

    def generate(self):
        with Progress(
            SpinnerColumn(),  # Spinner column for the main task
            TextColumn("[bold blue]{task.description}"),  # Task description
            BarColumn(bar_width=50),  # Progress bar for the sub-tasks
            "{task.percentage:>3.0f}%",  # Percentage completed for the sub-tasks
            expand=False
        ) as progress:

            # Main task spinner
            scenery_task = progress.add_task("[green]Generating scenery...", total=None)

            # Sub-task progress bars
            height_data_task = progress.add_task("Fetching height data", total=100)
            height_maps_task = progress.add_task("Making height maps", total=100)
            scenery_images_task = progress.add_task("Fetching scenery images", total=100)
            # textures_task = progress.add_task("Making textures", total=100)

            # for i in range(200):
            # while not progress.finished:
            updater = lambda x: progress.update(height_data_task, advance=x)
            self._fetch(prog=updater)
        # for i in range(200):
            updater = lambda x: progress.update(height_data_task, advance=x)
            self._reproj(prog=updater)
            updater = lambda x: progress.update(scenery_images_task, advance=x)
            self.imaging_wms(prog=updater)
            progress.update(scenery_task, completed=100)


    def _fetch(self, prog=None):
        self.raw_dem = f"{self.output}/raw_dem.tif"
        corners = self.corners
        bounds = (corners['latlon']['bottom_right'][0],
            corners['latlon']['top_left'][1],
            corners['latlon']['top_left'][0],
            corners['latlon']['bottom_right'][1]
        )
        if prog is not None:
            prog(20)
        # Add margin around boundary for reprojection issues`
        bounds = (bounds[1]-0.5, bounds[0]-0.5, bounds[3]+0.5, bounds[2]+0.5)
        elevation.clip(bounds=bounds, output=self.raw_dem)
        if prog is not None:
            prog(80)

    # Reproject and clip
    def _reproj(self, prog=None):
        self.reproj_dem = f"{self.output}/reproj_dem.tif"
        input_raster = gdal.Open(self.raw_dem)
        epsg = f"EPSG:{self._out_proj()}"
        gdal.Warp(f"{self.output}/unclipped.tif", input_raster, dstSRS=epsg, xRes=30, yRes=30)
        
            
        west, south, east, north = [self.corners['utm']['top_left']['easting'], 
                                    self.corners['utm']['bottom_right']['northing'], 
                                    self.corners['utm']['bottom_right']['easting'], 
                                    self.corners['utm']['top_left']['northing']]
        ds = gdal.Open(f"{self.output}/unclipped.tif")
        if prog is None:
            prog = lambda x: None
        prog(33)
        ds = gdal.Translate(self.reproj_dem, ds, projWin = [west, north, east, south])        
        prog(33)
        ds = gdal.Translate(self.reproj_dem.replace("tif", "bil"), ds, projWin = [west, north, east, south])   
        self.heights = ii(self.reproj_dem)
        prog(67)
        
    def _out_proj(self):
        epsg = CRS.from_dict({'proj': 'utm', 'zone': self.corners['utm']['top_left']['zone'][:-1], 
                              "south": self.corners['latlon']['top_left'][0]<0})
        return int(epsg.to_authority()[1])

    def save_trn(self, dest_file):
        heightmaps = os.path.dirname(dest_file) + "/HeightMaps"
        if not os.path.exists(heightmaps):
            os.mkdir(heightmaps)
        with open(dest_file, "wb") as f:
            f.write(struct.pack('h', (x_tiles*768)//3))
            f.write(struct.pack('h', 0))
            f.write(struct.pack('h', (y_tiles*768)//3))
            f.write(struct.pack('h', 0))
            f.write(struct.pack('f', 90))
            f.write(struct.pack('f', -90))
            f.write(struct.pack('f', 90))
            # f.write(struct.pack('h', 17076))
            # f.write(struct.pack('h', 0))
            f.write(struct.pack('f', float(int(self.corners['utm']['bottom_right']['easting']))))
            f.write(struct.pack('f', float(int(self.corners['utm']['bottom_right']['northing']))))
            f.write(struct.pack('h', self.zone))
            f.write(struct.pack('h', 0))
            f.write(struct.pack('h', 83)) # MYSTERY NUMBER
            f.write(struct.pack('h', 0))
            for x in range(0, self.tiles_x*768, 3):
                for y in range(0, self.tiles_y*768, 3):
                    f.write(struct.pack('h', self.heights[x, y]))
 
        # for x in range(self.tiles_x):
            # for y in range(self.tiles_y):
                # data = self.heights[y*192:(y+1)*192, x*192:(x+1)*192]
                
        
    def imaging_wms(self, fraction=8, prog=None):#, url, layer, crs=None, fraction=8, prog=None):
        if prog is None:
            prog=lambda x: None
        url = self.wms['url']
        layer = self.wms['layer']
        crs = self.wms['crs']
        folder = self.output
        wms = WebMapService(url, version='1.1.1')
        wms_layers= list(wms.contents)
        if layer > len(wms_layers):
            raise ValueError(f"Layer {layer} not found out of {len(wms_layers)}.")
        layer = layer - 1 # Zero indexing
        crs_options = wms[wms_layers[layer]].crsOptions
        if crs is None:
            crs = crs_options[0]
        else:
            if len([x for x in crs_options if str(crs) in x]) == 0:
                raise ValueError("Unknown or unsupported CRS {crs} - service returned {crs_options}.")
        crs = int(str(crs).replace("EPSG:", ""))
        raw_tiff = os.path.join(folder,'_raw.tif')
        georeferenced_tiff = os.path.join(folder, '_georeferenced.tif')
    
        srs_string = 'EPSG:' + str(crs)
    
        tle, tln = self.corners['utm']['top_left']['easting'], self.corners['utm']['top_left']['northing']
   
        N = (self.tiles_x*fraction) * (self.tiles_y*fraction)
        width_m = 23040//fraction
        for i in range(self.tiles_x*fraction):
            for j in range(self.tiles_y*fraction):
                prog(100*1/N)
                tl = tle + i*width_m, tln - j*width_m
                br = tl[0] + width_m, tl[1] - width_m
                self.fetch_tile_wms(url, crs, layer, tl, br, 2*8192//fraction, f"{i:02d}{j:02d}", wms, wms_layers)

        # Stitch together for final texture tiles
        
        try:
            os.mkdir(f"{self.output}/textures")
        except:
            pass
        # Stitch
            
        for x in range(self.tiles_x):
            for y in range(self.tiles_y):
                tiles = []
                for i in range(fraction):
                    row = []
                    for j in range(fraction): 
                        row.append(f"{self.output}/{(x*fraction + i):02d}{(y*fraction + j):02d}.bmp")
                    tiles.append(row)
                stitched = stitch_tiles(tiles, 2*8192//fraction)
                stitched.save(f"{self.output}/textures/{x:02d}{y:02d}.bmp")
                stitched = None
        
    def fetch_tile_wms(self, url, crs, layer, tl, br, width, name, wms, wms_layers):   
        # tl = tl[0]+200, tl[1] - 200
        # br = br[0]+200, br[1] - 200
        west, north, east, south = tl[0], tl[1], br[0], br[1]
        
        # Get a bit extra for warping
        tl = tl[0] - 500, tl[1] + 500 
        br = br[0] + 500, br[1] - 500  
        
        # Corners in crs of image to fetch
        tl3 = coords_translate(tl[1], tl[0], self._out_proj(), crs)
        br3 = coords_translate(br[1], br[0], self._out_proj(), crs)
    
        lonmin, lonmax = tl3[0], br3[0]
        latmin, latmax = br3[1], tl3[1]
        
        srs_string = f"EPSG:{crs}"
    
        raw_tiff = os.path.join(self.output, '_raw.tif')
        georeferenced_tiff = os.path.join(self.output, '_georeferenced.tif')
    
        
        img = wms.getmap(layers=[wms_layers[layer]], styles=['default'], 
                         srs=srs_string, bbox=( lonmin, latmin , lonmax, latmax), 
                         size=(width, width), format='image/GeoTIFF' )
                         # size=(width, width), format="image/tiff") # format='image/GeoTIFF' )
        out = open(georeferenced_tiff, 'wb')#raw_tiff
        out.write(img.read())
        out.close()
        out = None
    
        input_raster = gdal.Open(georeferenced_tiff)
        epsg = f"EPSG:{self._out_proj()}"
        res = 2.8125
        warp = gdal.Warp(f"{self.output}/tileunclipped_{name}.tif", input_raster, srcSRS=f"EPSG:{crs}", 
                         dstSRS=epsg, xRes=res, yRes=res)
        warp = None
        
        ds = gdal.Open(f"{self.output}/tileunclipped_{name}.tif")
        output = f"{self.output}/{name}.bmp"
        # print("Saving " + self.reproj_dem)
        # ds = gdal.Translate(self.reproj_dem, ds, projWin = [west, north, east, south])     
        ds = gdal.Translate(output, ds, projWin = [west, north, east, south], width=width//2, height=width//2)   
        ds = None


    def save_features_bmp(self, prog=lambda x: None):
        queries = {
                "trees":   dict(landcover="trees", landuse="forest", natural="wood"),
                "airport": dict(aeroway="aerodrome"),
                "water":   dict(natural="water", waterway=True),
                "houses":  dict(landuse="residential"),
                "roads":   dict(highway=True)
                }
        tle, tln = self.corners['utm']['top_left']['easting'], self.corners['utm']['top_left']['northing']
        n = 1/(len(queries) * self.tiles_x * self.tiles_y)
        for x in range(self.tiles_x):
            for y in range(self.tiles_y):
                prog(n) 
                tl = tle + x*23040, tln - y*23040
                br = tl[0] + 23040, tl[1] - 23040
                west, north, east, south = tl[0], tl[1], br[0], br[1]
                bounds = [west, south, east, north]
                raster_width = 8192 
                transform = from_bounds(*bounds, raster_width, raster_width) # square
                for name, query in queries.items():
                    gdf = self.get_osm_data((tl[0], tl[1]), (br[0], br[1]), name, query)
                    if gdf is None:
                        print(f"No {name} for {x:02d} {y:02d}")
                        save_nothing("{self.output}/{x:02d}{y:02d}_{name}.bmp")
                    try:
                        raster = rasterize(
                            [(geom, 1) for geom in gdf.geometry],
                            out_shape=(raster_width, raster_width),
                            transform=transform,
                            fill=0,
                            all_touched=True,  # Consider all pixels touched by geometries
                            dtype=rasterio.uint8
                        )
                    except:
                        print(f"Failed {name} for {x:02d} {y:02d}")
                        save_nothing("{self.output}/{x:02d}{y:02d}_{name}.bmp")
                        continue
                    with rasterio.open(
                        f'{self.output}/{x:02d}{y:02d}_{name}.tif', 'w',
                        driver='GTiff',
                        height=raster_width,
                        width=raster_width,
                        count=1,
                        dtype=raster.dtype,
                        crs=gdf.crs,
                        transform=transform,
                    ) as dst:
                        dst.write(raster, 1)
                
                # Save as BMP
                    image = Image.fromarray(raster)
                    image.save(f"{self.output}/{x:02d}{y:02d}_{name}.bmp")

                

        
    # bbox = (south_lat, west_lon, north_lat, east_lon)
    def get_osm_data(self, tl, br, name, query):
        corners = tl, br
        west, north, east, south = tl[0], tl[1], br[0], br[1]
        # Get a bit extra for warping
        tl = tl[0] - 500, tl[1] + 500 
        br = br[0] + 500, br[1] - 500  
        crs =4326#EPSG:3857
        # Corners in crs of image to fetch
        tl3 = coords_translate(tl[1], tl[0], self._out_proj(), crs)
        br3 = coords_translate(br[1], br[0], self._out_proj(), crs)
        lonmin, lonmax = tl3[1], br3[1]
        latmin, latmax = br3[0], tl3[0]

        # Define the custom filter
        custom_filter = query 

        # Download the geometries from OSM
        # north, south, east, west
        try:
            gdf = ox.features_from_bbox(latmin, latmax, lonmax, lonmin, custom_filter)
        except Exception as e: #InsufficientResponseError:
            return None
        # del gdf['nodes']
        gdf = gdf[['geometry']]
        # Reproject if necessary
        gdf = gdf.to_crs(self._out_proj())
        gdf = gdf.clip([west, south, east, north])

        # Save the data for reference
        gdf.to_file(f"{self.output}/{name}.geojson", driver='GeoJSON')
        return gdf
    
    def __repr__(self):
        s = f"Scenery centered on {self.center['lat']}, {self.center['lon']} - {self.tiles_x}x{self.tiles_x} tiles\n"
        s += f"  UTM zone: {self.corners['utm']['top_left']['zone']}"
        s += f"""
    Top left: 
        Northing: {int(self.corners['utm']['top_left']['northing'])}
        Easting: {int(self.corners['utm']['top_left']['easting'])}
    Bottom right: 
        Northing: {int(self.corners['utm']['bottom_right']['northing'])}
        Easting: {int(self.corners['utm']['bottom_right']['easting'])}
Output in EPSG:{self._out_proj()}"""
        return s

    def to_csv_utm(self):
        pass

    def to_shapefile(self):
        corners = self.corners
        bounding = make_bbox(corners['utm']['top_left']['easting'],
                            corners['utm']['top_left']['northing'],
                            corners['utm']['bottom_right']['northing'],
                            corners['utm']['bottom_right']['easting'])
        gpd.GeoDataFrame(pd.DataFrame(['p1'], columns = ['geom']),
        crs = f'EPSG:{self._out_proj()}',
        geometry = [bounding]).to_file(f'{self.output}/scenery.shp')

        

