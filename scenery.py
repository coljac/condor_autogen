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
from wand import image
import sys
import struct
import math
from dotmap import DotMap
import requests
from io import BytesIO
from xplane_airports.gateway import scenery_pack
from xplane_apt_convert import ParsedAirport
# some_key = 91f1293404efb2f855dec49d820e35d3

# TODO
# Proper progbars
# Fake textures
# Google static
# Bing wmts
# Google static
# Azure maps
# Web integration
# Multiprocess
# Check google and bing/tiles works well
# Color matching - use photopea.
# Runways - X-plane data?
# dxt3 compression; higher res tiles
# Flatten airports

def parse_steps(s):
    steps = []
    bits = s.split(",")
    for bit in bits:
        try:
            steps.append(int(bit))
        except:
            try:
                steps += list(range(int(bit.split("-")[0]), 1 + int(bit.split("-")[1])))
            except:
                raise ValueError(f"Can't parse steps {bit}.")
    return steps


def google_static_tile(lat, lon, zoom, token):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=290x290&maptype=satellite&style=feature:all|element:labels|visibility:off&key={token}"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    left, top, right, bottom = 17, 17, 256-17, 256-17
    img = img.crop((left, top, right, bottom))
    return img

def get_airport_elevation(airport_id, feet=False):
    recommended_pack = scenery_pack(airport_id)
    apt = recommended_pack.apt
    elevation_ft = int(ParsedAirport(apt)._airport.raw_lines[0].split()[1])
    if feet:
        return int(elevation_ft)
    elevation_m = elevation_ft * 0.3048
    return int(elevation_m)

def get_airport_footprint(airport_id):
    recommended_pack = scenery_pack(airport_id)
    apt = recommended_pack.apt
    pa = ParsedAirport(apt)

def ii(x):
    return np.array(Image.open(x))


def make_bbox(long0, lat0, lat1, long1):
    return Polygon([[long0, lat0],
                    [long1, lat0],
                    [long1, lat1],
                    [long0, lat1]])


def stopPrint(func, *args, **kwargs):
    with open(os.devnull, "w") as devNull:
        original = sys.stdout
        originalerr = sys.stderr
        sys.stdout = devNull
        sys.stderr = devNull
        func(*args, **kwargs)
        sys.stdout = original
        sys.stderr = originalerr


def calculate_utm_and_latlon_corners(lat, lon, x_tiles, y_tiles, tile_size):
    """
    This function takes a latitude/longitude pair and calculates the UTM coordinates
    and the corresponding latitude/longitude for the top left and bottom right corners
    of a box of x_tiles by y_tiles, with each tile being tile_size square meters.
    """
    # Convert the center point latitude and longitude to UTM coordinates
    center_easting, center_northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    center_easting = int(center_easting)
    center_northing = int(center_northing)

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
    bottom_right_lat, bottom_right_lon = utm.to_latlon(bottom_right_easting, bottom_right_northing, zone_number,
                                                       zone_letter)

    return {
        'utm': {
            'top_left': {'northing': top_left_northing, 'easting': top_left_easting,
                         'zone': f"{zone_number}{zone_letter}"},
            'bottom_right': {'northing': bottom_right_northing, 'easting': bottom_right_easting,
                             'zone': f"{zone_number}{zone_letter}"}
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
    coords = transform.TransformPoint(y, x)  # CHECK
    x, y = coords[0:2]
    return x, y


def stitch_tiles(tiles, tile_width, output_size=8192):
    rows = len(tiles)
    cols = len(tiles[0])

    # stitched_image = Image.new('RGB', (cols*tile_width, rows*tile_width))
    stitched_image = Image.new('RGB', (output_size, output_size))

    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            data = Image.open(tile)
            data.resize((tile_width // 2, tile_width // 2))
            stitched_image.paste(data, (i * tile_width // 2, j * tile_width // 2))
            data = None

    stitched_image.resize((output_size, output_size))
    return stitched_image


def save_nothing(name, size=8192):
    nothing = Image.new("RGB", (size, size))
    nothing.save(name)


def create_rgba_texture(rgb_texture_path, water_mask_path, output_path):
    # Load the RGB texture image
    rgb_texture = np.array(Image.open(rgb_texture_path).convert("RGB"))
    # Load the water mask image
    water_mask = np.array(Image.open(water_mask_path).convert("L"))  # Convert to grayscale
    alpha = 255 * (1 - np.array(water_mask))
    rgb_pixel = np.array(rgb_texture)
    # Create a new image with RGBA mode

    texture = np.concatenate([rgb_texture, alpha[:, :, np.newaxis]], axis=2)
    rgba_texture = Image.fromarray(texture, "RGBA")

    # Iterate over pixels

    # for x in range(rgb_texture.width):
    # for y in range(rgb_texture.height):
    # rgb_pixel = rgb_texture.getpixel((x, y))
    # mask_pixel = water_mask.getpixel((x, y))
    # alpha = 0 if mask_pixel == 1 else 255
    # rgba_texture.putpixel((x, y), rgb_pixel + (alpha,))

    # Save the result
    rgba_texture.save(output_path)


class MapIter(object):
    def __init__(self, scenery, bits=4, res=8192, tiles_only=False):
        # Reversed means bottom to top
        self.scenery = scenery
        self.res = res
        self.bits = bits
        self.x = 0
        self.y = 0
        self.y_max = scenery.tiles_y
        self.x_max = scenery.tiles_x
        self.i = -1
        self.j = 0
        self.tile = {}
        self.tiles_only = tiles_only

    def __iter__(self):
        return self

    def __next__(self):
        new = self.i < 0
        self.i += 1
        if self.i == self.bits:
            self.j += 1
            self.i = 0
        if self.j == self.bits:
            self.x += 1
            new = True
            self.i = 0
            self.j = 0
        if self.x >= self.x_max:
            self.y += 1
            new = True
            self.x = 0
        if self.y >= self.y_max:
            raise StopIteration
        i, j = self.i, self.j
        x, y = self.x, self.y
        b = self.bits
        res = self.res
        w = res // b

        if new:
            bottom_right = (self.scenery.corners['utm']['bottom_right']['easting'] - (23040 * x),
                            self.scenery.corners['utm']['bottom_right']['northing'] + (23040 * y))
            top_left = (bottom_right[0] - 23040, bottom_right[1] + 23040)
            self.tile = dict(
                bottom_right=bottom_right,
                top_left=top_left,
                corners=[dict(lat=x[0], lon=x[1]) for x in
                         [utm.to_latlon(*y, self.scenery.zone, self.scenery.zone_letter) for y in
                          [bottom_right, top_left]
                          ]]
            )
        # Coordinates in the mini-tile space
        xx = self.x * self.bits + self.i
        yy = self.y * self.bits + self.j

        # coordinates in the big pixel space
        xp = self.x_max * res - (i + 1) * w - x * res
        xp = slice(xp, xp + w)
        yp = self.y_max * res - (j + 1) * w - y * res
        yp = slice(yp, yp + w)

        # Coordinates in the current big tile
        xt = res - (i + 1) * w
        xt = slice(xt, xt + w)
        yt = res - (j + 1) * w
        yt = slice(yt, yt + w)
        name = f"{xx:02d}{yy:02d}"
        return DotMap(i=self.i, j=self.j, x=self.x, y=self.y, xp=xp, yp=yp, xt=xt, yt=yt, xx=xx, yy=yy, name=name,
                      tile=f"{self.x:02d}{self.y:02d}", new=new, tile_coords=self.tile)


class Scenery(object):
    def from_yaml(yaml_file):  # , output=os.getcwd()):
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)['scenery']
            name = config['name']
            output = os.getcwd()
            if 'output' in config and config['output'] is not None and len(config['output']) > 0:
                output = config['output']
            scenery = Scenery(name, config['center'], config['tiles']['x'], config['tiles']['y'],
                              config=config, output=output)
            scenery.output = output
            try:
                scenery.wms = config['WMTS']
            except:
                pass
            return scenery

    def __init__(self, name, center, tiles_x=4, tiles_y=4, config=None, output="."):
        self.progress = None
        self.progbars = {}
        self.show_progress = True
        self.name = name
        self.output = output
        self.w = f"{self.output}/{self.name}/Working"
        self.f = f"{self.output}/{self.name}"
        self.center = center
        self.tiles_x = tiles_x
        self.tiles_y = tiles_x
        self.tile_size = 23040
        self.corners = calculate_utm_and_latlon_corners(center['lat'],
                                                        center['lon'],
                                                        self.tiles_x,
                                                        self.tiles_x,
                                                        self.tile_size)
        self.raw_dem = f"{self.w}/dem.tif"
        self.reproj_dem = f"{self.w}/dem_reproj.tif"
        self.zone = int(self.corners['utm']['top_left']['zone'][:-1])
        self.zone_ns = "S" if self.center["lat"] < 0 else "N"
        self.zone_letter = self.corners['utm']['top_left']['zone'][-1]
        self.heights = None
        self._dirs()
        self.config = config
        self.N = self.tiles_x * self.tiles_y * 16

        self.imaging = self.imaging_wmts

        # url_here = 'https://maps.hereapi.com/v3/base/mc/{z}/{x}/{y}/png8?style=satellite.day&apiKey=' + HERE_API_KEY
        if config['textures']['source'].lower() == "google":
            # https://khms2.google.com/kh/v=966?x=7391&y=5023&z=13
            self.config['WMTS']['url'] = "https://khms2.google.com/kh/v=966?x={x}&y={y}&z={z}"#http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"
        if config['textures']['source'].lower() == "bing":
            self.config['WMTS']['url'] = ""
        if config['textures']['source'].lower() == "usgs":
            self.config['WMTS']['url']  = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        if config['textures']['source'] == "WMS":
            self.imaging = self.imaging_wms
        elif config['textures']['source'] == "WMTS":
            if self.config['WMTS']['url'] is None or len(self.config['WMTS']['url']) == 0:
                raise ValueError("WMTS URL not specified.")
        elif config['textures']['source'] == "generate":
            self.imaging = self.generate_images
        self.bits = 4

    def mk(self, x):
        try:
            os.mkdir(f"{self.output}/{x}")
        except FileExistsError:
            pass

    def _dirs(self):
        name = self.name
        self.mk("")
        self.mk(name)
        self.mk(f"{name}/Working")
        self.mk(f"{name}/Forestmaps")
        self.mk(f"{name}/HeightMaps")
        self.mk(f"{name}/Textures")
        self.mk(f"{name}/Airports")
        self.mk(f"{name}/Images")
        self.mk(f"{name}/Working/textures")
        self.mk(f"{name}/Working/features")

    def save_ini(self):
        with open(f"{self.output}/{self.name}/{self.name}.ini", "w") as f:
            f.write(f"[General]\nVersion={self.config['version']}\nRealtimeShading=1")

    def save_image(self):
        output = Image.new(mode="RGB", size=(256 * self.tiles_y, 256 * self.tiles_x))
        for t in self.iterate(res=256, bits=1):
            if t.new:
                i = Image.open(f"{self.w}/textures/{t.tile}.png")
                i = i.resize((256, 256))
                output.paste(i, (t.xp.start, t.yp.start))
        output.save(f"{self.f}/{self.name}.bmp")

    def update_progress(self, key, prog):
        if key not in self.progress:
            progress = None
            self.progress[key] = progress
        progress = self.progress[key]
        progress.update(prog)

    def _get_prog(self, name):
        if not self.show_progress:
            return lambda x: None

        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),  # Spinner column for the main task
                TextColumn("[bold blue]{Generating landscape}"),  # Task description
                BarColumn(bar_width=50),  # Progress bar for the sub-tasks
                "{task.percentage:>3.0f}%",  # Percentage completed for the sub-tasks
                expand=False)
       
        task = self.progbars.get(name, self.progress.add_task(name))
        # task = self.progress.add_task(name)
        self.progbars[name] = task
        return task

    def _finish(self):
        self.progress.finish()

    def __del__(self):
        try:
            if self.show_progress:
                self._finish()
        except:
            pass

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

            updater = lambda x: progress.update(height_data_task, advance=x)
            stopPrint(self._fetch, prog=updater)
            # self._fetch(prog=updater)
            updater = lambda x: progress.update(height_data_task, advance=x)
            self.reproj_dems(prog=updater)
            updater = lambda x: progress.update(scenery_images_task, advance=x)
            self.imaging_wms(prog=updater)
            progress.update(scenery_task, completed=100)

    def fetch_heights(self):
        prog = self._get_prog("[green] Fetching DEM (heights)")
        self.raw_dem = f"{self.w}/raw_dem.tif"
        corners = self.corners
        bounds = (corners['latlon']['bottom_right'][0],
                  corners['latlon']['top_left'][1],
                  corners['latlon']['top_left'][0],
                  corners['latlon']['bottom_right'][1]
                  )
        prog(20)
        # Add margin around boundary for reprojection issues`
        bounds = (bounds[1] - 0.5, bounds[0] - 0.5, bounds[3] + 0.5, bounds[2] + 0.5)
        stopPrint(elevation.clip, bounds=bounds, output=self.raw_dem)
        # elevation.clip(bounds=bounds, output=self.raw_dem)
        prog(80)

    # Reproject and clip
    def reproj_dems(self):
        prog = self._get_prog("[blue]Reprojecting DEM to UTS")
        self.reproj_dem = f"{self.w}/reproj_dem.tif"
        input_raster = gdal.Open(self.raw_dem)
        epsg = f"EPSG:{self._out_proj()}"
        gdal.Warp(f"{self.w}/dem_unclipped.tif", input_raster, dstSRS=epsg, xRes=30, yRes=30)

        west, south, east, north = [self.corners['utm']['top_left']['easting'],
                                    self.corners['utm']['bottom_right']['northing'],
                                    self.corners['utm']['bottom_right']['easting'],
                                    self.corners['utm']['top_left']['northing']]
        ds = gdal.Open(f"{self.w}/dem_unclipped.tif")
        prog(33)
        ds = gdal.Translate(self.reproj_dem, ds, projWin=[west, north, east, south])
        prog(33)
        ds = gdal.Translate(self.reproj_dem.replace("tif", "bil"), ds, projWin=[west, north, east, south])
        self.heights = ii(self.reproj_dem)
        prog(67)

    def _out_proj(self):
        epsg = CRS.from_dict({'proj': 'utm', 'zone': self.corners['utm']['top_left']['zone'][:-1],
                              "south": self.corners['latlon']['top_left'][0] < 0})
        return int(epsg.to_authority()[1])

    def save_trn(self):
        # heightmaps = os.path.dirname(dest_file) + "/HeightMaps"
        heightmaps = f"{self.f}/HeightMaps"
        dest_file = f"{self.f}/{self.name}.trn"
        with open(dest_file, "wb") as f:
            f.write(struct.pack('h', (self.tiles_x * 768) // 3))
            f.write(struct.pack('h', 0))
            f.write(struct.pack('h', (self.tiles_y * 768) // 3))
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
            f.write(struct.pack('h', 78 if self.center['lat'] > 0 else 83))  # N or S
            f.write(struct.pack('h', 0))
            if self.heights is None:
                self.heights = ii(self.reproj_dem)
            NX, NY = self.tiles_x * 768, self.tiles_y * 768
            a = self.heights
            a = np.fliplr(np.flipud(a)).astype(np.int16)
            a = np.clip(a, 0, 1e6).astype(np.uint16)
            for x in range(0, NX, 3):
                for y in range(0, NY, 3):
                    f.write(struct.pack('h', a[y, x]))
                    # f.write(struct.pack('h', self.heights[NY-y-1, NX-x-1]))

    def save_tr3s(self):
        # a = self.heights
        a = ii(self.reproj_dem).astype(np.int16)
        a = np.pad(a, [(1, 0), (1, 0)], mode="edge")
        # a = np.fliplr(np.flipud(a)).astype(np.int16)
        # a = np.flipud(a).astype(np.uint16)
        a = np.clip(a, 0, 1e6).astype(np.uint16)
        for t in self.iterate(res=768):
            arr = np.zeros((193, 193), dtype=np.int16)

            data = a[t.yp.start:t.yp.stop + 1, t.xp.start:t.xp.stop + 1]

            arr[:data.shape[0], :data.shape[1]] = data

            arr = np.rot90(arr, 3)
            arr = np.flipud(arr)

            arr.tofile(f"{self.f}/HeightMaps/h{t.name}.tr3")

    def figure_out_wmts_zoom_level(self, pixels=8192):
        zoom = 15
        width, height = 0, 0
        for t in self.iterate():
            top_left = t.tile_coords['corners'][1]
            bottom_right = t.tile_coords['corners'][0]
            while zoom < 18:
                x_max, y_max = lat_lon_to_tile_coords(bottom_right['lat'], bottom_right['lon'], zoom)
                x_min, y_min = lat_lon_to_tile_coords(top_left['lat'], top_left['lon'], zoom)
                width = (x_max - x_min)*256
                height = (y_max - y_min)*256
                if width > pixels*1.1 and height > pixels*1.1:
                    return zoom
                zoom += 1
        return zoom

    
    def imaging_wmts(self):
        tile_server_url = self.config['WMTS']['url']
        zoom = self.figure_out_wmts_zoom_level()
        # def download_and_stitch_tiles(zoom, x_min, x_max, y_min, y_max, tile_server_url):
        # Initialize an empty image of the right size
        for t in self.iterate():
            if not t.new:
                continue
            top_left = t.tile_coords['corners'][1]
            bottom_right = t.tile_coords['corners'][0]
            x_max, y_max = lat_lon_to_tile_coords(bottom_right['lat'], bottom_right['lon'], zoom)
            x_min, y_min = lat_lon_to_tile_coords(top_left['lat'], top_left['lon'], zoom)
            # add a bit extra to account for the tile coordinate transformation and crop later
            x_min -= 1
            x_max += 1
            y_min -= 1
            y_max += 1
            total_width = (x_max - x_min + 1) * 256
            total_height = (y_max - y_min + 1) * 256
            merged_image = Image.new('RGB', (total_width, total_height))
            # Download the tiles and paste them into the merged_image
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    url = tile_server_url.format(z=zoom, x=x, y=y)
                    response = requests.get(url)
                    # return
                    if response.status_code == 200:
                        tile = Image.open(BytesIO(response.content))
                        merged_image.paste(tile, box=((x - x_min) * 256, (y - y_min) * 256))
                        # tile.save(f"/tmp/{x:02d}{y:02d}.png")
                    else:
                        print(f"Tile server error {response.status_code}: Check URL ({url})")
                        return
            # save merged_image as a geotiff
            self._generate_geotiff(merged_image, zoom, x_min, x_max, y_min, y_max, f"{self.w}/textures/{t.tile}.tif")
            # Transform
            # Save as bmp
            tle, tln = self.corners['utm']['top_left']['easting'], self.corners['utm']['top_left']['northing']
            width_m = 23040
            tl = tle + t.x * width_m, tln - t.y * width_m
            br = tl[0] + width_m, tl[1] - width_m
            west, north, east, south = tl[0], tl[1], br[0], br[1]
            ds = gdal.Open(f"{self.w}/textures/{t.tile}.tif")
            output_file = f"{self.w}/textures/{t.tile}.bmp"
            gdal.Translate(output_file, ds, projWin=[west, north, east, south])
            break

    def _generate_geotiff(self, image, zoom, x_min, x_max, y_min, y_max, output_path):
        num_tiles_x = x_max - x_min + 1
        num_tiles_y = y_max - y_min + 1
        tile_size = 256
        # latitude = tile_coords_to_lat_lon((x_max+x_min)//2, y_min, zoom)[0]
        latitude = 0
        resolution = 156543.03 * math.cos(math.radians(latitude)) / (2 ** zoom)

        top_left_x = x_min * tile_size * resolution - 20037508.34
        top_left_y = 20037508.34 - y_min * tile_size * resolution

        geotransform = (top_left_x, resolution, 0, top_left_y, 0, -resolution)

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_path, image.width, image.height, 3, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(geotransform)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        dst_ds.SetProjection(srs.ExportToWkt())

        # Split the image into individual bands and write each one
        for i in range(3):
            band = dst_ds.GetRasterBand(i + 1)
            band_array = np.array(image.getchannel(i))
            band.WriteArray(band_array)

        dst_ds = None

    def imaging_wms(self):
        prog = None
        fraction = 8
        if prog is None:
            prog = lambda x: None
        url = self.config['WMTS']['url']
        layer = self.config['WMTS']['layer']
        crs = self.config['WMTS']['crs']
        folder = self.output
        wms = WebMapService(url, version='1.1.1')
        wms_layers = list(wms.contents)
        if layer > len(wms_layers):
            raise ValueError(f"Layer {layer} not found out of {len(wms_layers)}.")
        layer = layer - 1  # Zero indexing
        crs_options = wms[wms_layers[layer]].crsOptions
        if crs is None:
            crs = crs_options[0]
        else:
            if len([x for x in crs_options if str(crs) in x]) == 0:
                raise ValueError("Unknown or unsupported CRS {crs} - service returned {crs_options}.")
        crs = int(str(crs).replace("EPSG:", ""))
        # raw_tiff = os.path.join(folder, '_raw.tif')
        # georeferenced_tiff = os.path.join(folder, 'Working/georeferenced.tif')
        #
        # srs_string = 'EPSG:' + str(crs)

        tle, tln = self.corners['utm']['top_left']['easting'], self.corners['utm']['top_left']['northing']
        N = (self.tiles_x * fraction) * (self.tiles_y * fraction)
        width_m = 23040 // fraction

        for i in range(self.tiles_x * fraction):
            for j in range(self.tiles_y * fraction):
                prog(100 * 1 / N)
                tl = tle + i * width_m, tln - j * width_m
                br = tl[0] + width_m, tl[1] - width_m
                self.fetch_tile_wms(url, crs, layer, tl, br, 2 * 8192 // fraction, f"{i:02d}{j:02d}", wms, wms_layers)

        # Stitch together for final texture tiles
        for x in range(self.tiles_x):
            for y in range(self.tiles_y):
                tiles = []
                for i in range(fraction):
                    row = []
                    for j in range(fraction):
                        row.append(f"{self.w}/textures/f{(x * fraction + i):02d}{(y * fraction + j):02d}.bmp")
                    tiles.append(row)
                stitched = stitch_tiles(tiles, 2 * 8192 // fraction)
                xx = self.tiles_x - x - 1
                yy = self.tiles_y - y - 1
                stitched.save(f"{self.w}/textures/{xx:02d}{yy:02d}.bmp")
                stitched = None

    # def t(self, i, j, bits=4):
    #     return f"{i:02d}{j:02d}"

    def iterate(self, bits=4, res=8192):
        return MapIter(self, bits, res=res)

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

        raw_tiff = os.path.join(self.w, '_raw.tif')
        georeferenced_tiff = os.path.join(self.w, 'georeferenced.tif')

        img = wms.getmap(layers=[wms_layers[layer]], styles=['default'],
                         srs=srs_string, bbox=(lonmin, latmin, lonmax, latmax),
                         size=(width, width), format='image/GeoTIFF')
        # size=(width, width), format="image/tiff") # format='image/GeoTIFF' )
        out = open(georeferenced_tiff, 'wb')  # raw_tiff
        out.write(img.read())
        out.close()
        out = None

        input_raster = gdal.Open(georeferenced_tiff)
        epsg = f"EPSG:{self._out_proj()}"
        res = 2.8125
        warp = gdal.Warp(f"{self.w}/textures/tileunclipped_{name}.tif", input_raster, srcSRS=f"EPSG:{crs}",
                         dstSRS=epsg, xRes=res, yRes=res)
        warp = None

        ds = gdal.Open(f"{self.w}/textures/tileunclipped_{name}.tif")
        output = f"{self.w}/textures/f{name}.bmp"
        # print("Saving " + self.reproj_dem)
        # ds = gdal.Translate(self.reproj_dem, ds, projWin = [west, north, east, south])     
        ds = gdal.Translate(output, ds, projWin=[west, north, east, south], width=width // 2, height=width // 2)
        ds = None

    # trees 2048x2048
    def save_features_bmp(self, prog=lambda x: None):
        queries = {
            "water": dict(natural="water", waterway=True),
            "trees_b": dict(landcover="trees", natural="wood"),
            "trees_s": dict(landuse="forest"),
            "airport": dict(aeroway="aerodrome"),
            "houses": dict(landuse="residential"),
            "roads": dict(highway=True)
        }
        # tle, tln = self.corners['utm']['top_left']['easting'], self.corners['utm']['top_left']['northing']
        tle, tln = self.corners['utm']['bottom_right']['easting'], self.corners['utm']['bottom_right']['northing']
        n = 1 / (len(queries) * self.tiles_x * self.tiles_y)
        # for t in self.iterate():
        #     if True:
        #         x, y = t.x, t.y
        for x in range(self.tiles_x):
            for y in range(self.tiles_y):
                prog(n)
                tl = tle - (x + 1) * 23040, tln + (y + 1) * 23040
                br = tl[0] + 23040, tl[1] - 23040
                west, north, east, south = tl[0], tl[1], br[0], br[1]
                bounds = [west, south, east, north]
                raster_width = 8192
                transform = from_bounds(*bounds, raster_width, raster_width)  # square
                for name, query in queries.items():
                    gdf = self.get_osm_data((tl[0], tl[1]), (br[0], br[1]), name, query)
                    if gdf is None:
                        print(f"No {name} for {x:02d} {y:02d}")
                        save_nothing(f"{self.w}/features/{x:02d}{y:02d}_{name}.bmp")
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
                        save_nothing(f"{self.w}/features/{x:02d}{y:02d}_{name}.bmp")
                        continue
                    with rasterio.open(
                            f'{self.w}/features/{x:02d}{y:02d}_{name}.tif', 'w',
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
                    image.save(f"{self.w}/features/{x:02d}{y:02d}_{name}.bmp")

    def make_water(self):
        for x in range(self.tiles_x):
            for y in range(self.tiles_y):
                tex = f"{self.w}/textures/{x:02d}{y:02d}.bmp"
                water = f"{self.w}/features/{x:02d}{y:02d}_water.bmp"
                create_rgba_texture(tex, water, tex.replace("bmp", "png"))

    def make_forestmaps(self):
        for t in self.iterate(res=2048):
            x, y = t.x, t.y
            if t.new:
                img_s = Image.open(f"{self.w}/features/{x:02d}{y:02d}_trees_s.bmp").resize((2048, 2048))
                img_b = Image.open(f"{self.w}/features/{x:02d}{y:02d}_trees_b.bmp").resize((2048, 2048))
                forest_cover = np.array(img_s).astype(np.uint8) + \
                               (2 * np.array(img_b)).astype(np.uint8)
            output_data = forest_cover[t.xt, t.yt]
            with open(f"{self.f}/Forestmaps/{t.name}.for", "wb") as f:
                output_data.tofile(f)

    def make_thermal_map(self):
        output_data = np.zeros((self.tiles_y * 256, self.tiles_x * 256))
        for t in self.iterate(res=256):
            if not t.new:
                continue
        # for x in range(self.tiles_x):
        #     for y in range(self.tiles_y):
            img = Image.open(f"{self.w}/features/{t.x:02d}{t.y:02d}_thermal.bmp").resize((256, 256))
            thermal_strength = np.array(img)
            output_data[t.y * 256:(t.y + 1) * 256, t.x * 256:(t.x + 1) * 256] = thermal_strength.astype(np.uint8)

        with open(f"{self.f}/{self.name}.tha", "wb") as f:
            output_data.tofile(f)

    # def export_textures_bmp(self):
    #     W = 2048
    #     print(f"Width {W}, from bitmap, dxt3, mipmaps 12")
    #     for t in self.iterate():
    #         if t.new:
    #             img = None
    #         xx = t.x  # self.tiles_x - t.x - 1
    #         yy = t.y  # self.tiles_y - t.y - 1
    #         if img is None:
    #             bmp_img = image.Image(filename=f"{self.w}/textures/{xx:02d}{yy:02d}.bmp")
    #             # tex = f"{self.w}/textures/{x:02d}{y:02d}.bmp"
    #             water = f"{self.w}/features/{xx:02d}{yy:02d}_water.bmp"
    #             # create_rgba_texture(tex, water, tex.replace("bmp", "png"))
    #             alpha_image = image.Image(filename=water)
    #             alpha_image.type = 'grayscale'
    #             bmp_img.composite_channel('alpha', alpha_image, 'copy')
    #         output_data = bmp_img.clone()
    #         try:
    #             output_data.crop(t.xt.start, t.yt.start, width=W, height=W)
    #         except:
    #             print(t)
    #             return
    #         # output_file = f"{self.output}/{self.name}/Textures/t{(4*x)+i:02d}{(4*y)+j:02d}.dds"
    #         output_file = f"{self.f}/Textures/t{t.name}.dds"
    #         output_data.options['dds:mipmaps'] = '12'
    #         output_data.compression = 'dxt3'
    #         # output_data.compression = 'dxt3'
    #         output_data.save(filename=output_file)

    # nvDXT.exe -quick -nmips 12 -all -outdir "C:\Program Files\Condor\Landscapes\TestScenery\Working\Terragen\DDS" -dxt1c -Triangle
    def export_textures_the_hard_way(self):
        W = 2048
        print(f"Width {W}, dxt3, mipmaps 12")
        for t in self.iterate():
            if t.new:
                img = None
            xx = t.x  # self.tiles_x - t.x - 1
            yy = t.y  # self.tiles_y - t.y - 1
            if img is None:
                img = image.Image(filename=f"{self.w}/textures/{xx:02d}{yy:02d}.png")
            output_data = img.clone()
            try:
                output_data.crop(t.xt.start, t.yt.start, width=W, height=W)
            except:
                print(t)
                return
            # output_file = f"{self.output}/{self.name}/Textures/t{(4*x)+i:02d}{(4*y)+j:02d}.dds"
            output_file = f"{self.f}/Textures/t{t.name}.png"
            # output_data.options['dds:mipmaps'] = '12'
            # output_data.compression = 'dxt3'
            # output_data.compression = 'dxt3'
            output_data.save(filename=output_file)
            command = ["/usr/bin/wine", "/ssd/condor/Condor Landscape Toolkit_v2/nvdxt.exe",
                       "-quick", "-nmips", "12", "-all", "-outdir", f"{self.w}/textures/png", "-dxt1c", "-Triangle",
                       "-file", output_file]
            # print(f"/ssd/condor/Landscapes/Bacchus/convert.sh {self.w}/textures/png {output_file}")
            os.system(f"/ssd/condor/Landscapes/Bacchus/convert.sh {self.w}/textures/png {output_file}")
            # return

    def export_textures(self):
        W = 2048
        print(f"Width {W}, dxt3, mipmaps 12")
        for t in self.iterate():
            if t.new:
                img = None
            xx = t.x  # self.tiles_x - t.x - 1
            yy = t.y  # self.tiles_y - t.y - 1
            if img is None:
                img = image.Image(filename=f"{self.w}/textures/{xx:02d}{yy:02d}.png")
            output_data = img.clone()
            try:
                output_data.crop(t.xt.start, t.yt.start, width=W, height=W)
            except:
                print(t)
                return
            output_file = f"{self.f}/Textures/t{t.name}.dds"
            output_data.options['dds:mipmaps'] = '12'
            output_data.compression = 'dxt1'
            # output_data.compression = 'dxt3'
            output_data.save(filename=output_file)

    # bbox = (south_lat, west_lon, north_lat, east_lon)
    def get_osm_data(self, tl, br, name, query):
        corners = tl, br
        west, north, east, south = tl[0], tl[1], br[0], br[1]
        # Get a bit extra for warping
        tl = tl[0] - 500, tl[1] + 500
        br = br[0] + 500, br[1] - 500
        crs = 4326  # EPSG:3857
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
        except Exception as e:  # InsufficientResponseError:
            return None
        # del gdf['nodes']
        gdf = gdf[['geometry']]
        # Reproject if necessary
        gdf = gdf.to_crs(self._out_proj())
        gdf = gdf.clip([west, south, east, north])

        # Save the data for reference
        gdf.to_file(f"{self.w}/{name}.geojson", driver='GeoJSON')
        return gdf

    def __repr__(self):
        s = f"Landscape {self.name}, centered on {self.center['lat']}, {self.center['lon']} - {self.tiles_x}x{self.tiles_x} tiles\n"
        s += f"  UTM zone: {self.corners['utm']['top_left']['zone']}"
        s += f"""
    Top left: 
        Easting: {int(self.corners['utm']['top_left']['easting'])}
        Northing: {int(self.corners['utm']['top_left']['northing'])}
    Bottom right: 
        Easting: {int(self.corners['utm']['bottom_right']['easting'])}
        Northing: {int(self.corners['utm']['bottom_right']['northing'])}
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
        gpd.GeoDataFrame(pd.DataFrame(['p1'], columns=['geom']),
                         crs=f'EPSG:{self._out_proj()}',
                         geometry=[bounding]).to_file(f'{self.w}/scenery.shp')

    def cleanup(self):
        pass


    def flatten_airports(self):
        airports = self.config['airports']
        for ap in airports:
            # Look for elevation in the following sources
            # 1) The config file
            # 2) The airport database
            # 3) X-Plane
            pass
