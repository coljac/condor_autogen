from scenery import *
import yaml
import progress
import time
from time import sleep
import sys
import tqdm
import rich
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, track



# Bungal
latitude = -37.695021
longitude = 144.100390
x_tiles = 4
y_tiles = 4
tile_size = 23040  # Size of each tile in meters

# Calculate the corners
# corners = calculate_utm_and_latlon_corners(latitude, longitude, x_tiles, y_tiles, tile_size)
# print(corners)

# scenery = Scenery({"lat": latitude, "lon": longitude}, tiles_x=4, output="/ssd/condor/test01")
# scenery = Scenery.from_yaml("scenery.yaml")#, output="/ssd/condor/test01")
scenery = Scenery.from_yaml("segel.yaml")#, output="/ssd/condor/test01")
# scenery = Scenery.from_yaml("wmts.yaml")#, output="/ssd/condor/test01")
# scenery.to_shapefile()


# scenery.export_textures()
# scenery.make_thermal_map()
# scenery.save_ini()
# scenery.save_image()
print(scenery)
scenery.imaging()
# scenery.imaging()
# for x in scenery.iterate():
#     print(x)
#
# console = Console()
#
# scenery = Scenery.from_yaml("scenery.yaml")
# console.print("[bold cyan]Creating scenery with the following parameters:")
# console.print(scenery)
# tasks = ["[bold red]Fetching height data", "[bold yellow]Saving height maps", "Fetching scenery images", "Creating textures", "Fetching runway data",
#          "Making runway textures"]
#
#
# # scenery.generate()
# scenery.imaging_wms(scenery.wms['url'], scenery.wms['layer'], scenery.wms['crs'], prog=updater)
# sys.exit(0)
#
# # Create a Progress context manager
# with Progress(
#     SpinnerColumn(),  # Spinner column for the main task
#     TextColumn("[bold blue]{task.description}"),  # Task description
#     BarColumn(bar_width=50),  # Progress bar for the sub-tasks
#     "{task.percentage:>3.0f}%",  # Percentage completed for the sub-tasks
#     expand=False
# ) as progress:
#
#     # Main task spinner
#     scenery_task = progress.add_task("[green]Generating scenery...", total=None)
#
#     # Sub-task progress bars
#     height_data_task = progress.add_task("Fetching height data", total=100)
#     height_maps_task = progress.add_task("Making height maps", total=100)
#     scenery_images_task = progress.add_task("Fetching scenery images", total=100)
#     textures_task = progress.add_task("Making textures", total=100)
#
#     # Simulate some work with the progress bars
#     for i in range(200):
#     # while not progress.finished:
#         progress.update(height_data_task, advance=0.3)  # Simulate height data fetching
#         sleep(0.01)  # Pause to simulate work being done
#     for i in range(200):
#         progress.update(height_maps_task, advance=0.2)  # Simulate height map creation
#         # progress.update(scenery_images_task, advance=0.5)  # Simulate scenery image fetching
#         # progress.update(textures_task, advance=0.4)  # Simulate texture creation
#         sleep(0.01)  # Pause to simulate work being done
#
#
#     # Complete the main spinner task
#     progress.update(scenery_task, completed=100)
#
# # print()
# # for task in track(tasks):
# #     rich.print(f"{task}...")
# #     sleep(2)
#
# # with console.status("[bold green]Generating scenery...", spinner="moon") as status:
# #     while tasks:
# #         task = tasks.pop(0)
# #         for step in track(range(100)):
# #             sleep(0.05)
# #         # console.log(f"[bright red]{task}...")
# #         sleep(2)
# #         # console.log(f"{task} complete")
#
# # # YBSS
# # # latitude = -37.730956
# # # longitude = 144.422714
# # print(scenery)
# # from progress.spinner import MoonSpinner
#
# # with MoonSpinner('Processing: ') as bar:
# #     for i in range(100):
# #         time.sleep(0.2)
# #         bar.next()
#
#
#
# # # Bungal
# # latitude = -37.695021
# # longitude = 144.100390
# # x_tiles = 4
# # y_tiles = 4
# # tile_size = 23040  # Size of each tile in meters
#
# # # Calculate the corners
# # # corners = calculate_utm_and_latlon_corners(latitude, longitude, x_tiles, y_tiles, tile_size)
# # # print(corners)
#
# # scenery = Scenery({"lat": latitude, "lon": longitude}, tiles_x=4)
# # scenery.to_shapefile()
# # print(scenery)
# # # scenery._fetch()
#
