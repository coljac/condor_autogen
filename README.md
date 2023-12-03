# Condor 2 Landscape autogeneration

**Note**": This isn't in alpha yet, more testing and docs to come shortly.

This is a set of tools, written in python, to generate as much of a Condor 2 landscape as possible for you.

If you specify:
- The name of your landscape
- The lat, long of the center of your landscape;
- The number of tiles in x and y;
- A source for the ortho imagery to use;

then the tools will automate some or all of the following steps. The ones marked with ✅ are done, those with ❌ require manual intervention (though may be automated in future!).

✅ Calculate UTM zone and tile corners
✅ Download height data
✅ Make height maps raster files
✅ Download aerial imagery and make textures
✅ Fetch tree, water and land usage date from OSM
✅ Make bitmaps with tree and water coverage
❌ Color matching of imagery
✅ Make runways (depends on runway data availability)
❌ Make runway markings, add windsock
❌ Add buildings and textures

## Usage

Create a yaml configuration file following the template file.

Then either:

- Install all the python dependencies,
- and/or create a python virtual environment (recommended)
- or install docker and run with that (see below).

and run:

`condor-autogen config.yaml generate`

