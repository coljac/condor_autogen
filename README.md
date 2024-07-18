# Condor 2 Landscape autogeneration

**Note**": This isn't in alpha yet, more testing and docs to come shortly.

This is a set of tools, written in python, to generate as much of a Condor 2 landscape as possible for you. The ultimate goal is to take a few bits of information such as a lat/long and airport name, and product a flyable landscape in a few minutes.

If you specify:
- The name of your landscape
- The lat, long of the center of your landscape;
- The number of tiles in x and y;
- A source for the ortho imagery to use;

then the tools will automate some or all of the following steps for you. The ones marked with ✅ are done, those with ❌ require manual intervention (though may be automated in future!).

- ✅ Calculate UTM zone and tile corners
- ✅ Download height data
- ✅ Make height maps raster files
- ✅ Download aerial imagery and make textures
- ✅ Fetch tree and water data from OSM
- ✅ Make forest maps (one tree type only)
- ✅ Make textures with water alpha
- ✅ Make rudimentary thermal maps
- ✅ Export thermal map
- ❌ Color matching of imagery
- ❌ Generate forest and terrain hashes
- ✅ Make runways (depends on runway data availability)
- ❌ Add runway markings, add windsock
- ❌ Add buildings and other landscape objects

## Texture imagery

The tool can fetch orthographic imagery from a variety of sources. Your choices are:

- A WMTS tile server (a web service that gives an image of a square of earth at a particular zoom level). Examples include USGS Earth Explorer (via arcgis), Google Maps, Bing.
- A WMS server (a service that provides images in a variety of types and projections)

If you use a public imaging service, you should be sure you have appropriate permissions, especially if you plan to redistribute the landscape.

In the next version I plan to include functionality to automatically generate  synthetic textures, bypassing the need to fetch imagery (but foregoing the realism of ortho imaging).

## Usage

Create a yaml configuration file following the template file.

Then either:

- Install all the python dependencies,
- and/or create a python virtual environment (recommended)
- or install docker and run with that (see below).

and run:

`condor-autogen config.yaml generate`

## Non-automatable steps

- I don't know the hash algorithm, so can't make forest/terrain hashes. This may be secret on purpose.

## Wish list
- AI tree placement
- AI buildings
