
# Define the bounding box (south, west, north, east)
bbox = (south_lat, west_lon, north_lat, east_lon)

# Define the custom filter
custom_filter = '["landcover"="trees"]'

# Download the geometries from OSM
gdf = ox.geometries_from_bbox(north_lat, south_lat, east_lon, west_lon, custom_filter)

# Reproject if necessary
gdf = gdf.to_crs("EPSG:"32755)

# Save the data
gdf.to_file("trees_landcover.geojson", driver='GeoJSON')
