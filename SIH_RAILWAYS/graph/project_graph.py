import os
import geopandas as gpd
import networkx as nx
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Proj, transform
import warnings
import json

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def get_feature_centroid(feature):
    """Calculates the centroid for any geometry type."""
    geom = feature.geometry
    return None if geom is None or geom.is_empty else geom.centroid


def create_godot_export(geojson_path=None, stations_path=None):
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if geojson_path is None:
        geojson_path = os.path.join(_root, 'data', 'raw', 'export (2).geojson')
    if stations_path is None:
        stations_path = os.path.join(_root, 'data', 'processed', 'stations_only_nodes.csv')
    """
    Processes raw data to create a single, clean JSON file
    specifically for use in Godot, including both stations and signals as nodes.
    """
    print("--- Step 1: Loading Raw Data ---")
    gdf = gpd.read_file(geojson_path)
    gdf['unique_id'] = range(len(gdf))
    stations_df_raw = pd.read_csv(stations_path)
    print(f"Loaded {len(gdf)} map features and {len(stations_df_raw)} station points.")

    print("\n--- Step 2: Processing Nodes (Stations and Signals) ---")
    # Filter both 'station' and 'signal'
    nodes_gdf = gdf[gdf['railway'].isin(['station', 'signal'])].copy()
    nodes_gdf['centroid'] = nodes_gdf.apply(get_feature_centroid, axis=1)

    nodes_data = []
    for _, feature in nodes_gdf.iterrows():
        props, centroid = feature.to_dict(), feature['centroid']
        if centroid:
            nodes_data.append({
                'id': props['unique_id'],
                'name': props.get('name', f"Unnamed {props.get('railway', '')}"),
                'type': props.get('railway', 'unknown'),  # station or signal
                'lon': centroid.x,
                'lat': centroid.y
            })

    nodes_df = pd.DataFrame(nodes_data)
    print(f"Created {len(nodes_df)} nodes from 'station' and 'signal' features.")

    print("\n--- Step 3: Projecting Coordinates for Map Layout ---")
    inProj, outProj = Proj('epsg:4326'), Proj('epsg:3857')  # WGS84 to Web Mercator
    projected_coords = [transform(inProj, outProj, lon, lat, always_xy=True) for lon, lat in
                        zip(nodes_df['lon'], nodes_df['lat'])]
    nodes_df['proj_x'] = [coord[0] for coord in projected_coords]
    nodes_df['proj_y'] = [coord[1] for coord in projected_coords]
    print("Projected node coordinates successfully.")

    print("\n--- Step 4: Creating Edges by Connecting Stations ---")
    track_features = gdf[gdf['railway'] == 'rail']
    node_coords_for_tree = list(zip(nodes_df['lon'], nodes_df['lat']))

    if not node_coords_for_tree:
        print("FATAL: No nodes found. Cannot create graph.")
        return

    tree = cKDTree(node_coords_for_tree)
    edges_data = []
    for _, track in track_features.iterrows():
        if track.geometry and track.geometry.geom_type == 'LineString' and len(track.geometry.coords) > 1:
            start_coord, end_coord = track.geometry.coords[0], track.geometry.coords[-1]
            _, idx_start = tree.query(start_coord, k=1)
            _, idx_end = tree.query(end_coord, k=1)
            start_node_id, end_node_id = nodes_df.iloc[idx_start]['id'], nodes_df.iloc[idx_end]['id']
            if start_node_id != end_node_id:
                edges_data.append({'source': start_node_id, 'target': end_node_id})

    edges_df = pd.DataFrame(edges_data).drop_duplicates()
    print(f"Created {len(edges_df)} unique edges connecting nodes.")

    print("\n--- Step 5: Assembling and Exporting JSON for Godot ---")
    min_x = nodes_df['proj_x'].min()
    min_y = nodes_df['proj_y'].min()

    godot_nodes = []
    for _, row in nodes_df.iterrows():
        godot_nodes.append({
            "id": row['id'],
            "label": row['name'],
            "type": row['type'],  # include type for Godot to render differently if needed
            "pos_x": (row['proj_x'] - min_x) * 0.1,  # Scale down for visibility
            "pos_y": (row['proj_y'] - min_y) * -0.1  # Scale and invert Y
        })

    godot_edges = edges_df.to_dict('records')

    output_data = {"nodes": godot_nodes, "edges": godot_edges}
    output_filename = "godot_station_signal_graph.json"
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"\n✅ SUCCESS! Graph exported to '{output_filename}'.")
    print("You can now copy this file into your Godot project.")


if __name__ == '__main__':
    create_godot_export()
