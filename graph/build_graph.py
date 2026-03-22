# Re-run graph generation, lighter and more robust.
# If networkx.write_graphml or write_gexf fails due to missing optional deps, still produce CSVs.
import json, math, os
from collections import defaultdict
try:
    import networkx as nx
except Exception as e:
    raise RuntimeError(f"networkx import failed: {e}")

# Resolve project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INFILE = os.path.join(_PROJECT_ROOT, "data", "raw", "west_bengal_stations.json")
OUT_GRAPHML = os.path.join(_PROJECT_ROOT, "data", "processed", "west_bengal_graph.graphml")
OUT_GEXF = os.path.join(_PROJECT_ROOT, "data", "processed", "west_bengal_graph.gexf")
OUT_NODES_CSV = os.path.join(_PROJECT_ROOT, "data", "processed", "west_bengal_nodes.csv")
OUT_EDGES_CSV = os.path.join(_PROJECT_ROOT, "data", "processed", "west_bengal_edges.csv")
K_NEAREST_NEIGHBORS = 2

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

with open(INFILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data if isinstance(data, list) else data.get("features", data.get("features", []))

stations = []
for idx, feat in enumerate(features):
    props = feat.get("properties", {}) if isinstance(feat, dict) else {}
    geom = feat.get("geometry") if isinstance(feat, dict) else None
    if not geom or not geom.get("coordinates"):
        continue
    coords = geom["coordinates"]
    try:
        lon, lat = float(coords[0]), float(coords[1])
    except Exception:
        continue
    code = (props.get("code") or "").strip()
    name = (props.get("name") or "").strip()
    zone = props.get("zone")
    stations.append({
        "idx": idx,
        "id": code if code else f"N{idx}",
        "code": code if code else "",
        "name": name if name else "",
        "zone": zone if zone else "",
        "lon": lon,
        "lat": lat
    })

# ensure unique ids
seen = defaultdict(int)
for s in stations:
    base = s["id"] or s["name"] or f"N{s['idx']}"
    if seen[base] > 0:
        s["id"] = f"{base}_{seen[base]}"
    else:
        s["id"] = base if base else f"N{s['idx']}"
    seen[base] += 1

G = nx.Graph()
for s in stations:
    G.add_node(s["id"], code=s["code"], name=s["name"], zone=s["zone"], lon=s["lon"], lat=s["lat"])

coords = [(s["id"], s["lon"], s["lat"]) for s in stations]
for i, (nid, lon1, lat1) in enumerate(coords):
    dists = []
    for j, (nid2, lon2, lat2) in enumerate(coords):
        if nid == nid2:
            continue
        dkm = haversine_km(lon1, lat1, lon2, lat2)
        dists.append((dkm, nid2))
    dists.sort(key=lambda x: x[0])
    for k in range(min(K_NEAREST_NEIGHBORS, len(dists))):
        dkm, nbr = dists[k]
        if not G.has_edge(nid, nbr):
            G.add_edge(nid, nbr, distance_km=float(dkm))

# write CSVs (always), then attempt graph formats
import csv
with open(OUT_NODES_CSV, 'w', newline='', encoding='utf-8') as nf:
    writer = csv.DictWriter(nf, fieldnames=["id","code","name","zone","lon","lat"])
    writer.writeheader()
    for n in G.nodes():
        d = G.nodes[n]
        writer.writerow({"id": n, "code": d.get("code",""), "name": d.get("name",""), "zone": d.get("zone",""), "lon": d.get("lon"), "lat": d.get("lat")})

with open(OUT_EDGES_CSV, 'w', newline='', encoding='utf-8') as ef:
    writer = csv.DictWriter(ef, fieldnames=["source","target","distance_km"])
    writer.writeheader()
    for u, v, d in G.edges(data=True):
        writer.writerow({"source": u, "target": v, "distance_km": d.get("distance_km", d.get("distance",0.0))})

graphml_ok = False
gexf_ok = False
try:
    nx.write_graphml(G, OUT_GRAPHML)
    graphml_ok = True
except Exception as e:
    graphml_ok = False
    graphml_err = str(e)

try:
    nx.write_gexf(G, OUT_GEXF)
    gexf_ok = True
except Exception as e:
    gexf_ok = False
    gexf_err = str(e)

result = {
    "nodesppg_written_csv": OUT_NODES_CSV,
    "edgesppg_written_csv": OUT_EDGES_CSV,
    "graphmlppg_written": OUT_GRAPHML if graphml_ok else None,
    "gexfppg_written": OUT_GEXF if gexf_ok else None,
    "graphml_error": graphml_err if not graphml_ok else None,
    "gexf_error": gexf_err if not gexf_ok else None,
    "num_nodes": G.number_of_nodes(),
    "num_edges": G.number_of_edges(),
    "k_nearest": K_NEAREST_NEIGHBORS
}
result

