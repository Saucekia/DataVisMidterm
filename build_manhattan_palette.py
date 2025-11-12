import os, io, csv, time, json, pathlib
from typing import Tuple, List
import requests
import numpy as np
from PIL import Image


DATASET_ID = "9nt8-h7nd"  # 2020 NTAs

NTA_GEOJSON_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.geojson?$limit=5000"

TARGET_BORO = "Manhattan"  

DOWNLOAD_TILES = True
ZOOM = 16
IMG_SIZE = (512, 512)
TILE_DIR = pathlib.Path("tiles_manhattan")
OUT_CSV = pathlib.Path("manhattan_colors.csv")

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
MAPBOX_STYLE = "mapbox/satellite-v9"

SOCRATA_TOKEN = os.getenv("SOCRATA_APP_TOKEN")  # optional


def http_get(url, params=None):
    headers = {}
    if SOCRATA_TOKEN:
        headers["X-App-Token"] = SOCRATA_TOKEN
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r

def approx_polygon_centroid(coords):

    if not coords:
        raise ValueError("empty coords")
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(len(coords)-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        cross = x1*y2 - x2*y1
        A += cross
        Cx += (x1 + x2) * cross
        Cy += (y1 + y2) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    Cx /= (6*A)
    Cy /= (6*A)
    return (Cx, Cy)

def get_outer_ring(geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
    
        ring = max(coords, key=lambda r: len(r))
        return [(float(x), float(y)) for x, y in ring]
    elif gtype == "MultiPolygon":
        rings = [max(poly, key=lambda r: len(r)) for poly in coords]
        ring = max(rings, key=lambda r: len(r))
        return [(float(x), float(y)) for x, y in ring]
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")

def fetch_ntas_geojson():
    print("Requesting NTA GeoJSON (no $select)…")
    r = http_get(NTA_GEOJSON_URL)
    gj = r.json()
    feats = gj.get("features", [])
    print(f" → received {len(feats)} features")
    if feats:
        print("First feature property keys:", list(feats[0].get("properties", {}).keys()))
    return feats

def fetch_mapbox_tile(lon: float, lat: float):
    if not MAPBOX_TOKEN:
        raise RuntimeError("MAPBOX_TOKEN is not set. Export MAPBOX_TOKEN or set DOWNLOAD_TILES=False.")
    w, h = IMG_SIZE
    url = f"https://api.mapbox.com/styles/v1/{MAPBOX_STYLE}/static/{lon},{lat},{ZOOM},0/{w}x{h}?access_token={MAPBOX_TOKEN}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def average_color(img: Image.Image):
    small = img.resize((100, 100), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    return int(r), int(g), int(b)

def to_hex(r:int,g:int,b:int):
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    TILE_DIR.mkdir(parents=True, exist_ok=True)

    feats = fetch_ntas_geojson()
    if not feats:
        print("No features returned. Try again with a SOCRATA_APP_TOKEN or later.")
        return

  
    def prop(props, keylist, default=""):
        for k in keylist:
            if k in props and props[k] not in (None, ""):
                return props[k]
        return default

    
    manhattan_feats = []
    for f in feats:
        p = f.get("properties", {})
        boro = prop(p, ["boro_name","BoroName","boro","BORO_NAME"]).strip()
        if boro.lower() == TARGET_BORO.lower():
            manhattan_feats.append(f)


    if not manhattan_feats:
        print("No features matched borough name; falling back to NTA code prefix 'MN'…")
        for f in feats:
            p = f.get("properties", {})
            code = prop(p, ["nta2020","NTACode","ntacode","NTA2020"]).strip()
            if code.upper().startswith("MN"):
                manhattan_feats.append(f)

    print(f"Selected Manhattan features: {len(manhattan_feats)}")
    if not manhattan_feats:
        print("Still 0 after fallback; inspect the printed property keys above and we’ll adapt.")
        return

    rows = []
    for i, f in enumerate(manhattan_feats, 1):
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        name = prop(props, ["ntaname","NTAName","nta_name","NTAName2020"], "Unknown")
        code = prop(props, ["nta2020","NTACode","ntacode","NTA2020"], "Unknown")

        try:
            ring = get_outer_ring(geom)
            lon, lat = approx_polygon_centroid(ring)
        except Exception as e:
            print(f"[WARN] centroid failed for {name}: {e}")
            continue

        r = g = b = None
        hexcol = ""

        if DOWNLOAD_TILES:
            try:
                tile_path = TILE_DIR / f"{code.replace(' ', '_')}.jpg"
                if tile_path.exists():
                    img = Image.open(tile_path).convert("RGB")
                else:
                    img = fetch_mapbox_tile(lon, lat)
                    img.save(tile_path, "JPEG", quality=90)
                r, g, b = average_color(img)
                hexcol = to_hex(r, g, b)
            except Exception as e:
                print(f"[WARN] tile/color failed for {name}: {e}")

        rows.append({
            "nta2020": code,
            "ntaname": name,
            "borough": TARGET_BORO,
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "r": r if r is not None else "",
            "g": g if g is not None else "",
            "b": b if b is not None else "",
            "hex": hexcol
        })

        if i % 10 == 0:
            print(f"Processed {i}/{len(manhattan_feats)}")

        if DOWNLOAD_TILES:
            time.sleep(0.15)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["nta2020","ntaname","borough","lat","lon","r","g","b","hex"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {OUT_CSV} with {len(rows)} rows.")
    print(f"Tiles in: {TILE_DIR.resolve()}")

if __name__ == "__main__":
    main()
