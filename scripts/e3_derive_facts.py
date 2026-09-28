"""Derive per-stop and per-service facts from LTA DataMall and URA Master Plan 2019 files.

The raw files are read, never written, from the sg-bus-network-monitor checkout (set SG_BUS_RAW to override).
Only the derived tables below are written to data/sg_facts/, which is what the repository commits:

- stops.csv: stop code, name, road, the MP2019 planning area and region that contain the stop, the distance from
  the stop to the nearest planning-area boundary (metres, equirectangular approximation, good to about 1 m at
  Singapore's latitude), and the services that call at the stop.
- services.csv: service, direction, operator, category, terminals, number of stops, route length and the weekday
  first and last bus at the origin stop.

Contains information from LTA DataMall accessed 2026-09-26, and URA Master Plan 2019 Planning Area Boundary (No Sea)
from data.gov.sg, both made available under the terms of the Singapore Open Data Licence version 1.0.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.ops import nearest_points
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SG_BUS_RAW", Path(__file__).resolve().parents[2] / "sg-bus-network-monitor" / "data" / "raw"))
OUT = ROOT / "data" / "sg_facts"
SNAPSHOT = "20260926"
M_PER_DEG = 111_320.0


def main() -> None:
    stops = json.loads((RAW / "datamall" / f"BusStops_{SNAPSHOT}.json").read_text(encoding="utf-8"))
    services = json.loads((RAW / "datamall" / f"BusServices_{SNAPSHOT}.json").read_text(encoding="utf-8"))
    routes = json.loads((RAW / "datamall" / f"BusRoutes_{SNAPSHOT}.json").read_text(encoding="utf-8"))
    pa = json.loads((RAW / "mp2019_planning_area.geojson").read_text(encoding="utf-8"))

    polys = [shape(f["geometry"]) for f in pa["features"]]
    props = [f["properties"] for f in pa["features"]]
    tree = STRtree(polys)
    lat0 = math.radians(1.35)

    by_stop = defaultdict(set)
    seqs = defaultdict(list)
    for r in routes:
        by_stop[r["BusStopCode"]].add(r["ServiceNo"])
        seqs[(r["ServiceNo"], int(r["Direction"]))].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stops.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["stop_code", "stop_name", "road_name", "planning_area", "region", "boundary_dist_m", "services"])
        for s in sorted(stops, key=lambda s: s["BusStopCode"]):
            p = Point(s["Longitude"], s["Latitude"])
            area = region = ""
            dist = ""
            for i in tree.query(p, predicate="within"):
                area, region = props[i]["PLN_AREA_N"], props[i]["REGION_N"]
                # distance to the nearest edge (outer ring or hole), scaled from degrees to metres
                q = nearest_points(p, polys[i].boundary)[1]
                d = math.hypot((q.x - p.x) * M_PER_DEG * math.cos(lat0), (q.y - p.y) * M_PER_DEG)
                dist = f"{d:.1f}"
                break
            w.writerow([s["BusStopCode"], s["Description"], s["RoadName"], area, region, dist,
                        ";".join(sorted(by_stop.get(s["BusStopCode"], ()), key=_svc_key))])

    with (OUT / "services.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["service_no", "direction", "operator", "category", "origin_code", "destination_code",
                    "loop_desc", "n_stops", "route_km", "wd_first_bus", "wd_last_bus"])
        for s in sorted(services, key=lambda s: (_svc_key(s["ServiceNo"]), s["Direction"])):
            rs = sorted(seqs.get((s["ServiceNo"], int(s["Direction"])), []), key=lambda r: r["StopSequence"])
            if not rs:
                continue
            km = max(float(r["Distance"] or 0) for r in rs)
            w.writerow([s["ServiceNo"], s["Direction"], s["Operator"], s["Category"], s["OriginCode"],
                        s["DestinationCode"], s["LoopDesc"], len(rs), f"{km:.1f}", rs[0]["WD_FirstBus"],
                        rs[0]["WD_LastBus"]])
    print("wrote", OUT / "stops.csv", OUT / "services.csv")


def _svc_key(s: str):
    digits = "".join(c for c in s if c.isdigit())
    return (int(digits) if digits else 10**6, s)


if __name__ == "__main__":
    main()
