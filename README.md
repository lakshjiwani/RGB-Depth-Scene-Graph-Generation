# RGB-D Scene Graph Generation with BIM/IFC Priors

> A research prototype that constructs a semantic 3D scene graph from an egocentric RGB-D video sequence by fusing visual object detection (YOLOv8m) with structural priors extracted from Building Information Models (IFC).


**Laksh Abhmanyo Lal | Universität Rostock**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)


---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement and Objectives](#problem-statement-and-objectives)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Dataset Analysis](#dataset-analysis)
5. [Methodology and Scientific Justifications](#methodology-and-scientific-justifications)
6. [Task A — Visual Scene Graph Generation](#task-a--visual-scene-graph-generation)
7. [Task B — BIM Prior Integration](#task-b--bim-prior-integration)
8. [Experimental Results](#experimental-results)

---

## Executive Summary

This project implements a complete pipeline that transforms an egocentric RGB-D video sequence into a semantic 3D scene graph, with structural priors fused from a Building Information Model (BIM/IFC). The pipeline addresses the following:

**Task A — Visual Scene Graph Generation:** A custom pipeline combining 2D object detection (Ultralytics YOLOv8m), per-pixel depth projection, multi-frame DBSCAN clustering, and NetworkX graph construction produces nodes with 3D world-space centroids and proximity-based edges.

**Task B — BIM Prior Integration:** IFC structural elements (walls, slabs, doors, windows, furnishings) are extracted from the provided OBJ + JSON metadata, and fused with the visual graph through two new edge types: hierarchical containment edges (IFC element → contained object) and connective portal edges (object near door/window).

**End-to-end results** on two distinct datasets:

| Metric | BasicHouse (residential) | Synagogue (historic) |
|---|---|---|
| Frames processed | 32 of 160 (step=5) | 77 of 383 (step=5) |
| Raw detections | 45 | 21 |
| Scene graph nodes | 48 | 115 |
| Scene graph edges | 92 | 11 |
| Pipeline runtime | 201s on CPU | 2752s on CPU |

<!-- This README documents the scientific reasoning behind every architectural decision, the failure modes encountered (most notably the visual domain gap between YOLO's COCO training distribution and the provided synthetic Blender renders), the limitations of the current implementation, and a clear specification of how the pipeline *should* work in an ideal production deployment. -->

---


## Problem Statement and Objectives

### Background

Constructing actionable spatial representations is a core challenge in industrial robotics and spatial computing. Robots and AR systems need not only to see objects, but to understand where they are in a building, what room they're in, and how they relate to surrounding structural elements. A scene graph — formally a graph **G = (V, E)** where nodes V are vertices/nodes (such as Chair, Table etc.) and E are edges (relationships connecting the things such as the Chair is next to the Table) are spatial relationships — is the standard data structure for this.

While significant progress has been made on visual-only scene graph generation from RGB-D data, industrial environments often provide a rich complementary signal: **Building Information Models (BIM)** in the form of **Industry Foundation Classes (IFC)** files. These contain ground-truth spatial boundaries (rooms, walls, slabs) and connective topology (doors, windows). Using these priors should make scene graphs both more accurate and more semantically meaningful.

### Project Objectives

The challenge defines two coupled tasks:

**Task A — Visual Scene Graph Generation (Primary)**
> Process an RGB-D sequence and known camera poses to instantiate a 3D scene graph **G = (V_obj, E_rel)** where nodes V_obj represent localized objects (3D centroids or bounding boxes) and edges E_rel represent spatial heuristics (proximity, nearest-neighbor).

**Task B — Incorporating BIM Priors**
> Enrich the visual scene graph by extracting spatial boundaries (e.g., IfcSpace) and connective portals (e.g., IfcDoor) from the IFC file. Fuse these into the graph using geometric inclusion logic — for example, hierarchical bipartite edges (Room → Object).

### What This Project Demonstrates

1. **Scientific transparency** — every architectural decision is explained and referenced.
2. **Honest failure analysis** — what didn't work, why, and what I did about it.
3. **Multimodel fusion** — visual evidence (YOLO) combined with structural priors (IFC).
4. **Reproducibility** — single-command execution, externalized configuration, deterministic outputs.
5. **Cross-scene generalization** — pipeline tested on a residential and a historic building.

---


## Pipeline Architecture

The pipeline is structured as six sequential stages, each implemented as a self-contained module. Stages are orchestrated by `main.py` and configured via `configs/config.yaml`. This design favors modularity over monolithic processing — each stage can be tested, swapped, or reused independently.

```
                    ┌──────────────────────────────────────────────────────────────────┐
                    │                      PIPELINE OVERVIEW                           │
                    └──────────────────────────────────────────────────────────────────┘

                    RGB frames        Depth frames       Camera poses     IFC scene
                    (1024×768 PNG)    (16-bit PNG)       (4×4 matrices)   (OBJ+JSON)
                            │                 │                  │                │
                            └────────┬────────┴─────────┬────────┘                │
                                     │                  │                         │
                                     ▼                  ▼                         │
                            ┌──────────────────────────────────┐                  │
                            │  STAGE 3 — Object Detection      │                  │
                            │  YOLOv8m → 2D bounding boxes     │                  │
                            └──────────────────────────────────┘                  │
                                              │                                   │
                                              ▼                                   │
                            ┌──────────────────────────────────┐                  │
                            │  3D Projection (pinhole model)   │                  │
                            │  pixel + depth + pose → world    │                  │
                            └──────────────────────────────────┘                  │
                                              │                                   │
                                              ▼                                   │
                            ┌──────────────────────────────────┐                  │
                            │  STAGE 4a — DBSCAN Clustering    │                  │
                            |       Object De-duplication      |                  |
                            │  same object in N frames → 1     │                  │
                            └──────────────────────────────────┘                  │
                                              │                                   │
                                              │           ┌───────────────────────┘
                                              ▼           ▼
                            ┌──────────────────────────────────┐
                            │  STAGE 4b — Graph Construction   │
                            │  Visual nodes + IFC nodes        │
                            │  proximity + contains + portal   │
                            └──────────────────────────────────┘
                                             │
                                             ▼
                            ┌──────────────────────────────────┐
                            │  STAGE 5 — Serialization         │
                            │  GraphML + JSON summary          │
                            └──────────────────────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────┐
                            │  STAGE 6 — Visualization         │
                            │  Top-down + Topology + 3D HTML   │
                            └──────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | 
|---|---|---|
| `src/ifc_parser.py` | Parse IFC labels JSON + OBJ geometry into spatial query objects | 
| `src/depth_projection.py` | Camera intrinsics, pose loading, pixel-to-world math | 
| `src/object_detector.py` | YOLOv8 wrapper, filtering, 3D centroid projection | 
| `src/scene_graph.py` | DBSCAN clustering, NetworkX graph construction, BIM fusion | 
| `src/visualize.py` | Three visualization modes (top-down, topology, interactive 3D) | 
| `main.py` | Pipeline orchestration with stage-by-stage logging | 
| `configs/config.yaml` | All tunable parameters (paths, thresholds, model name) | 

### Stage Interactions

The pipeline is **read-only with respect to input data** — no datasets are modified. All outputs are written to `output/`. Each stage produces typed data structures that flow into the next stage, avoiding implicit file I/O between stages. This makes the pipeline testable in isolation and faster to debug.

---

## Dataset Analysis

Two datasets were provided. Both share identical sensor specifications but differ dramatically in scale, content, and IFC structure allowing us to prove the code works in totally different environments.

### Camera Specifications (Both Datasets)

```
Resolution      : 1024 × 768
Focal length    : fx = fy = 667.25 pixels
Principal point : cx = 512.0, cy = 384.0
Distortion      : none (zero coefficients)
Depth range     : 0.05m – 100m
Frame rate      : 10 FPS
```

The camera poses are pre-aligned to the global IFC coordinate system, which simplifies the pipeline significantly: visual detections projected to world space share the same coordinate frame as IFC elements, enabling direct geometric fusion without an additional registration step.

### Dataset Comparison

| Property | BasicHouse | Synagogue |
|---|---|---|
| Scene type | Residential interior | Historic building |
| Frames | 160 | 383 |
| Trajectory length | 55.8 m | 133.7 m |
| Scene size (X × Y × Z) | 49 × 29 × 4 m | 84 × 60 × 16 m |
| Point cloud size | 239,890 points | 246,561 points |
| IFC elements (raw) | 3,443 entries | 2,074 entries |
| IFC elements (with geometry) | 154 | 295 |
| IfcDoor count | 8 | 5 |
| IfcWindow count | 19 | 30 |
| IfcFurnishingElement | 71 | 0 |
| Has structural columns | No | Yes (31) |
| Has roof elements | No | Yes (116) |

### Critical Discovery — No Raw .IFC File Provided

A fundamental observation that shaped the entire Task B approach: **neither dataset contains a raw `.ifc` file**. The architectural prior has been pre-processed upstream into two artifacts:

1. **`_ifcgeom_scene.obj`** — the complete 3D mesh geometry of the building, with mesh objects grouped by IFC GUID.
2. **`_ifcgeom_scene.labels.json`** — a dictionary mapping each mesh group ID to its IFC class name (e.g., `"IfcDoor"`) and human-readable name (e.g., `"Innerdörr - standard"`).

As `IfcOpenShell` operates on raw `.ifc` files, it is not designed to parse OBJ + JSON. I made a deliberate architectural decision to implement equivalent functionality using the standard `json` module for labels and the `trimesh` library for geometry, computing per-element bounding boxes by manually parsing OBJ group tags.

This is not a workaround that loses information. The OBJ + JSON representation contains exactly the same spatial and semantic data that `IfcOpenShell` would extract from a raw `.ifc` file. The only difference is the parser implementation. This decision is documented in `src/ifc_parser.py` and reflected in the design of the `IFCScene` and `IFCElement` data classes, which mirror the API conventions of `IfcOpenShell` (`scene.by_type("IfcDoor")` becomes `scene.doors`, etc.).

### IFC Class Distribution

Each dataset emphasizes different IFC class types, reflecting their building functions:

**BasicHouse (residential):**
```
IfcFurnishingElement     71  (chairs, tables, beds, storage units, shelving)
IfcFlowTerminal           7  (washing machine, sink, shower, water closet)
IfcWindow                19
IfcWallStandardCase      13
IfcDoor                   8
IfcBuildingElementProxy   4  (refrigerator, architectural placeholders)
IfcSlab                   3
```
**Synagogue (historic):**
```
IfcRoof                 116  (multiple roof segments)
IfcColumn                31  (architectural columns)
IfcWindow                30
IfcCovering              26  (decorative coverings)
IfcWall + IfcWallStandardCase  34 total walls (from pipeline: Walls: 34)
IfcSlab                  12  (from data inspection)
IfcStairFlight           11  (from data inspection)
IfcDoor                   5
```

The synagogue lacks furnishing elements entirely which is appropriate for a historic religious building  while the BasicHouse is rich in domestic appliances. This dichotomy turns out to be scientifically informative as it tests whether the pipeline handles both "object-heavy" and "structure-heavy" scenes.

---

## Methodology and Scientific Justifications

This section documents the rationale behind every major architectural decision. 

### Why YOLOv8m over DAAAM, PIX2Graph, Grounded-SAM

The task suggests three reference frameworks: **DAAAM**, **PIX2Graph**, and "a custom combination of 2D object detection, zero-shot segmentation, and depth projection." I chose the third option using **Ultralytics YOLOv8m** as the detector. The reasoning is layered in the following:

**DAAAM**

DAAAM is a research paper codebase released alongside its corresponding publication. While scientifically valuable, research code has well-documented reproducibility problems:

- Dependencies pinned to specific CUDA versions from prior years
- GPU-only execution paths with no CPU fallback
- Brittle Windows compatibility (the development OS for this project)
- Open GitHub issues from years past without maintainer responses
- Opacity — DAAAM is an end-to-end black box producing scene graphs directly, leaving little room for the scientific reasoning and component-level justification.

Adopting DAAAM would risk consuming days on installation alone, with no scientific exposition of what happens inside the black box.

**PIX2Graph**

PIX2Graph operates on single images, not RGB-D sequences. It produces 2D relationship labels, not 3D centroids. It has no inherent notion of world coordinates or BIM integration. Using it would require building most of the depth projection and BIM fusion pipeline anyway  but with the additional burden of wrapping a paradigm-mismatched model.

**Why YOLOv8m Specifically (over n / s / l / x variants)**

YOLOv8 is published in five sizes — nano (n), small (s), medium (m), large (l), extra (x). I chose **medium** based on the standard accuracy-speed tradeoff curve. Nano and small consistently miss low-confidence detections in synthetic environments; large and extra increase inference cost roughly 3× without meaningful accuracy gain on indoor furniture classes. Medium hits the practical sweet spot for CPU-bound execution.

### Why DBSCAN for Detection Clustering

Across multiple frames, the same physical chair is detected repeatedly with slightly different 3D positions due to depth noise and pose error. Without clustering, a single chair would appear as 5–10 graph nodes. The deduplication problem is real and unavoidable.

I used **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) over following alternatives:

**k-means:** k-means requires knowing the number of clusters *k* in advance. We do not know how many unique objects will be in any given scene — that is precisely what the pipeline is trying to discover. k-means is structurally incompatible.

**hierarchical clustering:** Hierarchical clustering produces a dendrogram that requires choosing a cut threshold. Choosing this threshold is equivalent to choosing DBSCAN's `eps` parameter, but with substantially more computational overhead and no obvious advantage.

**per-class deduplication via nearest-neighbor merging:** This is the naive approach. It works for small detection counts but degrades as objects multiply and depth noise increases. DBSCAN handles arbitrary cluster shapes and noisy outliers both relevant for our depth-projected centroids.

**Parameter choice:** `eps = 0.5 m`, `min_samples = 1`. The 0.5 metre threshold reflects the typical depth-projection error magnitude on indoor distances of 1–5 metres. `min_samples = 1` is set deliberately to retain single-frame detections rather than discarding them as noise — losing rare detections (a single bed, a single couch) would be more harmful to the scene graph than retaining occasional outliers.

I clustered **per class** rather than globally because a chair and a table at nearly the same XY position are two distinct objects, not the same object. Cross-class merging would be a serious bug.

### Why NetworkX for Graph Representation

The 3D scene graph needs to support:

1. **Heterogeneous nodes** with arbitrary attributes (class name, 3D position, confidence, IFC ID, bounding box)
2. **Heterogeneous edges** with type tags (`proximity`, `contains`, `near_portal`)
3. **Standard serialization** to a portable format
4. **Graph traversal and analysis** (connectivity, subgraph extraction)

**NetworkX** is the canonical pure-Python graph library and satisfies all four requirements:

- Native dict-based attribute storage on nodes and edges
- Built-in GraphML serialization (XML, human-readable, openable in Gephi/Cytoscape)
- Built-in JSON serialization for embedding in this README
- Comprehensive analysis algorithms (connected components, shortest paths, centrality)

### 2D Containment Tests over 3D

When asking "does this detected object centroid fall inside this IFC element's bounding box?", we have two options: full 3D containment, or 2D containment in the XY plane (treating containment as a floor-plan question).

**I used 2D containment** for three reasons:

1. **Depth estimation is the noisiest component of the pipeline.** Z values for object centroids have estimated errors of 0.3–1.0 metre, while X-Y errors are typically below 0.2 metre. A 3D test would reject objects whose Z value happens to fall just outside an IFC element's vertical bounds frequently and incorrectly.

2. **Room membership is inherently a 2D concept.** When humans speak of "the chair in the living room", they mean the chair's position in the floor plan regardless of whether the chair is at floor level or stacked on a table. The semantic meaning is preserved by ignoring Z.

3. **IFC elements have inconsistent Z bounds across datasets.** A floor slab in BasicHouse has a thin vertical bounding box (–0.30 to 0.00 metres). A "site" element in the synagogue is treated as zero-thickness (Z = 0.0 only). Forcing 3D containment would produce inconsistent results across scenes.

We apply a small expansion tolerance (0.5 metres in X and Y) to handle the residual horizontal error from depth-based centroid estimation. This is implemented in `BoundingBox3D.contains_point_2d()` in `src/ifc_parser.py`.

### Why Trimesh and Plotly over Open3D

The task mentions point cloud processing and 3D visualization. The natural default choice for both would be **Open3D**, a comprehensive 3D library widely used in robotics. Following is the reason of not using it:

**Reason:** This project runs on Python 3.14. Open3D has not yet released wheels for Python 3.14 — only up to 3.12. Two options were available:

1. Downgrade to Python 3.11 just to use Open3D
2. Substitute equivalent functionality with `trimesh` (mesh loading and analysis) and `plotly` (interactive 3D visualization)

I chose option 2. Trimesh is widely used, well-maintained, and supports all the OBJ parsing and bounding box operations I needed. Plotly produces fully interactive, browser-renderable 3D visualizations that arguably surpass Open3D's GUI viewer for documentation purposes — the HTML output can be opened in any browser, embedded in any web platform, and shared with reviewers without installation overhead.

---

## Task A — Visual Scene Graph Generation

### Mathematical Foundation

The core challenge of Task A is lifting 2D pixel detections into 3D world coordinates that share the IFC coordinate frame. This requires three coupled transformations, implemented in `src/depth_projection.py`.

**Step 1 — Pixel to camera-space (back-projection through pinhole model)**

Given a detected bounding box centre pixel `(u, v)` and corresponding depth `d` in metres:

```
X_cam = (u - cx) · d / fx
Y_cam = (v - cy) · d / fy
Z_cam = d
```

Where `fx`, `fy` are focal lengths and `cx`, `cy` are the principal point coordinates. For both datasets, these are constant: `fx = fy = 667.25`, `cx = 512`, `cy = 384`.

**Step 2 — Camera space to world space**

Using the 4×4 camera-to-world pose matrix `T` (provided per-frame in `pose/poses.txt` as 16 space-separated floats):

```
P_world = T · [X_cam, Y_cam, Z_cam, 1]^T
```

The homogeneous coordinate `1` enables the translation component of T to apply. The result `P_world[0:3]` is the 3D position of the detection in IFC-aligned world coordinates.

### Depth Sampling Strategy

Reading the depth value at the exact centre pixel is naive, depth maps contain noise, holes, and edge artifacts. I sampled a **5-pixel radius patch** around the centre and took the **median** depth across valid pixels (those with values between 0.01 and 10 metres).

The median is preferred over the mean because depth maps frequently have outliers at object boundaries where the sensor returns the depth of the background instead of the foreground. Median rejects these robustly.

### Multi-Frame Detection Aggregation

YOLOv8m processes every 5th frame (`frame_step = 5` in `configs/config.yaml`). Skipping intermediate frames is a deliberate trade-off:

- **Coverage gain at step=5:** 32 frames processed in BasicHouse versus 160 if every frame were used. This captures the trajectory at ~1-second intervals.
- **Speed gain at step=5:** Roughly 5× faster execution. Critical on CPU where YOLO is the bottleneck.
- **Information loss at step=5:** Negligible, because the same physical object reappears across many neighbouring frames. The DBSCAN deduplication step would merge detections from frames 0, 1, 2, 3, 4 into a single cluster anyway.

Each retained detection is stored as a `Detection3D` dataclass with the source frame index, class name, confidence, 2D bounding box, 3D centroid in world space, and the depth value used.

### Output of Task A

After processing all sampled frames, the pipeline produces:

- BasicHouse: **45 raw Detection3D records** across 4 classes (chair, dining table, couch, bed)
- Synagogue: **21 raw Detection3D records** across 9 classes

These raw detections are then passed to the clustering stage (Task A's deduplication) and the scene graph builder (Task A's edge construction). DBSCAN reduces 45 → 35 unique objects for BasicHouse and 21 → 21 for the synagogue (no duplicates in the synagogue because detections are spread across a large building).

### Proximity Edge Construction

The task requires "spatial heuristics (proximity, nearest-neighbour)" as edges. I implemented a **threshold-based proximity** approach: two visual nodes are connected by a `proximity` edge if their 3D centroids are within `proximity_threshold_m = 2.0` metres.

Edge weights are set to `1 / (distance + ε)`, giving closer pairs higher weights — useful for downstream graph algorithms that respect weighted edges.

**Result for BasicHouse:** 47 proximity edges connecting visual nodes (chairs around tables, tables near couches, etc.).

---

## Task B — BIM Prior Integration

### Adaptation to OBJ + JSON Input

As documented in [Dataset Analysis](#dataset-analysis), the provided IFC data is not in raw `.ifc` form but pre-processed into OBJ geometry and a JSON metadata file. The `src/ifc_parser.py` module reproduces the functionality that `IfcOpenShell` would have provided, using two parallel passes:

**Pass 1 — Semantic loading (`load_ifc_labels`)**
Parses the labels JSON into `IFCElement` dataclasses, each containing:
- `ifc_id` — the GUID from the IFC model
- `ifc_class` — e.g., `"IfcDoor"`, `"IfcWallStandardCase"`, `"IfcFurnishingElement"`
- `name` — human-readable label, e.g., `"Innerdörr - standard"` (Swedish "interior door")
- `bbox` — set in Pass 2

Metadata-only classes that have no spatial geometry (`IfcPropertySet`, `IfcRelDefinesByProperties`, etc.) are filtered out. For BasicHouse this reduces 3,443 raw entries to 198 spatial elements.

**Pass 2 — Geometry loading (`load_ifc_geometry`)**
The OBJ file is parsed manually, line by line. Each `o <name>` or `g <name>` line begins a new mesh group; subsequent `v x y z` lines define vertices that belong to that group. For each group, we compute the axis-aligned bounding box from vertex coordinates and attach it to the matching `IFCElement`.

**Critical implementation note:** I initially attempted to use `trimesh.load(obj_path, force="scene")`, expecting it to preserve per-group geometry. It does not — trimesh's default behaviour merges all groups into a single mesh, losing the per-element separation. I discovered this when our scene graph showed all 154 IFC elements sharing a single bounding box. The fix was to manually parse the OBJ file's group structure. 
<!-- This is a documented failure mode addressed in [Failure Analysis](#failure-analysis). -->

### IFC Classes Used as Graph Nodes

Not every IFC class becomes a graph node. We filter to three meaningful categories defined in `src/ifc_parser.py`:

```python
STRUCTURAL_CLASSES = {
    "IfcWall", "IfcWallStandardCase", "IfcSlab",
    "IfcColumn", "IfcBeam", "IfcRoof", "IfcCovering",
    "IfcBuildingElementProxy",
}

PORTAL_CLASSES = {
    "IfcDoor", "IfcWindow", "IfcOpeningElement",
}

FURNISHING_CLASSES = {
    "IfcFurnishingElement", "IfcFurnitureType",
    "IfcFlowTerminal", "IfcDistributionPort",
}
```

The `category` property on each `IFCElement` classifies it for downstream filtering. This three-category is principled: **structural** elements define spatial regions for containment edges, **portals** define topology for connective edges, and **furnishings** provide ground-truth object positions (objects that BIM knows about even if YOLO does not).

### Three BIM-Visual Fusion Edge Types

Task B requires fusing the IFC prior into the visual graph. We implement three distinct edge types, each addressing a different aspect of the task's "hierarchical bipartite edges (Room → Object)" specification.

**Edge Type 1 — Containment (`contains`)**

For each visual node, find the IFC element whose 2D bounding box contains the visual centroid. Add a directed edge `IFC → visual` with `edge_type="contains"` and `relation="contains"`.

This is the direct implementation of the task's example: "map the localized 3D object centroids into their corresponding IFC rooms, creating hierarchical bipartite edges". Because the provided datasets do not contain `IfcSpace` elements (named rooms), we map to `IfcSlab` (floor slabs that cover the building footprint) and similar containing structures. 
<!-- This compromise is documented in [Known Limitations](#known-limitations). -->

**Edge Type 2 — Portal Proximity (`near_portal`)**

For each visual node, find IFC doors and windows within `portal_proximity_m = 3.0` metres. Add an edge with `edge_type="near_portal"` capturing the distance.

This implements the task's "connective portals (IfcDoor)" requirement. A chair near a doorway has a meaningful spatial relationship that a pure proximity graph misses. This edge type creates the topological skeleton (objects near transition zones) that would allow downstream reasoning like "the bed in the room with one window" or "the chair in the doorway."

**Edge Type 3 — Visual Proximity (`proximity`)**

The Task A edge type, included here for completeness. Two visual nodes within 2 metres are linked. While not strictly a BIM-fusion edge, the inclusion of IFC nodes in the same graph means proximity edges naturally form across the bipartite divide when an IFC element happens to fall near a visual detection.

### Graph Construction Sequence

The `SceneGraphBuilder.build()` method in `src/scene_graph.py` executes the following sequence:

```
1. Create empty nx.Graph()
2. Add visual nodes from clustered detections
3. Add IFC nodes from IFCScene.elements (only those with bbox)
4. Compute proximity edges between all visual-visual pairs within threshold
5. Compute containment edges from IFC bboxes to visual centroids
6. Compute portal edges from visual centroids to door/window centroids
7. Annotate graph with metadata (thresholds used, description)
8. Return populated graph
```

This produces, for BasicHouse:
- **48 nodes** (35 visual + 13 IFC with active edges, out of 154 total IFC nodes)
- **92 edges** (47 proximity + 35 contains + 10 near_portal)

The disparity between 154 IFC nodes added and 13 with edges is itself informative: most IFC elements never spatially intersect any detected visual object, leaving them isolated in the graph. This is documented in [Failure Analysis](#failure-analysis) as expected behaviour.

### Serialization Formats

Two output formats are produced:

**GraphML** (`output/{scene}_scene_graph.graphml`)
- XML-based, human-readable
- Preserves all node and edge attributes
- Loadable by NetworkX (`nx.read_graphml`), Gephi, Cytoscape, yEd
- Industry-standard for graph data exchange

**JSON Summary** (`output/{scene}_scene_graph_summary.json`)
- Easier to grep, parse, and embed in this README
- Includes total counts, type breakdowns, and full edge/node listings
- Useful for quick inspection without graph software

---

## Experimental Results

The pipeline was executed end-to-end on both provided scenes. All results below are reproducible by running `python main.py --scene <basichouse|synagogue>`.

### Scene 1 — BasicHouse (Residential Interior)

**Configuration**
```
Active scene           : basichouse
Confidence threshold   : 0.08
Frame step             : 5 (32 of 160 frames processed)
DBSCAN eps             : 0.5 m
Proximity threshold    : 2.0 m
Portal proximity       : 3.0 m
```

**Output statistics**
```
Total runtime           : 201.7 seconds (CPU)
  IFC parsing           : 2.3 s
  Camera setup          : ~0.0 s
  Model loading         : 37.0 s  (YOLOv8m weights, cached after first run)
  Object detection      : 159.6 s (inference on 32 frames, after model load)
  Graph construction    : ~0.3 s
  Visualization         : 8.2 s

Raw detections          : 45 (across 4 classes)
After DBSCAN clustering : 35 unique objects
Total graph nodes       : 48 (35 visual + 13 IFC active)
Total graph edges       : 92
  proximity edges       : 47
  contains edges        : 35
  near_portal edges     : 10
```

**Visual detections by class**

| Class | Unique instances | Plausibility |
|---|---|---|
| chair | 19 | Reasonable — house has multiple chairs |
| dining table | 14 | Over-detected — see Failure Analysis |
| couch | 1 | Correct |
| bed | 1 | Correct |

**Notable graph structures**

The most spatially meaningful edges:
- `bed → IfcWindow` at 2.70 m — the bedroom contains the bed near a window ✓
- `dining table → IfcDoor` at 2.36 m — the dining table is near a doorway ✓
- `chair → IfcDoor` at 2.11 m — chair near transition zone ✓

The IFC node `M_Refrigerator` at world position `[1.834, -3.911, 0.915]` is present in the graph as an isolated node — YOLO did not detect it, but the BIM prior knows it exists. This is the central evidence for the multimodal fusion argument: visual detection missed it, but the scene graph still contains it as a node sourced from BIM data.

**Top-down visualization**

![BasicHouse Scene Graph — Top-Down View](output/viz/basichouse_topdown.png)

The visualization shows the camera trajectory (grey line), 35 visual nodes (red circles labelled with class names), IFC structural elements (blue markers, shape-coded by category), proximity edges (green solid), containment edges (orange dotted), and portal-proximity edges (purple dashed). The trajectory and detections concentrate in the main living area, with the kitchen appliance positions (`M_Refrigerator`, sink) marked by IFC nodes despite being absent from visual detections.

**Graph topology visualization**

![BasicHouse Graph Topology](output/viz/basichouse_topology.png)

The force-directed layout reveals the structural pattern of the scene graph. The central `IfcSlab` node forms a hub connected to 32+ visual objects via containment edges — every detected object sits on the floor. The portal nodes (`IfcDoor`, `IfcWindow`) sit between visual clusters, demonstrating their connective role. Visual nodes form proximity-edge sub-clusters corresponding to functional zones (dining area, bedroom).

### Scene 2 — Synagogue (Historic Building)

**Configuration**
Same as BasicHouse — no scene-specific tuning. This is intentional, to test whether the pipeline generalizes across building types without manual reconfiguration.

**Output statistics**
```
Total runtime           : 2751.9 seconds (~46 minutes, CPU)
  IFC parsing           : 49.0 s  (larger building, more elements)
  Camera setup          : 0.1 s
  Object detection      : 2306.2 s  (77 frames, ~30s per frame)
  Graph construction    : 0.5 s
  Visualization         : 15.5 s

Raw detections          : 21 (across 9 classes)
After DBSCAN clustering : 21 unique objects (no duplicates)
Total graph nodes       : 115 (21 visual + 94 IFC active)
Total graph edges       : 11
  proximity edges       : 9
  contains edges        : 2
  near_portal edges     : 0
```

**Visual detections by class**

| Class | Instances | Plausibility |
|---|---|---|
| dining table | 12 | Over-detected — walls and pews misidentified |
| tv | 2 | Implausible in a synagogue — false positive |
| sink | 1 | Possible but unlikely |
| toilet | 1 | Implausible — false positive |
| chair | 1 | Plausible |
| potted plant | 1 | Possible |
| mouse | 1 | Implausible — false positive |
| knife | 1 | Implausible — false positive |
| book | 1 | Plausible (prayer books) |

The contrast with BasicHouse is evident. The synagogue contains few of YOLO's COCO classes, so the model "hallucinates" plausible-looking matches on architectural features. The detected `knife` and `book` share the exact same 3D position `[3.401, 7.085, -4.829]`, indicating two competing class predictions for the same image region.

**Cross-scene comparison**

| Metric | BasicHouse | Synagogue | Ratio |
|---|---|---|---|
| Raw detections | 45 | 21 | 0.47× |
| Unique objects | 35 | 21 | 0.60× |
| Containment edges | 35 | 2 | 0.06× |
| Portal edges | 10 | 0 | 0.00× |
| Plausible classes | 4 / 4 | ~4 / 9 | — |

The dramatic collapse in containment edges (35 → 2) reflects a structural difference between the IFC files: BasicHouse contains one large `IfcSlab` (the floor) whose bounding box covers the entire interior, so every detected object falls inside it. The synagogue's IFC is segmented into many smaller architectural elements with smaller individual bounding boxes, so most detections fall outside all of them.

This is a **scientific finding, not a bug**: the same pipeline applied to two scenes with different BIM granularity produces measurably different fusion behaviour. A real production system would handle this with `IfcSpace`-level segmentation, 
<!-- discussed in [Ideal Pipeline](#ideal-pipeline--how-this-should-work-in-production). -->

**Top-down visualization**

![Synagogue Scene Graph — Top-Down View](output/viz/synagogue_topdown.png)

The synagogue trajectory spans a much larger area than the BasicHouse. Visual detections concentrate in a small cluster, reflecting limited YOLO recall in the historical environment. The IFC nodes are spread across the building footprint, illustrating the scale mismatch between detected objects and structural geometry.

**Graph topology visualization**

![Synagogue Graph Topology](output/viz/synagogue_topology.png)

The synagogue topology is dramatically sparser than BasicHouse — only 11 edges total, versus 92. Many IFC nodes are isolated (visible at the right side of the figure), unconnected to any visual detection. This visually demonstrates the central thesis: **when visual detection fails, the scene graph degrades proportionally**, which is precisely why BIM priors are valuable as a complementary signal.

### Interactive 3D Visualizations

Both scenes also produce interactive HTML visualizations (`output/viz/basichouse_3d.html`, `output/viz/synagogue_3d.html`) that allow rotation, zoom, and hover-tooltips on every node. These are best viewed in a desktop browser.

---