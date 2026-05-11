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