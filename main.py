"""
main.py
───────────────
Single entry point for the RGB-D Scene Graph Generation pipeline.

Usage:
    python main.py --scene basichouse
    python main.py --scene synagogue
    python main.py --scene basichouse --frame_step 10
    python main.py --scene basichouse --no_viz

Assignment Context:
    The assignment requires "reproducible scripts" as a deliverable.
    This script runs the complete pipeline end-to-end:

      1. Parse IFC scene (Task B foundation)
      2. Load camera intrinsics + poses
      3. Run YOLOv8 object detection + 3D projection (Task A)
      4. Cluster detections to remove duplicates
      5. Build scene graph with BIM fusion (Tasks A + B)
      6. Save graph outputs (GraphML + JSON summary)
      7. Generate visualizations (top-down, topology, 3D interactive)

Pipeline Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  RGB-D frames + Depth frames + Camera poses         │
    │           │                                         │
    │           ▼                                         │
    │    YOLOv8m detection → 3D projection                │
    │           │                                         │
    │           ▼                                         │
    │    DBSCAN clustering (deduplication)                │
    │           │              ┌──────────────────┐       │
    │           ▼              │  IFC labels.json │       │
    │    Scene Graph Builder ◄─│  IFC geometry.obj│       │
    │           │              └──────────────────┘       │
    │           ▼                                         │
    │    NetworkX Graph (nodes + edges)                   │
    │           │                                         │
    │           ▼                                         │
    │    Visualization + Serialization                    │
    └─────────────────────────────────────────────────────┘

"""

import os
import sys
import time
import logging
import argparse

import yaml
import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline")


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Loads pipeline configuration from YAML file.

    Externalizing parameters into config.yaml means the pipeline
    can be reconfigured without touching source code — a standard
    software engineering practice for reproducible research.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_parse_ifc(cfg: dict, scene: str):
    """
    Stage 1 — Parse IFC scene from labels JSON and OBJ geometry.

    This is the Task B foundation. IFC elements provide:
      - Structural boundaries (walls, slabs)
      - Portal elements (doors, windows)
      - Furnishing positions (appliances, fixtures)

    These are used later in BIM fusion to enrich the visual graph.
    """
    from src.ifc_parser import parse_ifc_scene

    scene_cfg   = cfg["datasets"][scene]
    labels_path = scene_cfg["ifc_labels"]
    obj_path    = scene_cfg["ifc_obj"]

    logger.info("=" * 55)
    logger.info("STAGE 1 — IFC Scene Parsing (Task B)")
    logger.info("=" * 55)

    t0        = time.time()
    ifc_scene = parse_ifc_scene(labels_path, obj_path)
    elapsed   = time.time() - t0

    logger.info(f"IFC parsing complete in {elapsed:.1f}s")
    logger.info(f"  Elements   : {len(ifc_scene.elements)}")
    logger.info(f"  With bbox  : "
                f"{sum(1 for e in ifc_scene.elements.values() if e.bbox)}")
    logger.info(f"  Doors      : {len(ifc_scene.doors)}")
    logger.info(f"  Windows    : {len(ifc_scene.windows)}")
    logger.info(f"  Furnishings: {len(ifc_scene.furnishings)}")

    return ifc_scene


def stage_load_camera(cfg: dict, scene: str):
    """
    Stage 2 — Load camera intrinsics and poses.

    The assignment states poses are "already aligned to the global
    IFC coordinate system", which means detected 3D objects and IFC
    elements share the same coordinate frame — enabling direct fusion.
    """
    from src.depth_projection import CameraIntrinsics, PoseLoader

    scene_cfg = cfg["datasets"][scene]

    logger.info("=" * 55)
    logger.info("STAGE 2 — Camera Setup")
    logger.info("=" * 55)

    intrinsics = CameraIntrinsics.from_json(scene_cfg["camera_info"])
    poses      = PoseLoader(scene_cfg["pose_file"])

    logger.info(f"Intrinsics : {intrinsics}")
    logger.info(f"Poses      : {len(poses)} frames loaded")

    return intrinsics, poses


def stage_detect_objects(
    cfg: dict,
    scene: str,
    intrinsics,
    poses,
    frame_step: int = None,
):
    """
    Stage 3 — YOLOv8 detection + 3D projection (Task A).

    For each sampled frame:
      1. YOLOv8m detects objects in the RGB image
      2. Each detection's center pixel is looked up in the depth map
      3. Depth + camera intrinsics → 3D camera-space point
      4. Camera pose matrix → 3D world-space point

    Result: a list of Detection3D objects with world coordinates.
    """
    from src.depth_projection import FrameIterator
    from src.object_detector import YOLOv8Detector, detect_and_project

    scene_cfg   = cfg["datasets"][scene]
    det_cfg     = cfg["detection"]
    step        = frame_step or det_cfg.get("frame_step", 5)

    logger.info("=" * 55)
    logger.info("STAGE 3 — Object Detection + 3D Projection (Task A)")
    logger.info("=" * 55)
    logger.info(f"  Model      : {det_cfg['model']}")
    logger.info(f"  Confidence : {det_cfg['confidence_threshold']}")
    logger.info(f"  Frame step : {step}")

    # Load YOLOv8
    detector = YOLOv8Detector(
        model_name=det_cfg["model"],
        confidence_threshold=det_cfg["confidence_threshold"],
    )

    # Iterate frames
    iterator = FrameIterator(
        rgb_dir=scene_cfg["rgb_dir"],
        depth_dir=scene_cfg["depth_dir"],
        pose_loader=poses,
        frame_step=step,
        max_depth_m=cfg["spatial"]["max_depth_m"],
    )

    all_detections = []
    t0             = time.time()

    for rgb, depth, pose, idx in iterator:
        dets = detect_and_project(
            detector=detector,
            rgb_frame=rgb,
            depth_frame=depth,
            pose_matrix=pose,
            intrinsics=intrinsics,
            frame_idx=idx,
            max_depth_m=cfg["spatial"]["max_depth_m"],
        )
        all_detections.extend(dets)

    elapsed = time.time() - t0

    # Count by class
    from collections import Counter
    class_counts = Counter(d.class_name for d in all_detections)

    logger.info(f"Detection complete in {elapsed:.1f}s")
    logger.info(f"  Raw detections : {len(all_detections)}")
    logger.info(f"  Classes found  :")
    for cls, count in class_counts.most_common():
        logger.info(f"    {cls:<20} {count}")

    return all_detections


def stage_build_graph(
    cfg: dict,
    all_detections: list,
    ifc_scene,
):
    """
    Stage 4 — Cluster detections and build scene graph (Tasks A + B).

    Two sub-steps:
      a) DBSCAN clustering: merge detections of the same object
         seen across multiple frames into one canonical node

      b) Graph construction:
         - Visual nodes from clustered detections (Task A)
         - IFC nodes from BIM model (Task B)
         - Proximity edges between nearby visual objects (Task A)
         - Containment edges from IFC elements to objects (Task B)
         - Portal edges from objects to nearby doors/windows (Task B)
    """
    from src.scene_graph import (
        SceneGraphBuilder,
        cluster_detections,
        print_graph_summary,
    )

    spatial_cfg = cfg["spatial"]

    logger.info("=" * 55)
    logger.info("STAGE 4 — Scene Graph Construction (Tasks A + B)")
    logger.info("=" * 55)

    # Cluster detections
    clustered = cluster_detections(
        all_detections,
        eps_m=spatial_cfg["clustering_eps_m"],
        min_samples=1,
    )
    logger.info(f"  Clustering     : {len(all_detections)} raw → "
                f"{len(clustered)} unique objects")

    # Build graph
    builder = SceneGraphBuilder(
        proximity_threshold_m=spatial_cfg["proximity_threshold_m"],
        containment_tolerance=0.5,
        portal_proximity_m=3.0,
    )
    G = builder.build(clustered, ifc_scene)

    print_graph_summary(G)

    return G, builder


def stage_save_outputs(
    cfg: dict,
    G,
    builder,
    scene: str,
):
    """
    Stage 5 — Save graph to disk.

    Outputs:
      - GraphML file: machine-readable, loadable by NetworkX/Gephi
      - JSON summary: human-readable, embeddable in README
    """
    logger.info("=" * 55)
    logger.info("STAGE 5 — Saving Graph Outputs")
    logger.info("=" * 55)

    out_cfg = cfg["output"]
    os.makedirs(out_cfg["dir"], exist_ok=True)

    graphml_path = os.path.join(out_cfg["dir"], f"{scene}_scene_graph.graphml")
    json_path    = os.path.join(out_cfg["dir"], f"{scene}_scene_graph_summary.json")

    builder.save(G, graphml_path)
    builder.save_summary(G, json_path)

    logger.info(f"  GraphML  : {graphml_path}")
    logger.info(f"  JSON     : {json_path}")


def stage_visualize(
    cfg: dict,
    G,
    poses,
    scene: str,
):
    """
    Stage 6 — Generate visualizations.

    Produces three outputs:
      - Top-down PNG: floor plan with trajectory + nodes + edges
      - Topology PNG: abstract graph connectivity
      - 3D HTML: interactive rotatable scene

    These are the visual evidence embedded in the README.
    """
    from src.visualize import visualize_scene_graph

    logger.info("=" * 55)
    logger.info("STAGE 6 — Visualization")
    logger.info("=" * 55)

    viz_dir = os.path.join(cfg["output"]["dir"], "viz")

    visualize_scene_graph(
        G,
        output_dir=viz_dir,
        pose_loader=poses,
        scene_name=scene,
    )

    logger.info(f"  Top-down PNG : {viz_dir}/{scene}_topdown.png")
    logger.info(f"  Topology PNG : {viz_dir}/{scene}_topology.png")
    logger.info(f"  3D HTML      : {viz_dir}/{scene}_3d.html")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RGB-D Scene Graph Generation with BIM/IFC Priors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --scene basichouse
  python main.py --scene synagogue
  python main.py --scene basichouse --frame_step 10
  python main.py --scene basichouse --no_viz
        """,
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=["basichouse", "synagogue"],
        help="Which scene to process (default: uses active_scene from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=None,
        help="Process every Nth frame (overrides config value)",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip visualization generation (faster for testing)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    scene  = args.scene or cfg.get("active_scene", "basichouse")

    logger.info("╔" + "═" * 53 + "╗")
    logger.info("║  RGB-D Scene Graph Generation with BIM/IFC Priors  ║")
    logger.info("╚" + "═" * 53 + "╝")
    logger.info(f"Scene    : {scene}")
    logger.info(f"Config   : {args.config}")
    logger.info(f"Viz      : {'disabled' if args.no_viz else 'enabled'}")

    pipeline_start = time.time()

    # ── Validate data exists ──────────────────────────────────────────────────
    scene_cfg = cfg["datasets"][scene]
    if not os.path.exists(scene_cfg["data_dir"]):
        logger.error(f"Data directory not found: {scene_cfg['data_dir']}")
        logger.error("Please unzip the dataset into the data/ folder.")
        logger.error("See README.md for data setup instructions.")
        sys.exit(1)

    # ── Run pipeline stages ───────────────────────────────────────────────────
    ifc_scene            = stage_parse_ifc(cfg, scene)
    intrinsics, poses    = stage_load_camera(cfg, scene)
    all_detections       = stage_detect_objects(
        cfg, scene, intrinsics, poses,
        frame_step=args.frame_step,
    )
    G, builder           = stage_build_graph(cfg, all_detections, ifc_scene)
    stage_save_outputs(cfg, G, builder, scene)

    if not args.no_viz:
        stage_visualize(cfg, G, poses, scene)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    logger.info("╔" + "═" * 53 + "╗")
    logger.info("║                PIPELINE COMPLETE                   ║")
    logger.info("╚" + "═" * 53 + "╝")
    logger.info(f"Scene          : {scene}")
    logger.info(f"Total nodes    : {G.number_of_nodes()}")
    logger.info(f"Total edges    : {G.number_of_edges()}")
    logger.info(f"Total time     : {total_time:.1f}s")
    logger.info(f"Outputs saved  : output/{scene}_scene_graph.graphml")
    logger.info(f"                 output/{scene}_scene_graph_summary.json")
    if not args.no_viz:
        logger.info(f"Visualizations : output/viz/{scene}_topdown.png")
        logger.info(f"                 output/viz/{scene}_topology.png")
        logger.info(f"                 output/viz/{scene}_3d.html")


if __name__ == "__main__":
    main()