"""
scene_graph.py
──────────────
Constructs the final 3D scene graph by combining:
  - Visual detections from object_detector.py (Task A)
  - IFC structural elements from ifc_parser.py  (Task B)

Assignment Context:
    Task A requires: G = (V_obj, E_rel)
      V_obj → object nodes with 3D centroids
      E_rel → spatial edges (proximity, nearest-neighbor)

    Task B requires:
      - Mapping object centroids into IFC spaces
      - Creating hierarchical bipartite edges (Room → Object)
      - Adding connective portal edges (Door connects spaces)

Graph Structure:
    Node types:
      "visual"      → detected by YOLOv8 (chair, table, couch...)
      "ifc"         → from IFC model (IfcDoor, IfcWall, IfcFurnishing...)

    Edge types:
      "proximity"   → two objects within spatial threshold (Task A)
      "contains"    → IFC element spatially contains a visual object (Task B)
      "near_portal" → object is near an IFC door/window (Task B)

Why DBSCAN for clustering:
    The same object is detected across multiple frames (e.g., a chair
    appears in frames 0, 5, 10 from different angles). DBSCAN groups
    detections within eps=0.5m as the same physical object. We use
    DBSCAN over k-means because:
      - DBSCAN does not require knowing k in advance
      - DBSCAN handles noise (marks isolated detections as single objects)
      - DBSCAN is robust to elongated or irregular cluster shapes

Why NetworkX:
    NetworkX provides a pure-Python graph representation with built-in
    serialization (GraphML, JSON), traversal algorithms, and easy
    attribute storage per node/edge. For a scene graph of ~100-200 nodes
    the performance overhead is negligible.

"""

import os
import sys
import logging
import json
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Node ID helpers ───────────────────────────────────────────────────────────

def visual_node_id(cluster_id: int, class_name: str) -> str:
    """e.g. 'visual_0_chair' """
    return f"visual_{cluster_id}_{class_name.replace(' ', '_')}"


def ifc_node_id(ifc_id: str, ifc_class: str) -> str:
    """e.g. 'ifc_3L2hZu9KT9_IfcDoor' """
    return f"ifc_{ifc_id[:10]}_{ifc_class}"


# ── Clustering ────────────────────────────────────────────────────────────────

@dataclass
class ClusteredObject:
    """
    Represents one unique physical object after DBSCAN clustering.

    Multiple raw Detection3D instances that refer to the same
    physical object (seen across frames) are merged into one
    ClusteredObject with an averaged centroid.
    """

    cluster_id: int
    class_name: str
    centroid: np.ndarray        # averaged 3D world position
    confidence: float           # max confidence across all detections
    detection_count: int        # how many frames saw this object
    source_frame_ids: List[int] # which frames contributed


def cluster_detections(
    detections: List,
    eps_m: float = 0.5,
    min_samples: int = 1,
) -> List[ClusteredObject]:
    """
    Clusters raw 3D detections using DBSCAN to remove duplicates.

    The same object is detected across multiple frames. Without
    clustering, we'd add hundreds of nodes for the same chair.
    DBSCAN groups detections within eps_m metres as one object.

    Design decisions:
      eps_m=0.5   → detections within 0.5m = same object
      min_samples=1 → single-frame detections still become nodes
                      (don't discard objects seen only once)

    Args:
        detections  : list of Detection3D from object_detector.py
        eps_m       : maximum distance between same-object detections
        min_samples : minimum detections to form a cluster

    Returns:
        List of ClusteredObject — one per unique physical object
    """
    if not detections:
        logger.warning("No detections to cluster")
        return []

    # Group by class first — a chair and a table at the same position
    # are two different objects, not the same object
    class_groups = defaultdict(list)
    for det in detections:
        class_groups[det.class_name].append(det)

    clustered = []
    cluster_id = 0

    for class_name, class_dets in class_groups.items():
        if not class_dets:
            continue

        # Stack centroids into array for DBSCAN
        centroids = np.array([d.centroid_world for d in class_dets])

        if len(centroids) == 1:
            # Only one detection for this class — skip DBSCAN
            det = class_dets[0]
            clustered.append(ClusteredObject(
                cluster_id=cluster_id,
                class_name=class_name,
                centroid=det.centroid_world,
                confidence=det.confidence,
                detection_count=1,
                source_frame_ids=[det.frame_idx],
            ))
            cluster_id += 1
            continue

        # Run DBSCAN on 3D positions
        db = DBSCAN(eps=eps_m, min_samples=min_samples).fit(centroids)
        labels = db.labels_

        # Group detections by cluster label
        label_groups = defaultdict(list)
        for det, label in zip(class_dets, labels):
            label_groups[label].append(det)

        for label, group_dets in label_groups.items():
            # label=-1 means noise in DBSCAN, still create a node
            group_centroids = np.array([d.centroid_world for d in group_dets])

            clustered.append(ClusteredObject(
                cluster_id=cluster_id,
                class_name=class_name,
                centroid=group_centroids.mean(axis=0),
                confidence=max(d.confidence for d in group_dets),
                detection_count=len(group_dets),
                source_frame_ids=[d.frame_idx for d in group_dets],
            ))
            cluster_id += 1

    logger.info(
        f"Clustering: {len(detections)} raw detections → "
        f"{len(clustered)} unique objects"
    )
    return clustered


# ── Scene Graph Builder ───────────────────────────────────────────────────────

class SceneGraphBuilder:
    """
    Constructs the 3D scene graph from visual detections and IFC elements.

    The graph is a NetworkX MultiDiGraph (directed, allows multiple edges):
      - Directed: "contains" and "near_portal" are asymmetric
      - Multi: a node can have both "proximity" and "contains" edges

    Usage:
        builder = SceneGraphBuilder(config)
        G = builder.build(clustered_objects, ifc_scene)
        builder.save(G, "output/scene_graph.graphml")
    """

    def __init__(
        self,
        proximity_threshold_m: float = 2.0,
        containment_tolerance: float = 0.3,
        portal_proximity_m: float = 1.5,
    ):
        """
        Args:
            proximity_threshold_m  : max distance for "near" edge (Task A)
            containment_tolerance  : bbox expansion for room assignment (Task B)
            portal_proximity_m     : max distance from object to door for
                                     "near_portal" edge (Task B)
        """
        self.proximity_threshold_m = proximity_threshold_m
        self.containment_tolerance = containment_tolerance
        self.portal_proximity_m    = portal_proximity_m

    # ── Node Construction ─────────────────────────────────────────────────────

    def _add_visual_nodes(
        self,
        G: nx.Graph,
        clustered_objects: List[ClusteredObject],
    ) -> List[str]:
        """
        Adds one node per clustered visual detection.

        Node attributes stored for README documentation and viz:
            node_type, class_name, centroid, confidence, detection_count
        """
        node_ids = []
        for obj in clustered_objects:
            nid = visual_node_id(obj.cluster_id, obj.class_name)
            G.add_node(
                nid,
                node_type="visual",
                class_name=obj.class_name,
                centroid=obj.centroid.tolist(),
                x=float(obj.centroid[0]),
                y=float(obj.centroid[1]),
                z=float(obj.centroid[2]),
                confidence=float(obj.confidence),
                detection_count=int(obj.detection_count),
                source="yolov8m",
                label=obj.class_name,
            )
            node_ids.append(nid)
        logger.info(f"Added {len(node_ids)} visual nodes")
        return node_ids

    def _add_ifc_nodes(
        self,
        G: nx.Graph,
        ifc_scene,
    ) -> List[str]:
        """
        Adds one node per IFC element that has geometry.

        This is the Task B contribution — structural and furnishing
        elements from the BIM model become graph nodes. They provide
        positions for objects YOLO could not detect (appliances, fixtures).

        Only elements with a computed bounding box are added,
        since we need geometry for spatial queries.
        """
        node_ids = []
        for ifc_id, elem in ifc_scene.elements.items():
            if elem.bbox is None:
                continue

            nid = ifc_node_id(ifc_id, elem.ifc_class)
            G.add_node(
                nid,
                node_type="ifc",
                ifc_class=elem.ifc_class,
                ifc_id=ifc_id,
                class_name=elem.short_name,
                category=elem.category,
                centroid=elem.bbox.center.tolist(),
                x=float(elem.bbox.center[0]),
                y=float(elem.bbox.center[1]),
                z=float(elem.bbox.center[2]),
                bbox_min=elem.bbox.min_xyz.tolist(),
                bbox_max=elem.bbox.max_xyz.tolist(),
                source="ifc",
                label=f"{elem.ifc_class}: {elem.short_name}",
            )
            node_ids.append(nid)
        logger.info(f"Added {len(node_ids)} IFC nodes")
        return node_ids

    # ── Edge Construction ─────────────────────────────────────────────────────

    def _add_proximity_edges(
        self,
        G: nx.Graph,
        visual_node_ids: List[str],
    ) -> int:
        """
        Task A — Adds "proximity" edges between visual nodes within threshold.

        This implements the assignment requirement:
            "Edges should represent spatial heuristics (e.g., proximity,
             nearest-neighbor)"

        We use a simple distance threshold rather than k-NN because:
          - k-NN forces edges even between distant objects
          - Threshold-based proximity has intuitive physical meaning
            (objects within 2m are "near" each other)

        Returns:
            Number of edges added
        """
        edge_count = 0
        for i, nid_a in enumerate(visual_node_ids):
            for j, nid_b in enumerate(visual_node_ids):
                if i >= j:
                    continue

                pos_a = np.array(G.nodes[nid_a]["centroid"])
                pos_b = np.array(G.nodes[nid_b]["centroid"])
                dist  = float(np.linalg.norm(pos_a - pos_b))

                if dist <= self.proximity_threshold_m:
                    G.add_edge(
                        nid_a, nid_b,
                        edge_type="proximity",
                        relation="near",
                        distance=round(dist, 3),
                        weight=1.0 / (dist + 1e-6),
                    )
                    edge_count += 1

        logger.info(f"Added {edge_count} proximity edges (Task A)")
        return edge_count

    def _add_containment_edges(
        self,
        G: nx.Graph,
        visual_node_ids: List[str],
        ifc_node_ids: List[str],
        ifc_scene,
    ) -> int:
        """
        Task B — Adds "contains" edges from IFC elements to visual objects.

        This implements the assignment requirement:
            "Map the localized 3D object centroids into their corresponding
             IFC rooms, creating hierarchical bipartite edges (Room → Object)
             using simple geometric inclusion logic."

        For each visual object, we find which IFC element's bounding box
        contains it (2D containment in X-Y plane, ignoring Z because
        depth noise makes Z containment unreliable).

        The result is a bipartite subgraph:
            IfcFurnishingElement → chair
            IfcWall              → dining table
            IfcSlab              → couch

        Returns:
            Number of edges added
        """
        edge_count = 0

        for vis_nid in visual_node_ids:
            vis_pos = np.array(G.nodes[vis_nid]["centroid"])

            # Find containing IFC element
            containing = ifc_scene.find_containing_element(
                vis_pos,
                tolerance=self.containment_tolerance,
            )

            if containing is None:
                continue

            ifc_nid = ifc_node_id(containing.ifc_id, containing.ifc_class)

            if ifc_nid not in G.nodes:
                continue

            # Directed edge: IFC element → visual object
            G.add_edge(
                ifc_nid,
                vis_nid,
                edge_type="contains",
                relation="contains",
                ifc_class=containing.ifc_class,
            )
            edge_count += 1

        logger.info(f"Added {edge_count} containment edges (Task B)")
        return edge_count

    def _add_portal_edges(
        self,
        G: nx.Graph,
        visual_node_ids: List[str],
        ifc_scene,
    ) -> int:
        """
        Task B — Adds "near_portal" edges from objects near doors/windows.

        Doors are connective portals in the IFC model (IfcDoor).
        Objects near a door are likely in the transitional zone between
        two spaces — a meaningful spatial relationship.

        This partially implements the "connective portals (IfcDoor)"
        requirement from the assignment.

        Returns:
            Number of edges added
        """
        edge_count = 0
        portals = ifc_scene.doors + ifc_scene.windows

        for vis_nid in visual_node_ids:
            vis_pos = np.array(G.nodes[vis_nid]["centroid"])

            for portal in portals:
                if portal.bbox is None:
                    continue

                dist = float(np.linalg.norm(
                    vis_pos - portal.bbox.center
                ))

                if dist <= self.portal_proximity_m:
                    portal_nid = ifc_node_id(
                        portal.ifc_id, portal.ifc_class
                    )
                    if portal_nid not in G.nodes:
                        continue

                    G.add_edge(
                        vis_nid,
                        portal_nid,
                        edge_type="near_portal",
                        relation="near_portal",
                        distance=round(dist, 3),
                        portal_type=portal.ifc_class,
                    )
                    edge_count += 1

        logger.info(f"Added {edge_count} portal edges (Task B)")
        return edge_count

    # ── Main Build ────────────────────────────────────────────────────────────

    def build(
        self,
        clustered_objects: List[ClusteredObject],
        ifc_scene,
    ) -> nx.Graph:
        """
        Main entry point — builds the complete scene graph.

        Sequence:
          1. Create empty graph
          2. Add visual nodes (from YOLO detections)
          3. Add IFC nodes (from BIM model)
          4. Add proximity edges (Task A)
          5. Add containment edges (Task B)
          6. Add portal edges (Task B)

        Args:
            clustered_objects : deduplicated visual detections
            ifc_scene         : parsed IFC scene from ifc_parser.py

        Returns:
            NetworkX Graph — the complete 3D scene graph
        """
        G = nx.Graph()

        # Store metadata on the graph itself
        G.graph["description"] = "RGB-D Scene Graph with BIM Priors"
        G.graph["proximity_threshold_m"] = self.proximity_threshold_m
        G.graph["containment_tolerance"] = self.containment_tolerance

        # 1. Add nodes
        logger.info("Building scene graph...")
        vis_nids = self._add_visual_nodes(G, clustered_objects)
        ifc_nids = self._add_ifc_nodes(G, ifc_scene)

        # 2. Add edges
        self._add_proximity_edges(G, vis_nids)
        self._add_containment_edges(G, vis_nids, ifc_nids, ifc_scene)
        self._add_portal_edges(G, vis_nids, ifc_scene)

        # 3. Log summary
        logger.info("Scene graph built:")
        logger.info(f"  Total nodes : {G.number_of_nodes()}")
        logger.info(f"  Visual nodes: {len(vis_nids)}")
        logger.info(f"  IFC nodes   : {len(ifc_nids)}")
        logger.info(f"  Total edges : {G.number_of_edges()}")

        return G

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, G: nx.Graph, output_path: str) -> None:
        """
        Saves the scene graph to GraphML format.

        GraphML is XML-based and human-readable. It preserves all
        node/edge attributes and can be opened in Gephi, Cytoscape,
        or loaded back with NetworkX for further analysis.

        Args:
            G           : the scene graph
            output_path : path to save (e.g. output/scene_graph.graphml)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # GraphML requires all attributes to be primitive types
        # Convert lists (centroid) to strings for serialization
        G_save = G.copy()
        for nid, data in G_save.nodes(data=True):
            if "centroid" in data:
                data["centroid"] = str(data["centroid"])
            if "bbox_min" in data:
                data["bbox_min"] = str(data["bbox_min"])
            if "bbox_max" in data:
                data["bbox_max"] = str(data["bbox_max"])
            if "source_frame_ids" in data:
                data["source_frame_ids"] = str(data["source_frame_ids"])

        nx.write_graphml(G_save, output_path)
        logger.info(f"Scene graph saved: {output_path}")

    def save_summary(self, G: nx.Graph, output_path: str) -> None:
        """
        Saves a human-readable JSON summary of the scene graph.
        Useful for README documentation and debugging.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        node_types  = Counter(d.get("node_type") for _, d in G.nodes(data=True))
        edge_types  = Counter(d.get("edge_type") for _, _, d in G.edges(data=True))
        class_counts = Counter(
            d.get("class_name")
            for _, d in G.nodes(data=True)
            if d.get("node_type") == "visual"
        )

        summary = {
            "total_nodes"  : G.number_of_nodes(),
            "total_edges"  : G.number_of_edges(),
            "node_types"   : dict(node_types),
            "edge_types"   : dict(edge_types),
            "visual_classes": dict(class_counts),
            "nodes": [
                {
                    "id"        : nid,
                    "node_type" : d.get("node_type"),
                    "class_name": d.get("class_name"),
                    "position"  : [
                        round(d.get("x", 0), 3),
                        round(d.get("y", 0), 3),
                        round(d.get("z", 0), 3),
                    ],
                    "source"    : d.get("source"),
                }
                for nid, d in G.nodes(data=True)
            ],
            "edges": [
                {
                    "from"      : u if G.nodes[u].get("node_type") == "ifc"
                                or d.get("edge_type") == "proximity"
                                else v,
                    "to"        : v if G.nodes[u].get("node_type") == "ifc"
                                or d.get("edge_type") == "proximity"
                                else u,
                    "edge_type" : d.get("edge_type"),
                    "relation"  : d.get("relation"),
                    "distance"  : d.get("distance"),
                }
                for u, v, d in G.edges(data=True)
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Scene graph summary saved: {output_path}")


# ── Graph Analysis Helpers ────────────────────────────────────────────────────

def print_graph_summary(G: nx.Graph) -> None:
    """Prints a detailed human-readable summary of the scene graph."""

    print("\n" + "=" * 60)
    print("SCENE GRAPH SUMMARY")
    print("=" * 60)

    # Node breakdown
    node_types = Counter(d.get("node_type") for _, d in G.nodes(data=True))
    print(f"\nNodes: {G.number_of_nodes()} total")
    for ntype, count in node_types.items():
        print(f"  {ntype:<12} : {count}")

    # Visual object classes
    visual_classes = Counter(
        d.get("class_name")
        for _, d in G.nodes(data=True)
        if d.get("node_type") == "visual"
    )
    if visual_classes:
        print("\nVisual objects detected:")
        for cls, count in visual_classes.most_common():
            print(f"  {cls:<20} {count}")

    # IFC class breakdown
    ifc_classes = Counter(
        d.get("ifc_class")
        for _, d in G.nodes(data=True)
        if d.get("node_type") == "ifc"
    )
    if ifc_classes:
        print("\nIFC elements in graph:")
        for cls, count in ifc_classes.most_common(10):
            print(f"  {cls:<35} {count}")

    # Edge breakdown
    edge_types = Counter(d.get("edge_type") for _, _, d in G.edges(data=True))
    print(f"\nEdges: {G.number_of_edges()} total")
    for etype, count in edge_types.items():
        print(f"  {etype:<20} : {count}")

    # Containment summary — which IFC elements contain visual objects
    print("\nContainment (IFC → Visual):")
    containment_found = False
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "contains":
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            # Edge direction: IFC element → visual object (contains)
            ifc_label = u_data.get("ifc_class", "?") if u_data.get("node_type") == "ifc" else v_data.get("ifc_class", "?")
            vis_label = v_data.get("class_name", "?") if u_data.get("node_type") == "ifc" else u_data.get("class_name", "?")
            print(f"  [{ifc_label}] → {vis_label}")
            containment_found = True
    if not containment_found:
        print("  (none — visual objects outside IFC element bounds)")

    # Portal proximity summary
    print("\nPortal Proximity (Visual → Door/Window):")
    portal_found = False
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "near_portal":
            u_cls = G.nodes[u].get("class_name", "?")
            v_cls = G.nodes[v].get("ifc_class", "?")
            dist  = d.get("distance", 0)
            print(f"  {u_cls} → [{v_cls}] at {dist:.2f}m")
            portal_found = True
    if not portal_found:
        print("  (none within portal proximity threshold)")

    print("=" * 60)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("scene_graph.py — Self Test")
    print("=" * 60)

    # Add project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.ifc_parser import parse_ifc_scene
    from src.depth_projection import CameraIntrinsics, PoseLoader, FrameIterator
    from src.object_detector import YOLOv8Detector, detect_and_project

    # Paths
    labels_path = "data/BasicHouse_with_pc/_ifcgeom_scene.labels.json"
    obj_path    = "data/BasicHouse_with_pc/_ifcgeom_scene.obj"
    camera_json = "data/BasicHouse_with_pc/camera_info.json"
    pose_file   = "data/BasicHouse_with_pc/pose/poses.txt"
    rgb_dir     = "data/BasicHouse_with_pc/rgb"
    depth_dir   = "data/BasicHouse_with_pc/depth_png16"

    # ── Step 1: Parse IFC ─────────────────────────────────────────────────────
    print("\n[1] Parsing IFC scene...")
    ifc_scene = parse_ifc_scene(labels_path, obj_path)
    print(f"    IFC elements: {len(ifc_scene.elements)}")
    print(f"    With geometry: "
          f"{sum(1 for e in ifc_scene.elements.values() if e.bbox is not None)}")

    # ── Step 2: Run object detection ──────────────────────────────────────────
    print("\n[2] Running YOLOv8 detection across frames...")
    intrinsics = CameraIntrinsics.from_json(camera_json)
    poses      = PoseLoader(pose_file)
    detector   = YOLOv8Detector(
        model_name="yolov8m.pt",
        confidence_threshold=0.08,
    )

    iterator = FrameIterator(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        pose_loader=poses,
        frame_step=5,
    )

    all_detections = []
    for rgb, depth, pose, idx in iterator:
        dets = detect_and_project(
            detector=detector,
            rgb_frame=rgb,
            depth_frame=depth,
            pose_matrix=pose,
            intrinsics=intrinsics,
            frame_idx=idx,
            max_depth_m=10.0,
        )
        all_detections.extend(dets)

    print(f"    Raw detections: {len(all_detections)}")

    # ── Step 3: Cluster detections ────────────────────────────────────────────
    print("\n[3] Clustering detections...")
    clustered = cluster_detections(
        all_detections,
        eps_m=0.5,
        min_samples=1,
    )
    print(f"    Unique objects after clustering: {len(clustered)}")
    for obj in clustered:
        print(
            f"    [{obj.class_name:<15}] "
            f"centroid={np.round(obj.centroid, 2)} "
            f"seen_in={obj.detection_count} frames"
        )

    # ── Step 4: Build scene graph ─────────────────────────────────────────────
    print("\n[4] Building scene graph...")
    builder = SceneGraphBuilder(
        proximity_threshold_m=2.0,
        containment_tolerance=0.5,
        portal_proximity_m=3.0,
    )
    G = builder.build(clustered, ifc_scene)

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    print_graph_summary(G)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    print("\n[6] Saving outputs...")
    os.makedirs("output", exist_ok=True)
    builder.save(G, "output/scene_graph.graphml")
    builder.save_summary(G, "output/scene_graph_summary.json")
    print("    output/scene_graph.graphml")
    print("    output/scene_graph_summary.json")

    print("\n" + "=" * 60)
    print("✅ scene_graph.py complete!")
    print("=" * 60)