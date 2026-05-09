"""
visualize.py
────────────
Produces three visualizations of the 3D scene graph for documentation
and qualitative evaluation:

  1. Top-down floor plan (PNG)
       Bird's-eye matplotlib view showing camera trajectory,
       visual detections, IFC elements, and edges.
       Embeddable in README.md.

  2. Interactive 3D scene (HTML)
       Plotly-based rotatable 3D scatter showing all nodes in
       world coordinates with edges and hover tooltips.

  3. Graph topology (PNG)
       NetworkX spring-layout drawing showing pure connectivity
       patterns colored by node type and edge type.
       Embeddable in README.md.

Task Context:
    The task requires demonstrating that the scene graph correctly
    represents spatial relationships. Visualization is the most direct
    way to prove this and provides evidence.

Design decisions:
    - matplotlib for static PNGs (easy embedding in markdown)
    - plotly for interactive 3D (richer exploration, optional)
    - No open3d dependency — Python 3.14 compatible

"""

import os
import sys
import logging
from typing import Optional, Tuple, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Color Schemes ─────────────────────────────────────────────────────────────
# Using a consistent palette across all visualizations so figures
# can be compared in the README.

NODE_COLORS = {
    "visual": "#E63946",     # red — YOLO detections
    "ifc":    "#457B9D",     # blue — IFC structural elements
}

EDGE_COLORS = {
    "proximity":   "#2A9D8F",  # green — near (Task A)
    "contains":    "#F4A261",  # orange — IFC contains object (Task B)
    "near_portal": "#9B5DE5",  # purple — near door/window (Task B)
}

IFC_CATEGORY_MARKERS = {
    "structural":  "s",   # square — walls, slabs, columns
    "portal":      "^",   # triangle — doors, windows
    "furnishing":  "D",   # diamond — appliances, furniture
    "other":       "o",   # circle — fallback
}


# ── Top-Down Floor Plan ───────────────────────────────────────────────────────

def plot_top_down(
    G: nx.Graph,
    output_path: str,
    pose_loader=None,
    title: str = "Scene Graph — Top-Down View (X-Y plane)",
    figsize: Tuple[int, int] = (14, 11),
) -> None:
    """
    Generates a bird's-eye view of the scene graph in the X-Y plane.

    This is the most informative single-image representation because
    indoor scenes are essentially 2D layouts (Z is just floor height).

    Includes:
      - Camera trajectory (gray line) if pose_loader provided
      - Visual nodes (red circles) labeled with class name
      - IFC nodes (blue markers, shape per category)
      - Edges in three colors per edge type

    Args:
        G          : the scene graph
        output_path: where to save the PNG
        pose_loader: optional PoseLoader to draw the camera trajectory
        title      : plot title
        figsize    : matplotlib figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # ── Draw camera trajectory ────────────────────────────────────────────────
    if pose_loader is not None:
        traj_xy = np.array([
            [pose_loader.get(i)[0, 3], pose_loader.get(i)[1, 3]]
            for i in range(len(pose_loader))
        ])
        ax.plot(
            traj_xy[:, 0], traj_xy[:, 1],
            color="lightgray", linewidth=1.2, zorder=1,
            label="Camera trajectory",
        )
        # Mark start with a green star
        ax.scatter(
            traj_xy[0, 0], traj_xy[0, 1],
            color="green", marker="*", s=200, zorder=4,
            edgecolors="black", linewidth=1,
        )

    # ── Draw edges (under nodes for readability) ──────────────────────────────
    edge_count_by_type = {"proximity": 0, "contains": 0, "near_portal": 0}

    for u, v, d in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        x = [u_data.get("x", 0), v_data.get("x", 0)]
        y = [u_data.get("y", 0), v_data.get("y", 0)]

        etype = d.get("edge_type", "proximity")
        color = EDGE_COLORS.get(etype, "gray")
        edge_count_by_type[etype] = edge_count_by_type.get(etype, 0) + 1

        # Make different edge types visually distinct
        if etype == "proximity":
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.55, zorder=2)
        elif etype == "contains":
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.4,
                    linestyle=":", zorder=2)
        elif etype == "near_portal":
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.7,
                    linestyle="--", zorder=3)

    # ── Draw IFC nodes ────────────────────────────────────────────────────────
    ifc_drawn = {"structural": 0, "portal": 0, "furnishing": 0, "other": 0}

    for nid, data in G.nodes(data=True):
        if data.get("node_type") != "ifc":
            continue
        category = data.get("category", "other")
        marker   = IFC_CATEGORY_MARKERS.get(category, "o")
        ax.scatter(
            data.get("x", 0), data.get("y", 0),
            color=NODE_COLORS["ifc"],
            marker=marker, s=120, zorder=5,
            edgecolors="black", linewidth=0.8,
        )
        # Small label on hover-style positioning
        ax.text(
            data.get("x", 0) + 0.15,
            data.get("y", 0) + 0.15,
            data.get("ifc_class", "")[3:],  # strip "Ifc" prefix
            fontsize=7, color="#1D3557", zorder=6,
        )
        ifc_drawn[category] = ifc_drawn.get(category, 0) + 1

    # ── Draw visual nodes ─────────────────────────────────────────────────────
    for nid, data in G.nodes(data=True):
        if data.get("node_type") != "visual":
            continue
        ax.scatter(
            data.get("x", 0), data.get("y", 0),
            color=NODE_COLORS["visual"],
            marker="o", s=140, zorder=6,
            edgecolors="black", linewidth=1.2,
        )
        ax.text(
            data.get("x", 0) + 0.15,
            data.get("y", 0) - 0.25,
            data.get("class_name", ""),
            fontsize=8, color="#7A0E1F", zorder=7,
            fontweight="bold",
        )

    # ── Build legend ──────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Visual (YOLO)",
               markerfacecolor=NODE_COLORS["visual"], markersize=10,
               markeredgecolor="black"),
        Line2D([0], [0], marker="s", color="w", label="IFC Structural",
               markerfacecolor=NODE_COLORS["ifc"], markersize=10,
               markeredgecolor="black"),
        Line2D([0], [0], marker="^", color="w", label="IFC Portal (Door/Window)",
               markerfacecolor=NODE_COLORS["ifc"], markersize=10,
               markeredgecolor="black"),
        Line2D([0], [0], marker="D", color="w", label="IFC Furnishing",
               markerfacecolor=NODE_COLORS["ifc"], markersize=10,
               markeredgecolor="black"),
        Line2D([0], [0], color=EDGE_COLORS["proximity"], lw=1.5,
               label=f'Proximity edge ({edge_count_by_type["proximity"]})'),
        Line2D([0], [0], color=EDGE_COLORS["contains"], lw=1.5, linestyle=":",
               label=f'Contains edge ({edge_count_by_type["contains"]})'),
        Line2D([0], [0], color=EDGE_COLORS["near_portal"], lw=1.5,
               linestyle="--",
               label=f'Near-portal edge ({edge_count_by_type["near_portal"]})'),
    ]

    if pose_loader is not None:
        legend_elements.append(
            Line2D([0], [0], color="lightgray", lw=2,
                   label="Camera trajectory")
        )

    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, framealpha=0.9)

    ax.set_xlabel("X (metres)", fontsize=11)
    ax.set_ylabel("Y (metres)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Top-down view saved: {output_path}")


# ── Graph Topology View ───────────────────────────────────────────────────────

def plot_graph_topology(
    G: nx.Graph,
    output_path: str,
    title: str = "Scene Graph Topology",
    figsize: Tuple[int, int] = (14, 11),
) -> None:
    """
    Plots the abstract graph structure using NetworkX spring layout.

    Unlike the top-down view (which uses real X-Y coordinates), this
    visualization uses a force-directed layout where connected nodes
    cluster together. It reveals the connectivity pattern of the graph
    independent of physical positions — useful for showing that the
    proximity + containment + portal edges form meaningful clusters.

    Args:
        G          : the scene graph
        output_path: where to save the PNG
        title      : plot title
        figsize    : matplotlib figure size
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Spring layout — connected nodes pulled together
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=80)

    # ── Draw edges by type ────────────────────────────────────────────────────
    for etype, color in EDGE_COLORS.items():
        edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == etype
        ]
        if not edge_list:
            continue
        style = (
            "solid" if etype == "proximity"
            else "dotted" if etype == "contains"
            else "dashed"
        )
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_list,
            edge_color=color,
            width=1.3,
            alpha=0.65,
            style=style,
            ax=ax,
        )

    # ── Draw nodes by type ────────────────────────────────────────────────────
    visual_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "visual"
    ]
    ifc_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "ifc"
    ]

    nx.draw_networkx_nodes(
        G, pos, nodelist=visual_nodes,
        node_color=NODE_COLORS["visual"],
        node_size=400,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=ifc_nodes,
        node_color=NODE_COLORS["ifc"],
        node_size=300,
        node_shape="s",
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = {}
    for nid, data in G.nodes(data=True):
        if data.get("node_type") == "visual":
            labels[nid] = data.get("class_name", nid)[:8]
        else:
            ifc_cls = data.get("ifc_class", nid)
            labels[nid] = ifc_cls[3:][:10]  # strip "Ifc" prefix, max 10 chars

    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Visual node",
               markerfacecolor=NODE_COLORS["visual"], markersize=12,
               markeredgecolor="black"),
        Line2D([0], [0], marker="s", color="w", label="IFC node",
               markerfacecolor=NODE_COLORS["ifc"], markersize=12,
               markeredgecolor="black"),
        Line2D([0], [0], color=EDGE_COLORS["proximity"], lw=2,
               label="Proximity edge"),
        Line2D([0], [0], color=EDGE_COLORS["contains"], lw=2,
               linestyle=":", label="Contains edge"),
        Line2D([0], [0], color=EDGE_COLORS["near_portal"], lw=2,
               linestyle="--", label="Near-portal edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Graph topology saved: {output_path}")


# ── Interactive 3D View (Plotly) ──────────────────────────────────────────────

def plot_3d_interactive(
    G: nx.Graph,
    output_path: str,
    pose_loader=None,
    title: str = "3D Scene Graph",
) -> None:
    """
    Generates an interactive 3D HTML visualization using Plotly.

    Users can rotate, zoom, and hover over nodes to see attributes.
    Useful for exploring the graph in detail beyond the static PNGs.

    Args:
        G          : the scene graph
        output_path: where to save the HTML file
        pose_loader: optional, draws camera trajectory in 3D
        title      : plot title
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed — skipping interactive 3D view")
        return

    fig = go.Figure()

    # ── Camera trajectory ─────────────────────────────────────────────────────
    if pose_loader is not None:
        traj = np.array([
            [pose_loader.get(i)[0, 3],
             pose_loader.get(i)[1, 3],
             pose_loader.get(i)[2, 3]]
            for i in range(len(pose_loader))
        ])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="lines",
            line=dict(color="lightgray", width=3),
            name="Camera trajectory",
        ))

    # ── Edges (one trace per edge type for legend control) ────────────────────
    for etype, color in EDGE_COLORS.items():
        edge_x, edge_y, edge_z = [], [], []
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") != etype:
                continue
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            edge_x.extend([u_data.get("x", 0), v_data.get("x", 0), None])
            edge_y.extend([u_data.get("y", 0), v_data.get("y", 0), None])
            edge_z.extend([u_data.get("z", 0), v_data.get("z", 0), None])

        if not edge_x:
            continue

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{etype} edges",
            hoverinfo="skip",
        ))

    # ── Visual nodes ──────────────────────────────────────────────────────────
    vis_x, vis_y, vis_z, vis_text = [], [], [], []
    for nid, data in G.nodes(data=True):
        if data.get("node_type") != "visual":
            continue
        vis_x.append(data.get("x", 0))
        vis_y.append(data.get("y", 0))
        vis_z.append(data.get("z", 0))
        vis_text.append(
            f"<b>{data.get('class_name')}</b><br>"
            f"confidence: {data.get('confidence', 0):.2f}<br>"
            f"detected in {data.get('detection_count', 0)} frames<br>"
            f"position: ({data.get('x', 0):.2f}, "
            f"{data.get('y', 0):.2f}, {data.get('z', 0):.2f})"
        )

    fig.add_trace(go.Scatter3d(
        x=vis_x, y=vis_y, z=vis_z,
        mode="markers",
        marker=dict(
            size=8,
            color=NODE_COLORS["visual"],
            line=dict(color="black", width=1),
        ),
        name="Visual nodes (YOLO)",
        text=vis_text,
        hoverinfo="text",
    ))

    # ── IFC nodes ─────────────────────────────────────────────────────────────
    ifc_x, ifc_y, ifc_z, ifc_text = [], [], [], []
    for nid, data in G.nodes(data=True):
        if data.get("node_type") != "ifc":
            continue
        ifc_x.append(data.get("x", 0))
        ifc_y.append(data.get("y", 0))
        ifc_z.append(data.get("z", 0))
        ifc_text.append(
            f"<b>{data.get('ifc_class')}</b><br>"
            f"name: {data.get('class_name', '')}<br>"
            f"category: {data.get('category', '')}<br>"
            f"position: ({data.get('x', 0):.2f}, "
            f"{data.get('y', 0):.2f}, {data.get('z', 0):.2f})"
        )

    fig.add_trace(go.Scatter3d(
        x=ifc_x, y=ifc_y, z=ifc_z,
        mode="markers",
        marker=dict(
            size=7,
            color=NODE_COLORS["ifc"],
            symbol="square",
            line=dict(color="black", width=1),
        ),
        name="IFC nodes (BIM)",
        text=ifc_text,
        hoverinfo="text",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Interactive 3D view saved: {output_path}")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def visualize_scene_graph(
    G: nx.Graph,
    output_dir: str = "output/viz",
    pose_loader=None,
    scene_name: str = "scene",
) -> None:
    """
    Runs all three visualizations and saves them to output_dir.

    Args:
        G          : the scene graph from scene_graph.py
        output_dir : where to save visualization files
        pose_loader: optional PoseLoader for camera trajectory
        scene_name : prefix for output files (e.g. "basichouse")
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating visualizations for scene: {scene_name}")

    plot_top_down(
        G,
        output_path=os.path.join(output_dir, f"{scene_name}_topdown.png"),
        pose_loader=pose_loader,
        title=f"{scene_name.title()} — Scene Graph (Top-Down View)",
    )

    plot_graph_topology(
        G,
        output_path=os.path.join(output_dir, f"{scene_name}_topology.png"),
        title=f"{scene_name.title()} — Graph Topology",
    )

    plot_3d_interactive(
        G,
        output_path=os.path.join(output_dir, f"{scene_name}_3d.html"),
        pose_loader=pose_loader,
        title=f"{scene_name.title()} — 3D Scene Graph",
    )

    logger.info(f"All visualizations saved to {output_dir}/")


# ── Self Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("visualize.py — Self Test")
    print("=" * 60)

    # Allow running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.ifc_parser import parse_ifc_scene
    from src.depth_projection import CameraIntrinsics, PoseLoader, FrameIterator
    from src.object_detector import YOLOv8Detector, detect_and_project
    from src.scene_graph import (
        SceneGraphBuilder,
        cluster_detections,
    )

    # Paths
    labels_path = "data/BasicHouse_with_pc/_ifcgeom_scene.labels.json"
    obj_path    = "data/BasicHouse_with_pc/_ifcgeom_scene.obj"
    camera_json = "data/BasicHouse_with_pc/camera_info.json"
    pose_file   = "data/BasicHouse_with_pc/pose/poses.txt"
    rgb_dir     = "data/BasicHouse_with_pc/rgb"
    depth_dir   = "data/BasicHouse_with_pc/depth_png16"

    # ── Step 1: Build scene graph end-to-end ──────────────────────────────────
    print("\n[1] Parsing IFC scene...")
    ifc_scene = parse_ifc_scene(labels_path, obj_path)

    print("\n[2] Loading camera + poses...")
    intrinsics = CameraIntrinsics.from_json(camera_json)
    poses      = PoseLoader(pose_file)

    print("\n[3] Running YOLOv8 detection...")
    detector = YOLOv8Detector(
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
    print(f"    Got {len(all_detections)} raw detections")

    print("\n[4] Clustering + building graph...")
    clustered = cluster_detections(all_detections, eps_m=0.5, min_samples=1)
    builder = SceneGraphBuilder(
        proximity_threshold_m=2.0,
        containment_tolerance=0.5,
        portal_proximity_m=3.0,
    )
    G = builder.build(clustered, ifc_scene)
    print(f"    Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # ── Step 2: Generate visualizations ───────────────────────────────────────
    print("\n[5] Generating visualizations...")
    visualize_scene_graph(
        G,
        output_dir="output/viz",
        pose_loader=poses,
        scene_name="basichouse",
    )

    print("\n" + "=" * 60)
    print("✅ visualize.py complete!")
    print("=" * 60)
