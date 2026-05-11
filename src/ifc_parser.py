"""
Parses pre-converted IFC geometry data from:
  - _ifcgeom_scene.labels.json  : maps mesh-object IDs → IFC class + name
  - _ifcgeom_scene.obj          : full 3D mesh geometry of the building

Context (Task B):
    The task requires extracting spatial boundaries (IfcSpace) and connective
    portals (IfcDoor) from IFC data to enrich the visual scene graph.

    Since the provided datasets contain no raw .ifc file (the IFC has been
    pre-converted to OBJ + labels JSON via Blender/IfcOpenShell upstream),
    this module replicates the IfcOpenShell workflow by:
      1. Loading semantic labels from JSON  (equivalent to ifc.by_type())
      2. Loading geometry from OBJ          (equivalent to ifcopenshell.geom)
      3. Computing per-element bounding boxes for spatial inclusion queries

"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import trimesh

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── IFC classes we care about ─────────────────────────────────────────────────
# Structural elements that form the building skeleton
STRUCTURAL_CLASSES = {
    "IfcWall", "IfcWallStandardCase", "IfcSlab",
    "IfcColumn", "IfcBeam", "IfcRoof", "IfcCovering",
    "IfcBuildingElementProxy"
}

# Portal elements — openings that connect spaces
PORTAL_CLASSES = {
    "IfcDoor", "IfcWindow", "IfcOpeningElement"
}

# Furnishing elements — movable objects (what YOLOv8 also detects visually)
FURNISHING_CLASSES = {
    "IfcFurnishingElement", "IfcFurnitureType",
    "IfcFlowTerminal", "IfcDistributionPort"
}

# All classes we want to extract
ALL_RELEVANT_CLASSES = STRUCTURAL_CLASSES | PORTAL_CLASSES | FURNISHING_CLASSES


# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class BoundingBox3D:
    """
    Axis-aligned bounding box in 3D world space.
    Used to check whether a detected object centroid falls
    'inside' a structural IFC element (spatial inclusion logic).
    """
    min_xyz: np.ndarray   # shape (3,) — minimum corner
    max_xyz: np.ndarray   # shape (3,) — maximum corner

    @property
    def center(self) -> np.ndarray:
        return (self.min_xyz + self.max_xyz) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz

    def contains_point(self, point: np.ndarray,
                        tolerance: float = 0.1) -> bool:
        """
        Returns True if the 3D point lies within this bounding box.
        Tolerance expands the box slightly to handle boundary cases
        and depth noise in detected centroids.
        """
        lo = self.min_xyz - tolerance
        hi = self.max_xyz + tolerance
        return bool(np.all(point >= lo) and np.all(point <= hi))

    def contains_point_2d(self, point: np.ndarray,
                           tolerance: float = 0.2) -> bool:
        """
        2D containment check using only X and Y (ignores Z height).
        More robust for room task when depth estimates are noisy.
        This is the primary method used for Room → Object assignment.
        """
        lo = self.min_xyz[:2] - tolerance
        hi = self.max_xyz[:2] + tolerance
        p  = point[:2]
        return bool(np.all(p >= lo) and np.all(p <= hi))


@dataclass
class IFCElement:
    """
    Represents a single IFC building element extracted from the dataset.

    Equivalent to what IfcOpenShell returns when calling:
        model.by_type("IfcDoor")  →  list of IfcDoor entities
    """
    ifc_id:    str             # unique mesh object ID from labels JSON
    ifc_class: str             # e.g. "IfcDoor", "IfcWall", "IfcFurnishingElement"
    name:      str             # human-readable name from IFC model
    bbox:      Optional[BoundingBox3D] = None   # 3D bounding box (set after OBJ load)
    vertices:  Optional[np.ndarray]   = None    # raw vertices (N, 3) if needed

    @property
    def category(self) -> str:
        """Returns high-level category: structural / portal / furnishing / other"""
        if self.ifc_class in STRUCTURAL_CLASSES:
            return "structural"
        elif self.ifc_class in PORTAL_CLASSES:
            return "portal"
        elif self.ifc_class in FURNISHING_CLASSES:
            return "furnishing"
        return "other"

    @property
    def short_name(self) -> str:
        """Returns the clean object name without IFC ID suffix."""
        # Names are formatted as "TypeName:VariantName:ID"
        # e.g. "Innerdörr - standard:D9:1298042" → "Innerdörr - standard"
        parts = self.name.split(":")
        return parts[0].strip() if parts else self.name


@dataclass
class IFCScene:
    """
    Container for all parsed IFC elements from one scene.
    Provides lookup methods used by scene_graph.py for BIM fusion.
    """
    elements:   Dict[str, IFCElement] = field(default_factory=dict)
    scene_path: str = ""

    # ── Filtered accessors ────────────────────────────────────────────────────
    @property
    def doors(self) -> List[IFCElement]:
        return [e for e in self.elements.values()
                if e.ifc_class in PORTAL_CLASSES and "Door" in e.ifc_class]

    @property
    def windows(self) -> List[IFCElement]:
        return [e for e in self.elements.values()
                if "Window" in e.ifc_class]

    @property
    def walls(self) -> List[IFCElement]:
        return [e for e in self.elements.values()
                if "Wall" in e.ifc_class]

    @property
    def furnishings(self) -> List[IFCElement]:
        return [e for e in self.elements.values()
                if e.ifc_class in FURNISHING_CLASSES]

    @property
    def structural_elements(self) -> List[IFCElement]:
        return [e for e in self.elements.values()
                if e.category == "structural"]

    # ── Spatial Query ─────────────────────────────────────────────────────────
    def find_nearest_structural_element(
            self,
            point: np.ndarray,
            max_distance: float = 3.0
    ) -> Optional[Tuple[IFCElement, float]]:
        """
        Finds the nearest structural IFC element to a given 3D point.

        This is the core BIM fusion function used in Task B.
        For each visually detected object (3D centroid from depth projection),
        we find which structural element it is closest to — effectively
        assigning it to a spatial region of the building.

        Args:
            point:        3D world coordinate (X, Y, Z) of detected object
            max_distance: ignore elements farther than this (metres)

        Returns:
            (IFCElement, distance) tuple, or None if nothing found nearby
        """
        best_elem  = None
        best_dist  = float("inf")

        for elem in self.elements.values():
            if elem.bbox is None:
                continue
            # Distance from point to bbox center
            dist = float(np.linalg.norm(point - elem.bbox.center))
            if dist < best_dist and dist < max_distance:
                best_dist = dist
                best_elem = elem

        if best_elem is None:
            return None
        return (best_elem, best_dist)

    def find_containing_element(
            self,
            point: np.ndarray,
            classes: Optional[List[str]] = None,
            tolerance: float = 0.2
    ) -> Optional[IFCElement]:
        """
        Returns the IFC element whose bounding box CONTAINS the given point.

        This implements the geometric inclusion logic described in the task:
            "map the localized 3D object centroids into their corresponding
             IFC rooms, creating hierarchical bipartite edges (Room → Object)"

        We use 2D containment (X, Y only) because:
        - Depth estimation from PNG16 has Z noise
        - Room membership is inherently a floor-plan concept
        - Z containment would incorrectly exclude objects on shelves/tables

        Args:
            point:   3D world coordinate of detected object centroid
            classes: filter to specific IFC classes (e.g. ["IfcWall"])
            tolerance: expand bbox by this amount (metres) for robustness
        """
        candidates = list(self.elements.values())
        if classes:
            candidates = [e for e in candidates if e.ifc_class in classes]

        for elem in candidates:
            if elem.bbox is None:
                continue
            if elem.bbox.contains_point_2d(point, tolerance=tolerance):
                return elem
        return None

    def summary(self) -> Dict[str, int]:
        """Returns a count of each IFC class — useful for README documentation."""
        counts = defaultdict(int)
        for elem in self.elements.values():
            counts[elem.ifc_class] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ── Core Parsing Functions ────────────────────────────────────────────────────

def load_ifc_labels(labels_path: str) -> Dict[str, IFCElement]:
    """
    Loads and parses the _ifcgeom_scene.labels.json file.

    The JSON maps mesh object IDs → {"ifc_class": ..., "name": ...}
    We filter to only the classes relevant for our scene graph.

    Args:
        labels_path: path to _ifcgeom_scene.labels.json

    Returns:
        Dictionary of {ifc_id: IFCElement} for all relevant elements
    """
    logger.info(f"Loading IFC labels from: {labels_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    elements = {}
    skipped  = 0

    for ifc_id, data in raw.items():
        ifc_class = data.get("ifc_class", "unknown")
        name      = data.get("name", "unnamed")

        # Skip non-spatial metadata classes
        skip_classes = {
            "IfcPropertySet", "IfcRelDefinesByProperties",
            "IfcRelAssociatesMaterial", "IfcElementQuantity",
            "IfcRelDefinesByType", "IfcRelVoidsElement",
            "IfcRelFillsElement", "IfcRelAggregates",
            "IfcRelConnectsPathElements", "IfcRelConnectsPortToElement",
            "IfcWindowStyle", "IfcWindowLiningProperties",
            "IfcDoorStyle", "IfcDistributionElementType",
            "IfcFurnitureType"
        }
        if ifc_class in skip_classes:
            skipped += 1
            continue

        elements[ifc_id] = IFCElement(
            ifc_id=ifc_id,
            ifc_class=ifc_class,
            name=name
        )

    logger.info(f"  Loaded {len(elements)} elements "
                f"(skipped {skipped} metadata-only entries)")
    return elements


def load_ifc_geometry(
    obj_path: str,
    elements: Dict[str, IFCElement],
) -> Dict[str, IFCElement]:
    """
    Loads the _ifcgeom_scene.obj mesh and computes per-element bounding boxes
    by manually parsing OBJ group/object tags.

    Trimesh merges all groups into one mesh when loading IFC-exported OBJ files,
    so we parse the file line-by-line to extract per-group vertex data.

    OBJ format reference:
        o / g <name>  →  start of a new named group
        v x y z       →  vertex coordinate
    """
    logger.info(f"Parsing OBJ geometry manually: {obj_path}")

    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    # ── Pass 1: collect all vertices and group membership ─────────────────────
    all_vertices   = []   # global vertex list (1-indexed in OBJ)
    group_vert_ids = defaultdict(list)  # group_name → [vertex indices]
    current_group  = None

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # New object or group
            if line.startswith("o ") or line.startswith("g "):
                current_group = line.split(None, 1)[1].strip()

            # Vertex definition
            elif line.startswith("v ") and not line.startswith("vt") \
                    and not line.startswith("vn"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        all_vertices.append([x, y, z])
                        if current_group:
                            # Store 0-based index of this vertex
                            group_vert_ids[current_group].append(
                                len(all_vertices) - 1
                            )
                    except ValueError:
                        continue

    all_vertices = np.array(all_vertices) if all_vertices else np.zeros((0, 3))
    logger.info(
        f"  Parsed {len(all_vertices)} vertices "
        f"across {len(group_vert_ids)} groups"
    )

    if len(group_vert_ids) == 0:
        logger.warning("  No named groups found in OBJ — bbox extraction skipped")
        return elements

    # ── Pass 2: match groups to IFC elements and compute bboxes ───────────────
    matched   = 0
    unmatched = 0

    for group_name, vert_indices in group_vert_ids.items():
        if not vert_indices:
            continue

        # Try direct match, then substring match
        matched_id = None

        if group_name in elements:
            matched_id = group_name
        else:
            for ifc_id in elements:
                if ifc_id in group_name or group_name in ifc_id:
                    matched_id = ifc_id
                    break

        if matched_id is None:
            unmatched += 1
            continue

        # Extract vertices for this group and compute bbox
        verts = all_vertices[vert_indices]
        if len(verts) == 0:
            continue

        elements[matched_id].bbox = BoundingBox3D(
            min_xyz=verts.min(axis=0),
            max_xyz=verts.max(axis=0),
        )
        elements[matched_id].vertices = verts
        matched += 1

    logger.info(f"  Geometry matched: {matched} | Unmatched: {unmatched}")
    return elements


def parse_ifc_scene(labels_path: str, obj_path: str) -> IFCScene:
    """
    Main entry point — parses both the labels JSON and OBJ geometry
    and returns a fully populated IFCScene ready for BIM fusion.

    Usage in run_pipeline.py:
        from src.ifc_parser import parse_ifc_scene
        ifc_scene = parse_ifc_scene(labels_path, obj_path)

    Args:
        labels_path: path to _ifcgeom_scene.labels.json
        obj_path:    path to _ifcgeom_scene.obj

    Returns:
        IFCScene with all elements, bounding boxes, and spatial query methods
    """
    # Step 1: Load semantic labels
    elements = load_ifc_labels(labels_path)

    # Step 2: Load geometry and compute bounding boxes
    elements = load_ifc_geometry(obj_path, elements)

    # Step 3: Wrap in scene container
    scene = IFCScene(elements=elements, scene_path=obj_path)

    # Step 4: Log summary
    logger.info("IFC Scene parsed successfully:")
    logger.info(f"  Total elements : {len(scene.elements)}")
    logger.info(f"  Doors          : {len(scene.doors)}")
    logger.info(f"  Windows        : {len(scene.windows)}")
    logger.info(f"  Walls          : {len(scene.walls)}")
    logger.info(f"  Furnishings    : {len(scene.furnishings)}")
    logger.info(f"  With geometry  : "
                f"{sum(1 for e in scene.elements.values() if e.bbox is not None)}")

    return scene


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to verify IFC parsing works on your local data:
        python src/ifc_parser.py
    """
    import sys

    # Test on BasicHouse (primary scene)
    labels = "data/BasicHouse_with_pc/_ifcgeom_scene.labels.json"
    obj    = "data/BasicHouse_with_pc/_ifcgeom_scene.obj"

    if not os.path.exists(labels):
        print("Data not found. Please unzip BasicHouse_with_pc into data/")
        sys.exit(1)

    scene = parse_ifc_scene(labels, obj)

    print("\n── IFC Class Summary ──────────────────────────")
    for cls, count in scene.summary().items():
        print(f"  {cls:<35} {count}")

    print("\n── Doors Found ─────────────────────────────────")
    for door in scene.doors:
        bbox_info = (f"center={door.bbox.center.round(2)}"
                     if door.bbox else "no geometry")
        print(f"  [{door.ifc_class}] {door.short_name} | {bbox_info}")

    print("\n── Furnishings Found ───────────────────────────")
    for f in scene.furnishings[:5]:
        bbox_info = (f"center={f.bbox.center.round(2)}"
                     if f.bbox else "no geometry")
        print(f"  [{f.ifc_class}] {f.short_name} | {bbox_info}")

    print("\n── Spatial Query Test ──────────────────────────")
    test_point = np.array([2.0, 1.0, -1.0])
    result = scene.find_nearest_structural_element(test_point)
    if result:
        elem, dist = result
        print(f"  Nearest to {test_point}: [{elem.ifc_class}] "
              f"{elem.short_name} at {dist:.2f}m")
    else:
        print("  No nearby element found for test point")


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    labels = "data/BasicHouse_with_pc/_ifcgeom_scene.labels.json"
    obj    = "data/BasicHouse_with_pc/_ifcgeom_scene.obj"

    if not os.path.exists(labels):
        print("Data not found. Please unzip BasicHouse_with_pc into data/")
        sys.exit(1)

    # ── Phase 1: Labels only (instant) ───────────────────────────────────────
    print("=" * 55)
    print("Phase 1: Loading IFC labels (no geometry)...")
    print("=" * 55)

    elements = load_ifc_labels(labels)
    scene    = IFCScene(elements=elements)

    print(f"\n IFC Class Summary")
    print("-" * 55)
    for cls, count in scene.summary().items():
        print(f"  {cls:<40} {count}")

    print(f"\n Doors ({len(scene.doors)} found)")
    print("-" * 55)
    for door in scene.doors:
        print(f"  [{door.ifc_class}] {door.short_name}")

    print(f"\n Furnishings ({len(scene.furnishings)} found, showing first 5)")
    print("-" * 55)
    for furn in scene.furnishings[:5]:
        print(f"  [{furn.ifc_class}] {furn.short_name}")

    # ── Phase 2: Geometry (slow — 1 to 3 minutes) ────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 2: Loading OBJ geometry...")
    print("This takes 1-3 minutes. Please wait...")
    print("=" * 55)

    elements = load_ifc_geometry(obj, elements)
    with_geo = sum(1 for e in elements.values() if e.bbox is not None)
    print(f"\n Elements with bounding boxes: {with_geo}")

    # ── Phase 3: Spatial query test ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Phase 3: Spatial query test")
    print("=" * 55)

    scene_full = IFCScene(elements=elements)
    test_point = np.array([2.0, 1.0, -1.0])
    result     = scene_full.find_nearest_structural_element(test_point)

    if result:
        elem, dist = result
        print(f"  Nearest to {test_point}:")
        print(f"  [{elem.ifc_class}] {elem.short_name} at {dist:.2f}m")
    else:
        print("  No nearby element found for test point")

    print("\n✅ ifc_parser.py complete!")