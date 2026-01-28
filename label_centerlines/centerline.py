import logging
import operator
from collections.abc import Generator, Sequence
from itertools import combinations
from typing import Literal, overload

import networkx as nx
import numpy as np
from networkx.exception import NetworkXNoPath
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Voronoi
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from label_centerlines.exceptions import CenterlineError

logger = logging.getLogger(__name__)


def get_centerline(
    geom: Polygon | MultiPolygon,
    segmentize_maxlen: float = 0.5,
    max_points: int = 3000,
    simplification: float = 0.05,
    smooth_sigma: float = 5,
    max_paths: int = 5,
) -> LineString | MultiLineString:
    """
    Return centerline from geometry.

    Parameters:
    -----------
    geom : shapely Polygon or MultiPolygon
    segmentize_maxlen : Maximum segment length for polygon borders.
        (default: 0.5)
    max_points : Number of points per geometry allowed before simplifying.
        (default: 3000)
    simplification : Simplification threshold.
        (default: 0.05)
    smooth_sigma : Smoothness of the output centerlines.
        (default: 5)
    max_paths : Number of longest paths used to create the centerlines.
        (default: 5)

    Returns:
    --------
    geometry : LineString or MultiLineString

    Raises:
    -------
    CenterlineError : if centerline cannot be extracted from Polygon
    TypeError : if input geometry is not Polygon or MultiPolygon

    """
    logger.debug("geometry type %s", geom.geom_type)

    if geom.geom_type == "Polygon":
        # segmentized Polygon outline
        outline = _segmentize(geom.exterior, segmentize_maxlen)
        logger.debug("outline: %s", outline)

        # simplify segmentized geometry if necessary and get points
        outline_points = outline.coords
        simplification_updated = simplification
        while len(outline_points) > max_points:
            # if geometry is too large, apply simplification until geometry
            # is simplified enough (indicated by the "max_points" value)
            simplification_updated += simplification
            outline_points = outline.simplify(simplification_updated).coords
        logger.debug("simplification used: %s", simplification_updated)
        logger.debug("simplified points: %s", MultiPoint(outline_points))

        # calculate Voronoi diagram and convert to graph but only use points
        # from within the original polygon
        vor = Voronoi(outline_points)
        graph = _graph_from_voronoi(vor, geom)
        logger.debug("voronoi diagram: %s", _multilinestring_from_voronoi(vor, geom))

        # determine longest path between all end nodes from graph
        end_nodes = _get_end_nodes(graph)
        if len(end_nodes) < 2:
            logger.debug("Polygon has too few points")
            raise CenterlineError("Polygon has too few points")
        logger.debug("get longest path from %s end nodes", len(end_nodes))
        longest_paths = _get_longest_paths(end_nodes, graph, max_paths)
        if not longest_paths:
            logger.debug("no paths found between end nodes")
            raise CenterlineError("no paths found between end nodes")
        if logger.getEffectiveLevel() <= 10:
            logger.debug("longest paths:")
            for path in longest_paths:
                logger.debug(LineString(vor.vertices[path]))

        # get least curved path from the longest paths, smooth and
        # return as LineString
        centerline = _smooth_linestring(
            LineString(vor.vertices[_get_least_curved_path(longest_paths, vor.vertices)]),
            smooth_sigma,
        )
        logger.debug("centerline: %s", centerline)
        logger.debug("return linestring")
        return centerline

    elif geom.geom_type == "MultiPolygon":
        logger.debug("MultiPolygon found with %s sub-geometries", len(geom.geoms))
        # get centerline for each part Polygon and combine into MultiLineString
        sub_centerlines: list[LineString | MultiLineString] = []
        for subgeom in geom.geoms:
            try:
                sub_centerline = get_centerline(subgeom, segmentize_maxlen, max_points, simplification, smooth_sigma)
                sub_centerlines.append(sub_centerline)
            except CenterlineError as e:
                logger.debug("subgeometry error: %s", e)
        # for MultPolygon, only raise CenterlineError if all subgeometries fail
        if sub_centerlines:
            return MultiLineString(sub_centerlines)
        else:
            raise CenterlineError("all subgeometries failed")

    else:
        raise TypeError(f"Geometry type must be Polygon or MultiPolygon, not {geom.geom_type}")


def _segmentize(geom: LinearRing, max_len: float) -> LineString:
    """Interpolate points on segments if they exceed maximum length."""
    points: list[tuple[float, float]] = []
    for previous, current in zip(geom.coords, geom.coords[1:], strict=False):
        line_segment = LineString([previous, current])
        # add points on line segment if necessary
        points.extend([line_segment.interpolate(max_len * i).coords[0] for i in range(int(line_segment.length / max_len))])
        # finally, add end point
        points.append(current)
    return LineString(points)


def _smooth_linestring(linestring: LineString, smooth_sigma: float) -> LineString:
    """Use a gauss filter to smooth out the LineString coordinates."""
    return LineString(
        zip(
            np.array(gaussian_filter1d(linestring.xy[0], smooth_sigma)),
            np.array(gaussian_filter1d(linestring.xy[1], smooth_sigma)),
            strict=False,
        )
    )


def _get_longest_paths(nodes: list[int], graph: nx.Graph, max_paths: int) -> list[list[int]]:
    """Return longest paths of all possible paths between a list of nodes."""

    def _gen_paths_distances() -> Generator[tuple[float, list[int]], None, None]:
        for node1, node2 in combinations(nodes, r=2):
            try:
                yield nx.single_source_dijkstra(G=graph, source=node1, target=node2, weight="weight")
            except NetworkXNoPath:
                continue

    return [x for (y, x) in sorted(_gen_paths_distances(), reverse=True)][:max_paths]


def _get_least_curved_path(paths: list[list[int]], vertices: NDArray[np.floating]) -> list[int]:
    """Return path with smallest angles."""
    return min(
        zip(
            [_get_path_angles_sum(path, vertices) for path in paths],
            paths,
            strict=False,
        ),
        key=operator.itemgetter(0),
    )[1]


def _get_path_angles_sum(path: Sequence[int], vertices: NDArray[np.floating]) -> float:
    """Return all angles between edges from path."""
    return float(
        sum(
            [
                _get_absolute_angle((vertices[pre], vertices[cur]), (vertices[cur], vertices[nex]))
                for pre, cur, nex in zip(path[:-1], path[1:], path[2:], strict=False)
            ]
        )
    )


def _get_absolute_angle(
    edge1: tuple[NDArray[np.floating], NDArray[np.floating]],
    edge2: tuple[NDArray[np.floating], NDArray[np.floating]],
) -> np.floating:
    """Return absolute angle between edges."""
    v1 = edge1[0] - edge1[1]
    v2 = edge2[0] - edge2[1]
    return abs(np.degrees(np.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))))


def _get_end_nodes(graph: nx.Graph) -> list[int]:
    """Return list of nodes with just one neighbor node."""
    return [i for i in graph.nodes() if len(list(graph.neighbors(i))) == 1]


def _graph_from_voronoi(vor: Voronoi, geometry: Polygon) -> nx.Graph:
    """Return networkx.Graph from Voronoi diagram within geometry."""
    graph = nx.Graph()
    for x, y, dist in _yield_ridge_vertices(vor, geometry, dist=True):
        graph.add_nodes_from([x, y])
        graph.add_edge(x, y, weight=dist)
    return graph


def _multilinestring_from_voronoi(vor: Voronoi, geometry: Polygon) -> MultiLineString:
    """Return MultiLineString geometry from Voronoi diagram."""
    return MultiLineString(
        [LineString([Point(vor.vertices[[x, y]][0]), Point(vor.vertices[[x, y]][1])]) for x, y in _yield_ridge_vertices(vor, geometry)]
    )


@overload
def _yield_ridge_vertices(vor: Voronoi, geometry: Polygon, dist: Literal[True]) -> Generator[tuple[int, int, float], None, None]: ...


@overload
def _yield_ridge_vertices(vor: Voronoi, geometry: Polygon, dist: Literal[False] = ...) -> Generator[tuple[int, int], None, None]: ...


def _yield_ridge_vertices(
    vor: Voronoi, geometry: Polygon, dist: bool = False
) -> Generator[tuple[int, int, float], None, None] | Generator[tuple[int, int], None, None]:
    """Yield Voronoi ridge vertices within geometry."""
    for x, y in vor.ridge_vertices:
        if x < 0 or y < 0:
            continue
        point1 = Point(vor.vertices[[x, y]][0])
        point2 = Point(vor.vertices[[x, y]][1])
        # Eliminate all points outside our geometry.
        if point1.within(geometry) and point2.within(geometry):
            if dist:
                yield x, y, point1.distance(point2)
            else:
                yield x, y
