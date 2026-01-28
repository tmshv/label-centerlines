# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

label_centerlines extracts smoothed centerlines from Polygon/MultiPolygon geometries for label placement. It uses Voronoi diagrams to create a polygon skeleton, then selects and smooths the centerline.

## Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest test/test_module.py::test_centerline

# Lint
uv run ruff check

# Type check
uv run ty check

# Run CLI
uv run label_centerlines --help
uv run label_centerlines input.geojson output.geojson
```

## Architecture

The package has two main entry points:

1. **Python API** (`label_centerlines.get_centerline`): Takes a Shapely Polygon/MultiPolygon and returns a LineString/MultiLineString centerline
2. **CLI** (`label_centerlines` command): Reads geospatial files via Fiona, processes features in parallel using ProcessPoolExecutor, writes output

### Core Algorithm (`_src.py`)

The centerline extraction process:
1. Segmentize polygon outline to get evenly distributed points
2. Simplify if point count exceeds `max_points`
3. Build Voronoi diagram from outline points
4. Create NetworkX graph from Voronoi edges inside the polygon
5. Find longest paths between end nodes using Dijkstra
6. Select least curved path from candidates
7. Smooth with Gaussian filter

### Exception Handling

`CenterlineError` is raised when centerline extraction fails (e.g., too few points, no paths found). For MultiPolygons, individual part failures are logged but only raise if all parts fail.
