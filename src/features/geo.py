import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point


def water_fraction(point_lat: float,
                   point_lon: float,
                   water_shapes: GeoSeries,
                   radius_km: int = 2) -> float:
    km_per_degree_of_lat = 111.32  # https://v-ipc.ru/guides/coord
    radius_deg: float = radius_km / km_per_degree_of_lat

    center: Point | None = Point(point_lon, point_lat)
    circle_gdf: GeoDataFrame = gpd.GeoDataFrame(geometry=[(center.buffer(radius_deg))],
                                                crs="EPSG:4326").to_crs("EPSG:3857")
    water_gdf = gpd.GeoDataFrame(water_shapes, crs="EPSG:4326").to_crs("EPSG:3857")
    intersection: GeoDataFrame = gpd.overlay(circle_gdf, water_gdf, how='intersection')

    if intersection.empty:
        return 0.0

    return np.round(intersection.geometry.area.sum() / circle_gdf.geometry.area.iloc[0], 4)


def extract_polygon(multipolygon, point):
    for geom in multipolygon.geoms:
        if geom.contains(point):
            return geom
    return None


def within_shape(lon, lat, gdf):
    return gdf.geometry.intersects(Point(lon, lat)).any()
