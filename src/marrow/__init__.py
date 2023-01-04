__version__ = "0.0.1"

from shapely.geometry import Polygon,Point,LineString,MultiLineString,MultiPoint
from shapely.affinity import translate
from shapely.ops import nearest_points,unary_union,linemerge
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


def disp(poly, *args, ax=None, **kwargs):
    res = gpd.GeoDataFrame(geometry=[poly]).plot(ax=ax, *args, **kwargs)
    if ax is None:
        return res
    else:
        return ax


def simple_polygon_skeleton(poly):
    ax=None
    contour = poly.exterior
    min_dist = 0.125/5 # Should be < min distance between all points / 2
    max_dist = 10 # Should be > max distance between all points

    c = contour.coords[:-1]
    v = np.array(list(zip(c[-1:] + c[:-1], c, c[1:] + c[:1])))
    v1 = v[:,1,:] - v[:,0,:]
    v2 = v[:,2,:] - v[:,1,:]
    v1 = v1 / (np.sum(v1**2, axis=1)**.5).reshape(-1,1)
    v2 = v2 / (np.sum(v2**2, axis=1)**.5).reshape(-1,1)
    
    vv = np.array([
        s * (v - u) if (s := (-np.sign(np.arctan2(np.cross(u,v), sum(u**2)**.5 * sum(v**2)**.5)))) != 0
        else np.array([[0,1],[-1,0]]).dot(v)
        for u,v in zip(v1,v2)
    ])

    c1 = c
    c2 = c + vv

    ax = disp(poly, alpha=0.4)
    for p,q in zip(c1,c2):
        ax = disp(LineString([p,q]), ax=ax, color="r")
    plt.show()
    

    vs = (c2 - c1)

    res = MultiLineString([])

    old_b = None
    ax=None
    for i,(p,v) in enumerate(zip(c1, vs)):
        b = Point(p)
        l = LineString([p + v * min_dist, p + v * max_dist])
        if (
            l.intersects(contour) and l.intersects(res) and 
            b.distance(l.intersection(res)) < b.distance(l.intersection(contour))
        ) or (
            not l.intersects(contour) and l.intersects(res)
        ):
            b = nearest_points(b, l.intersection(res))[1]
            old_b = None
        else:
            d = 0
            while 1:
                new_b = translate(b, *(v*min_dist))
                new_d = new_b.distance(contour)
                if new_d < d:
                    break
                else:
                    b = new_b
                    d = new_d

            if old_b is not None:
                res = res.union(LineString([old_b, b]))
            elif not res.is_empty:
                old_b = nearest_points(b, res)[1]
                res = res.union(LineString([old_b, b]))

            ax = disp(Point(b).buffer(d), ax=ax, alpha=i/len(c1))
            old_b = b
        ax = disp(LineString([p, p + v / 4]), ax=ax, color="r")
        
    disp(res, ax=ax, color="g")

    disp(poly, ax=ax, alpha=0.4)
    plt.show()


if __name__ == "__main__":
    poly = Polygon([(0,0), (1,0), (1,1), (0,1)]).difference(
        Polygon([(0.25,0), (0.75,0), (0.75,0.5), (0.25,0.5)])
    ).union(
        Polygon([(-1,0),(0,0),(0,0.25),(-1,0.25)])
    )
    simple_polygon_skeleton(poly)
    



