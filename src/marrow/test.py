__version__ = "0.0.1"

from shapely.geometry import Polygon,Point,LineString,MultiLineString,MultiPoint,LinearRing
from shapely.affinity import translate,rotate
from shapely.ops import nearest_points,unary_union,linemerge,substring,snap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import time

MIN_DIST = 0.000001
MAX_DIST = 1000


def disp(poly, *args, ax=None, **kwargs):
    res = gpd.GeoDataFrame(geometry=[poly]).plot(ax=ax, *args, **kwargs)
    if ax is None:
        return res
    else:
        return ax




def get_contour_id(
    base_coords,
    contours,
):
    return sorted(
        enumerate(contours),
        key = lambda t: t[1].distance(Point(base_coords)),
    )[0][0]

def get_contour_ratio(
    base_coords,
    contours,
):
    return contours[get_contour_id(base_coords, contours)].project(Point(base_coords))


def get_base_seg(
    base_coords,
    contours,
):
    contour = contours[get_contour_id(base_coords, contours)]
    distance = contour.project(Point(base_coords))
    coords = list(contour.coords)
    for i,p in enumerate(coords):
        pd = contour.project(Point(p)) if i + 1 < len(coords) else len(coords) - 1
        if pd == distance: # On the point itself
            i1 = i-1 if i > 0 else len(coords) - 2
            i2 = i+1 if i + 1 < len(coords) else 1
            return LineString([coords[i1], coords[i], coords[i2]])
        if pd > distance: # inbetween points
            return LineString([coords[i-1], contour.interpolate(distance).coords[0], coords[i]])


def get_contour_coords(
    base_coords,
    contours,
):
    return np.array(sorted(
        [
            contour.interpolate(contour.project(Point(base_coords)))
            for contour in contours
        ],
        key = lambda p: p.distance(Point(base_coords))
    )[0].coords[0])

def get_base_normal(
    base_coords, 
    contours,
):
    contour = contours[get_contour_id(base_coords, contours)]
    distance = contour.project(Point(base_coords))
    coords = list(contour.coords)
    for i,p in enumerate(coords):
        pd = contour.project(Point(p)) if i + 1 < len(coords) else len(coords) - 1
        if pd == distance: # On the point itself
            i1 = i-1 if i > 0 else len(coords) - 2
            i2 = i+1 if i + 1 < len(coords) else 1
            return normal_from_points(coords[i1], coords[i], coords[i2])
        if pd > distance: # inbetween points
            return normal_from_points(coords[i-1], contour.interpolate(distance).coords[0], coords[i])

def normal_from_points(c0, c1, c2):
    u = np.array([c1[0] - c0[0], c1[1] - c0[1]])
    u = normalize(u)
    v = np.array([c2[0] - c1[0], c2[1] - c1[1]])
    v = normalize(v)
    return s * (v - u) if (s := (np.sign(np.arctan2(np.cross(u,v), sum(u**2)**.5 * sum(v**2)**.5)))) != 0 else np.array([[0,-1],[1,0]]).dot(v)

def get_direction_vec(
    base_coords, 
    direction_vec,
    contours,
):
    tip_coords = base_coords + direction_vec
    normal_coords = get_contour_coords(base_coords, contours)
    normal_vec = normalize(base_coords - normal_coords)

    next_direction_vec = direction_vec - 2 * (direction_vec * normal_vec) * normal_vec
    # ax = disp(MultiLineString(contours), ax=None)
    # disp(Point(tip_coords), ax=ax)
    # disp(Point(base_coords), ax=ax)
    # disp(Point(normal_coords), ax=ax)
    # disp(Point(normal_coords + next_direction_vec), ax=ax)
    # plt.show()

    return next_direction_vec


def get_angle(
    base_coords, 
    next_base_coords, 
    contours,
):
    contour = contours[get_contour_id(base_coords, contours)]
    distance = contour.project(Point(base_coords))
    coords = list(contour.coords)
    for i,p in enumerate(coords):
        pd = contour.project(Point(p)) if i + 1 < len(coords) else len(coords) - 1
        if pd == distance: # On the point itself
            i1 = i-1 if i > 0 else len(coords) - 2
            i2 = i+1 if i + 1 < len(coords) else 1
            return angle_between(np.array(coords[i]) - coords[i1], np.array(base_coords) - coords[i])
        if pd > distance: # inbetween points
            return angle_between(np.array(coords[i]) - coords[i-1], np.array(base_coords) - contour.interpolate(distance).coords[0])
    

def angle_between(u,v):
    return -np.arctan2(np.dot(u,v),np.cross(u,v))



def normalize(v):
    return v / sum(v**2)**.5

def skel_iteration(
    previous_base_coords,
    base_coords,
    direction_vec,
    base_seg,
    contours,
    base_points,
    skel_points,
):
    direction_vec = normalize(direction_vec)

    lines = MultiLineString(contours)
    max_length = Point(base_coords).distance(
        (
            back_point := nearest_points(
                Point(base_coords),
                LineString(
                    [
                        base_coords,
                        base_coords + direction_vec * MAX_DIST,
                    ]
                ).intersection(lines)
            )[1]
        )
    )


    right_l = max_length
    left_l = 0
    good_l = 0
    while 1:
        mid_l = (right_l + left_l) / 2

        if False:
            ax = disp(lines)
            ax = disp(LineString([base_coords, back_point]), color="yellow", ax=ax)
            ax = disp(Point(base_coords), color="b", ax=ax)
            ax = disp(Point(base_coords + direction_vec * right_l), color="r", ax=ax)
            ax = disp(Point(base_coords + direction_vec * left_l), color="g", ax=ax)
            ax = disp(Point(base_coords + direction_vec * mid_l), color="purple", ax=ax)
            ax = disp(base_seg, color="yellow", ax=ax)
            ax = disp(
                nearest_points(
                    Point(base_coords + direction_vec * mid_l), 
                    lines,
                )[1], ax=ax, color="orange"
            )
            plt.show()
        
        if base_seg.distance(
            nearest_points(
                Point(base_coords + direction_vec * mid_l),
                lines,
            )[1]
        ) < MIN_DIST:
            good_l = mid_l
            right_l = right_l
            left_l = mid_l
        else:
            right_l = mid_l
            left_l = left_l

        if ((right_l - left_l) < MIN_DIST):
            break
    
    next_base_coords = base_coords + direction_vec * (good_l + MIN_DIST)

    return (
        next_base_coords,
        get_direction_vec(next_base_coords, direction_vec, contours),
        get_base_seg(next_base_coords, contours),
        base_points,
        skel_points,
    )


def disp_backlogs(backlogs, contours, base_seg, base_coords, direction_vec):
    ax = None
    ax = disp(MultiLineString(contours), ax=ax, color="b")
    disp(base_seg, ax=ax, color="purple")
    disp(LineString([base_coords, base_coords + direction_vec * 0.3]), ax=ax, color="yellow")
    disp(Point(base_coords), ax=ax, color="black")
    disp(Point(base_coords + direction_vec * 0.3), ax=ax, color="g")
    for previous_base_coords,base_coords,direction_vec,base_seg in backlogs:
        disp(LineString([base_coords, base_coords + direction_vec * 0.3]), ax=ax, color="g")
        disp(Point(base_coords), ax=ax, color="r")
        disp(Point(base_coords + direction_vec * 0.3), ax=ax, color="orange")
    plt.show()

def disp_result(skel_lists, contours, skel_points):
    skel_lists = [
        sorted(skel_list, key = lambda c: c[:2])
        for skel_list in skel_lists
    ]

    ax = disp(MultiLineString(contours),ax=None, color="b")
    for skel_list in skel_lists:
        for (_,_,p1),(_,_,p2) in zip(skel_list[:-1], skel_list[1:]):
            disp(LineString([p1, p2]), ax=ax)
    disp(skel_points, ax=ax, color="r")
    plt.show()





def complex_polygon_skeleton(poly):
    contours = [LineString(poly.exterior.coords[::-1])] + [LineString(i.coords[::-1]) for i in poly.interiors]

    backlogs = [
        (
            None,
            base_coords + (direction_vec := get_base_normal(base_coords, contours)) * MIN_DIST,
            direction_vec,
            get_base_seg(base_coords, contours),
        )
        for contour in contours
        for base_coords in contour.coords[:-1]
    ]

    skel_lists = [[] for _ in contours]
    junction_segs = list()

    base_points = MultiPoint([])
    skel_points = MultiPoint([])


    while backlogs:
        previous_base_coords,base_coords,direction_vec,base_seg = backlogs.pop(0)
        disp_backlogs(backlogs, contours, base_seg, base_coords, direction_vec)
        disp_result(skel_lists, contours, skel_points)

        if previous_base_coords is not None and Point(previous_base_coords).distance(Point(base_coords)) < MIN_DIST * 10:
            continue

        (
            next_base_coords,
            next_direction_vec,
            next_base_seg,
            base_points,
            skel_points,
        ) = skel_iteration(
            previous_base_coords,
            base_coords,
            direction_vec,
            base_seg,
            contours,
            base_points,
            skel_points,
        )

        backlogs.insert(
            0,
            (
                base_coords,
                next_base_coords,
                next_direction_vec,
                next_base_seg,
            ),
        )

        if not skel_points.is_empty and Point(next_base_coords).distance(skel_points) < MIN_DIST * 10:
            next_base_coords = snap(Point(next_base_coords), skel_points, MIN_DIST * 10).coords[0]
        else:
            skel_points = skel_points.union(Point(next_base_coords))

        skel_lists[
            get_contour_id(base_coords, contours)
        ].append(
            (
                get_contour_ratio(base_coords, contours),
                get_angle(base_coords, next_base_coords, contours),
                next_base_coords,
            )
        )
        skel_lists[
            get_contour_id(next_base_coords, contours)
        ].append(
            (
                get_contour_ratio(next_base_coords, contours),
                get_angle(next_base_coords, get_contour_coords(next_base_coords, contours), contours),
                next_base_coords,
            )
        )

        if previous_base_coords is not None:
            junction_segs.append(LineString([base_coords, next_base_coords]))

        
    skel_lists = [
        sorted(skel_list, key = lambda c: c[:2])
        for skel_list in skel_lists
    ]

    ax = disp(MultiLineString(contours),ax=None, color="b")
    for skel_list in skel_lists:
        for (_,_,p1),(_,_,p2) in zip(skel_list[:-1], skel_list[1:]):
            disp(LineString([p1, p2]), ax=ax)
    disp(skel_points, ax=ax, color="r")
    plt.show()

    


    ax = disp(all_lines, ax=None, color="b")
    for l in points_lists[:1]:

        l = list(sorted(l, key=lambda x:x[:2]))
        print(l)
        ax = disp(Point(l[0][2]), ax=ax, color="r")


        for r,_,p in l:
            ax = disp(LineString([contours[0].interpolate(r), p]) ,ax=ax, color="r")
        l = [p[2] for p in l]
        lines = MultiLineString(
            [
                [p1,p2] for p1,p2 in zip(l[:-1], l[1:])
            ]
        )
        ax = disp(lines, ax=ax, color="g")
    plt.show()

        
    
    return skeleton

if __name__ == "__main__":
    # EASY
    poly = Polygon([(0,0), (1,0), (1,1), (0,1)]).difference(
        Polygon([(0.25,0), (0.75,0), (0.75,0.5), (0.25,0.5)])
    )

    # LOW
    # poly = Polygon([(0,0), (1,0), (1,1), (0,1)]).difference(
        # Polygon([(0.25,0), (0.75,0), (0.75,0.95), (0.25,0.95)])
    # )

    # ).union(
        # Polygon([(-3,0),(0,0),(0,0.25),(-3,0.25)])
    # ).union(
        # Polygon([(-2,1),(-2.1,-2),(-0.9,-2),(-1,1.5)])
    # )

    """
    poly = Polygon([(0,0), (1,0), (1,1), (0,1)]).difference(
        Polygon([(0.125,0), (0.75,0), (0.75,0.2), (0.25,0.5)])
    ).union(
        Polygon([(-1,0),(0,0),(0,0.25),(-1,0.25)])
    )
    poly = poly.union(translate(poly, 2, 0))

    poly = poly.union(translate(rotate(poly, 90), 1, 1))
    poly = poly.union(translate(rotate(poly, 180), 0, 0))
    poly = poly.union(translate(poly, 3, 1))
    """
    # poly1 = poly.buffer(0.0412)
    # poly = poly1

    # disp(poly)
    # plt.show()
    # simple_polygon_skeleton(poly)
    complex_polygon_skeleton(poly)
    



