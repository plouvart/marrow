__version__ = "0.0.1"

from shapely.geometry import Polygon,Point,LineString,MultiLineString,MultiPoint,LinearRing,MultiPolygon
from shapely.affinity import translate,rotate
from shapely.ops import nearest_points,unary_union,linemerge,substring,snap,polygonize
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
    u = np.array([c2[0] - c1[0], c2[1] - c1[1]])
    u = normalize(u)
    v = np.array([c0[0] - c1[0], c0[1] - c1[1]])
    v = normalize(v)
    a = ((np.arctan2(*v[::-1]) + np.arctan2(*u[::-1])) / 2 if np.arctan2(*v[::-1]) > np.arctan2(*u[::-1]) else (np.arctan2(*v[::-1]) + np.arctan2(*u[::-1])) / 2 + np.pi)
    # return normalize(s * (v - u) if (s := (np.sign(np.arctan2(np.cross(u,v), sum(u**2)**.5 * sum(v**2)**.5)))) != 0 else np.array([[0,-1],[1,0]]).dot(v))
    return np.array([np.cos(a), np.sin(a)])

def get_direction_vec(
    base_coords, 
    direction_vec,
    contours,
):
    tip_coords = base_coords + direction_vec
    normal_coords = get_contour_coords(base_coords, contours)
    normal_vec = normalize(base_coords - normal_coords)

    next_direction_vec = direction_vec - 2 * sum(direction_vec * normal_vec) * normal_vec
    # ax = disp(MultiLineString(contours), ax=None)
    # disp(LineString([base_coords, tip_coords]), ax=ax, color="b")
    # disp(LineString([base_coords, normal_coords]), ax=ax, color="g")
    # disp(LineString([base_coords, base_coords + next_direction_vec]), ax=ax, color="r")
    # disp(Point(normal_coords), color="purple", ax=ax)
    # print(next_direction_vec, normal_vec, direction_vec, (direction_vec * normal_vec) * normal_vec)
    # plt.show()

    return normalize(next_direction_vec)


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
            angle = angle_between(np.array(coords[i1]) - coords[i], np.array(next_base_coords) - coords[i])
            # ax=disp(MultiLineString(contours), ax=None, color="b")
            # disp(Point(coords[i]), ax=ax, color="r")
            # disp(Point(next_base_coords), ax=ax, color="g")
            # disp(Point(Point(coords[i1])), ax=ax, color="orange")
            # print(angle, angle*180/np.pi)
            # plt.show()
            return angle
        if pd > distance: # inbetween points
            angle = angle_between(np.array(coords[i-1]) - contour.interpolate(distance).coords[0], np.array(next_base_coords) - contour.interpolate(distance).coords[0])
            # ax=disp(MultiLineString(contours), ax=None, color="b")
            # disp(contour.interpolate(distance), ax=ax, color="r")
            # disp(Point(next_base_coords), ax=ax, color="g")
            # disp(Point(Point(coords[i-1])), ax=ax, color="orange")
            # print(angle, angle*180/np.pi)
            # plt.show()
            return angle
    

def angle_between(u,v):
    # return np.arctan2(np.cross(u,v), np.dot(u,v))
    return (np.arctan2(*v)-np.arctan2(*u)) % (2*np.pi)



def normalize(v):
    return v / sum(v**2)**.5

def skel_iteration(
    previous_base_coords,
    base_coords,
    direction_vec,
    base_seg,
    contours,
    debug=False,
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

        if debug:
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

    return next_base_coords


def disp_backlogs(backlogs, contours, base_seg, base_coords, direction_vec):
    ax = None
    ax = disp(MultiLineString(contours), ax=ax, color="b")
    disp(base_seg, ax=ax, color="purple")
    disp(LineString([base_coords, base_coords + direction_vec * 0.3]), ax=ax, color="yellow")
    disp(Point(base_coords), ax=ax, color="black")
    disp(Point(base_coords + direction_vec * 0.3), ax=ax, color="g")
    print("DIRECTION:", direction_vec)
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
    colors = ["red", "blue", "green"]

    ax = disp(MultiLineString(contours),ax=None, color="b")
    for color,skel_list in zip(colors, skel_lists):
        for (c1,_,p1),(c2,_,p2) in zip(skel_list[:-1], skel_list[1:]):
            disp(LineString([contours[0].interpolate(c1), p1]), ax=ax, color="g")
            disp(LineString([contours[0].interpolate(c2), p2]), ax=ax, color="g")
            disp(LineString([p1, p2]), ax=ax, color=color)
    disp(skel_points, ax=ax, color="r")
    plt.show()





def complex_polygon_skeleton(poly):
    t1 = time.time()
    contours = [LineString(poly.exterior.coords[::-1])] + [LineString(i.coords[::-1]) for i in poly.interiors]

    backlogs = [
        (
            None,
            base_coords + (direction_vec := normalize(get_base_normal(base_coords, contours))) * MIN_DIST,
            normalize(direction_vec),
            get_base_seg(base_coords, contours),
        )
        for contour in contours
        for base_coords in contour.coords[:-1]
    ]

    skel_lists = [[] for _ in contours]
    junction_segs = list()

    base_points = MultiPoint([p for contour in contours for p in contour.coords])
    skel_points = MultiPoint([])


    count = 0
    while backlogs:
        previous_base_coords,base_coords,direction_vec,base_seg = backlogs.pop(0)
        count += 1
        if count >= 10000:
            disp_backlogs(backlogs, contours, base_seg, base_coords, direction_vec)
            disp_result(skel_lists, contours, skel_points)

        # if previous_base_coords is not None and Point(previous_base_coords).distance(Point(base_coords)) < MIN_DIST * 10:
            # continue

        next_base_coords = skel_iteration(
            previous_base_coords,
            base_coords,
            direction_vec,
            base_seg,
            contours,
            debug = count >= 10000
        )

        # backlogs.insert(
            # 0,
            # (
                # base_coords,
                # next_base_coords,
                # next_direction_vec,
                # next_base_seg,
            # ),
        # )

        if not skel_points.is_empty and Point(next_base_coords).distance(skel_points) < 0.05:
            next_base_coords = snap(Point(next_base_coords), skel_points, 0.05).coords[0]
        else:
            skel_points = skel_points.union(Point(next_base_coords))

        snapped_base_coords = snap(Point(base_coords), base_points, MIN_DIST*10).coords[0]
        skel_lists[
            get_contour_id(base_coords, contours)
        ].append(
            (
                get_contour_ratio(snapped_base_coords, contours),
                get_angle(base_coords, next_base_coords, contours),
                next_base_coords,
            )
        )
        snapped_base_coords = snap(Point(next_base_coords), base_points, MIN_DIST*10).coords[0]
        skel_lists[
            get_contour_id(next_base_coords, contours)
        ].append(
            (
                get_contour_ratio(snapped_base_coords, contours),
                get_angle(get_contour_coords(snapped_base_coords, contours), next_base_coords, contours),
                next_base_coords,
            )
        )

        # if previous_base_coords is not None:
            # junction_segs.append(LineString([base_coords, next_base_coords]))

    # disp_result(skel_lists, contours, skel_points)

        
    skel_lists = [
        sorted(skel_list, key = lambda c: c[:2])
        for skel_list in skel_lists
    ]

    # ax = disp(MultiLineString(contours),ax=None, color="b")
    # for skel_list in skel_lists:
        # for (_,_,p1),(_,_,p2) in zip(skel_list[:-1], skel_list[1:]):
            # disp(LineString([p1, p2]), ax=ax)
    # disp(skel_points, ax=ax, color="r")
    # plt.show()

    lines = [
        LineString([p1, p2])
        for skel_list in skel_lists
        for (_,_,p1),(_,_,p2) in zip(skel_list[:-1], skel_list[1:])
    ]
    junctions = [
        LineString([contours[i].interpolate(r), p])
        for i,skel_list in enumerate(skel_lists)
        for (r,_,p) in skel_list
    ]
    interiors = MultiPolygon([Polygon(i.coords) for i in poly.interiors])
    polys = [
        poly
        for poly in polygonize(MultiLineString(lines))
        # if not Polygon(poly.exterior.coords).intersects(interiors)
    ]
    # lines = [
        # l
        # for l in lines
        # if not polys.covers(l)
    # ]
    # + [
        # LineString([c, poly.centroid])
        # for poly in polys.geoms
        # for c in poly.exterior.coords
    # ]

    polys_geom = MultiPolygon(polys).buffer(MIN_DIST)
    lines = [
        line
        for line in lines
        if not polys_geom.contains(line)
    ]

    junctions = [
        junction
        for junction in junctions
        if junction.distance(base_points) < MIN_DIST
    ]
    
    sublines = []
    subjunctions = []
    for poly in polys:
        sl,sj = complex_polygon_skeleton(poly)
        lines += sl + sj

    # ax=disp(polys_geom, ax=None, color = "b", alpha=0.2)
    # ax=disp(MultiLineString(lines), ax=ax, color = "r")
    # ax=disp(MultiLineString(junctions), ax=ax, color = "g")
    # plt.show()
    
    return lines,junctions

if __name__ == "__main__":
    # EASY
    # poly = Polygon([(0,0), (1,0), (1,1), (0,1)]).difference(
        # Polygon([(0.25,0.25), (0.75,0.25), (0.25,0.75)])
    # )
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

    poly = Polygon([(0,0), (1,0), (1,1), (0,1)]).difference(
        Polygon([(0.125,0), (0.75,0), (0.75,0.2), (0.25,0.5)])
    ).union(
        Polygon([(-1,0),(0,0),(0,0.25),(-1,0.25)])
    )
    poly = poly.union(translate(poly, 2, 0))

    poly = poly.union(translate(rotate(poly, 90), 1, 1))
    poly = poly.union(translate(rotate(poly, 180), 0, 0))
    poly = poly.union(translate(poly, 3, 1))
    # poly = poly.union(translate(rotate(poly, 45), 1, 3))
    """
    """
    # poly1 = poly.buffer(0.0412)
    # poly = poly1

    # disp(poly)
    # plt.show()
    # simple_polygon_skeleton(poly)
    t1 = time.time()
    lines,junctions = complex_polygon_skeleton(poly)
    print("Time:", time.time() - t1)

    ax = disp(poly, color="b", ax=None)
    ax = disp(MultiLineString(lines), ax=ax, color="r")
    plt.show()
    



