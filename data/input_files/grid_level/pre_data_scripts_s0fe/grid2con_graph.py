import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from constants import *
from graph_constants_falcon import *


def create_graph(nodes_loc, edges):
    G = nx.Graph()

    pos = {}
    for id in nodes_loc:
        area = nodes_loc[id]
        if len(area) == 4:
            imin, imax, jmin, jmax = area
            pos[id] = [(imin + imax)/2.0, (jmin + jmax)/2.0]
        elif len(area) == 8:
            imin1, imax1, jmin1, jmax1, imin2, imax2, jmin2, jmax2 = area
            pos[id] = [(imin1 + imax1 + imin2 + imax2)/4.0, (jmin1 + jmax1 + jmin2 + jmax2)/4.0]

    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges)
    for n, p in pos.items():
       G.nodes[n]['pos'] = p

    node_colors = []
    for id in nodes_loc:
        if 'n' in id:
            node_colors.append('green')
        else:
            node_colors.append('#00b4d9')

    # plt.figure(figsize=(7,8))
    # nx.draw(G, pos=pos, with_labels=True, font_size=12, node_size=500, node_color=node_colors)
    # plt.show()



def dijkstra_with_wall(id, c1, walls, con_points_hub, does_print=False):
    area = ORIG_NODE_LOC[id]
    hole_zmin_c1, hole_zmax_c1, hole_xmin_c1, hole_xmax_c1 = con_points_hub[id][c1]
    source = (hole_zmin_c1, hole_xmin_c1)

    assert len(area) == 4 #TODO: how about irregular areas that are separated
    imin, imax, jmin, jmax = area
    Q = []
    dist = {}
    for i in range(imin, imax):
        for j in range(jmin, jmax):
            if [i, j] not in walls:
                dist[(i, j)] = np.inf
                Q.append((i, j))

    for c2 in con_points_hub[id]:
        hole_zmin_c2, hole_zmax_c2, hole_xmin_c2, hole_xmax_c2 = con_points_hub[id][c2]
        for i in range(hole_zmin_c2, hole_zmax_c2+1):
            for j in range(hole_xmin_c2, hole_xmax_c2+1):
                dist[(i, j)] = np.inf
                if (i, j) not in Q:
                    Q.append((i, j))

    dist[source] = 0

    while len(Q) > 0:
        u = Q[0]
        for q in Q:
            if dist[q] < dist[u]:
                u = q
        Q.remove(u)
        ui, uj = u
        for v in [(ui-1, uj), (ui+1, uj), (ui, uj-1), (ui, uj+1)]:
            if v in Q:
                alt = dist[u] + 1
                if alt < dist[v]:
                    dist[v] = alt
    return dist

def modify_graph(discon, con, con_points_hub):
    # The rest of the graph remain to be the same
    rest_node_loc = {}
    rest_edges = []
    modi_edges = []

    for id in ORIG_NODE_LOC:
        if (id not in discon) or ('n' in id):
            rest_node_loc[id] = ORIG_NODE_LOC[id]

    for e in ORIG_EDGE:
        u, v = e
        if u not in discon and v not in discon:
            rest_edges.append(e)
        else:
            modi_edges.append(e)

    for e in con:
        u, v = e
        if u not in discon and v not in discon:
            rest_edges.append(e)
        else:
            modi_edges.append(e)

    # if a neighbor has to be disconnected from all of the rest neighbors, just remove this edge.
    # Otherwise, conside split.
    split = {}
    for id in discon:
        if 'n' not in id:
            split[id] = []
            avoid_l = []
            neig = set([])
            for e in modi_edges:
                u, v = e
                if u == id:
                    neig.add(v)
                elif v == id:
                    neig.add(u)
            for n1 in neig:
                remove_n1_edge = True
                for n2 in neig:
                    if n1 != n2:
                        if [n1, n2] not in discon[id]:
                            remove_n1_edge = False
                if remove_n1_edge:
                    # print ('edge should remove: ', n1, id)
                    if [n1, id] in modi_edges:
                        modi_edges.remove([n1, id])
                    else:
                        modi_edges.remove([id, n1])
                    avoid_l.append(n1)

            for e in discon[id]:
                u, v = e
                if u not in avoid_l and v not in avoid_l:
                    split[id].append(e)
    new_edges = []
    for e in rest_edges:
        new_edges.append(e)

    for id in discon:
        if 'n' in id: #special case for the room node
            for id2 in con_points_hub[id]:
                if [id, id2] not in discon[id]:
                    new_edges.append([id, id2])

    # This split location is not an accurate one but only for graph
    new_node_loc = {}
    for id in split:
        if len(split[id]) > 0:
            split_neigh = [[]]
            for e in modi_edges:
                if id in e:
                    u, v = e
                    if u == id:
                        n = v
                    else:
                        n = u
                    party = 0
                    done = False
                    while not done:
                        if party < len(split_neigh):
                            this_party = True
                            for ele in split_neigh[party]:
                                if [ele, n] in split[id]:
                                    this_party = False
                            if this_party:
                                split_neigh[party].append(n)
                                done = True
                            else:
                                party += 1
                        else:
                            split_neigh.append([])
            n_e = []
            for ext in range(len(split_neigh)):
                new_id = id + '_' + str(ext)
                pos = ORIG_NODE_LOC[id]
                new_node_loc[new_id] = [p+ext*4 for p in pos]
                new_node_loc[new_id][0] = pos[0]
                new_node_loc[new_id][1] = pos[1]
                for ele in split_neigh[ext]:
                    new_edges.append([new_id, ele])
        else:
            new_node_loc[id] = ORIG_NODE_LOC[id]
            for e in modi_edges:
                if id in e:
                    new_edges.append(e)
    for id in rest_node_loc:
        new_node_loc[id] = rest_node_loc[id]

    # print ('edges', new_edges)
    create_graph(new_node_loc, new_edges)
    return new_node_loc, new_edges


def analyze_wall_separated_nodes(wall_dict, con_points_hub):
    # idea: use bfs to determine what connects are remained
    # if two rooms originally connected are still connected, then it is possible to navigate between their doors.
    discon = {}
    for id in wall_dict:
        if len(wall_dict[id]) > 0:
            discon[id] = []
            if 'n' not in id: # consider walls in hallway and intersections
                for c1 in con_points_hub[id]:
                    dist = dijkstra_with_wall(id, c1, wall_dict[id], con_points_hub, does_print=False)
                    for c2 in con_points_hub[id]:
                        if c1 != c2:
                            t = (con_points_hub[id][c2][0], con_points_hub[id][c2][2])
                            if t in dist:
                                if np.isinf(dist[t]):
                                    discon[id].append([c1, c2])
                            else:
                                discon[id].append([c1, c2])
            else: # consider walls in a room
                for c1 in con_points_hub[id]:
                    dist = dijkstra_with_wall(id, c1, wall_dict[id], con_points_hub, does_print=False)
                    inf_count = len(np.where(np.isinf([dist[k] for k in dist]))[0])
                    if inf_count > len(dist) * 0.9:
                        discon[id].append([id, c1])
    return discon

def analyze_hole_connected_nodes(hole_dict):
    con = {}
    for room_name in hole_dict:
        for hole in hole_dict[room_name]:
            xmin, xmax, ymin, ymax, zmin, zmax = hole
            xbar, zbar = (xmin+xmax)/2.-MAP_XMIN_CROPPED, (zmin+zmax)/2.-MAP_ZMIN
            loc = [zmin - MAP_ZMIN, zmax - MAP_ZMIN, xmin - MAP_XMIN_CROPPED, xmax - MAP_XMIN_CROPPED]
            if '/' in room_name:
                r1, r2 = room_name.split('/')
                if r2[0] == ' ':
                    r2 = r2[1:] #delete space
                if r1 in REGIONNAME_TO_ID and r2 in REGIONNAME_TO_ID:
                    if r1 != r2:
                        con[(REGIONNAME_TO_ID[r1], REGIONNAME_TO_ID[r2])] = loc
            else:
                 #find the room on the other side of the hole if not recorded
                if room_name in REGIONNAME_TO_ID:
                    id1 = REGIONNAME_TO_ID[room_name]
                else:
                    possible_IDs = LOC_TO_ID(zbar, xbar)
                    id1 = possible_IDs[0]

                z_id1, x_id1 = ORIG_NODE_LOC[id1][0], ORIG_NODE_LOC[id1][2]
                z_sign, x_sign = np.sign(z_id1 - zbar), np.sign(x_id1 - xbar)
                if z_sign > 0:
                    z_new = zmin - 1 - MAP_ZMIN
                else:
                    z_new = zmax + 1 - MAP_ZMIN
                if x_sign > 0:
                    x_new = xmin - 1 - MAP_XMIN_CROPPED
                else:
                    x_new = xmax + 1 - MAP_XMIN_CROPPED
                possible_IDs = LOC_TO_ID(z_new, x_new)
                id2 = possible_IDs[0]
                for id in possible_IDs:
                    if id != id1:
                        id2 = id
                if id1 != id2:
                    con[(id1, id2)] = loc
    return con

def find_wall(perturb):
    wall_dict = {}
    for id in ORIG_NODE_LOC:
        wall_dict[id] = []
    for p in perturb:
        if p['feature_type'] == 'Blockage':
            room_name = p['room_name']
            rel_z, rel_x = p['z'] - MAP_ZMIN, p['x'] - MAP_XMIN_CROPPED
            if room_name in REGIONNAME_TO_ID:
                id = REGIONNAME_TO_ID[room_name]
            else:
                possible_IDs = LOC_TO_ID(rel_z, rel_x)
                id = possible_IDs[0]
            if [rel_z, rel_x] not in wall_dict[id]:
                wall_dict[id].append([rel_z, rel_x])
    return wall_dict


def merge_hole(holes): #connected small holes can be considered as a big hole
    merged = []
    pointer = 0
    does_selected = [False] * len(holes)
    while pointer < len(holes):
        px, py, pz = holes[pointer] # construct a new hole
        xmin, xmax, ymin, ymax, zmin, zmax = px, px, py, py, pz, pz
        does_selected[pointer] = True
        for iter in range(len(holes)-pointer):
            for hi in range(pointer, len(holes)):
                if not does_selected[hi]:
                    hx, hy, hz = holes[hi]
                    if xmin-1 <= hx <= xmax+1 and ymin-1 <= hy <= ymax+1 and zmin-1 <= hz <= zmax+1:
                        does_selected[hi] = True
                        xmin, xmax = np.min([xmin, hx]), np.max([xmax, hx])
                        ymin, ymax = np.min([ymin, hy]), np.max([ymax, hy])
                        zmin, zmax = np.min([zmin, hz]), np.max([zmax, hz])
        merged.append([xmin, xmax, ymin, ymax, zmin, zmax])
        while does_selected[pointer] == True:
            pointer += 1
            if pointer > len(holes) - 1:
                break
    return merged

def find_hole(perturb):
    hole_dict = {}
    for p in perturb:
        if p['feature_type'] == 'Opening - Passable':
            if p['room_name'] not in hole_dict:
                hole_dict[p['room_name']] = []
            hole_dict[p['room_name']].append([int(p['x']), int(p['y']), int(p['z'])])

    new_hole_dict = {}
    for r in hole_dict:
        new_hole_dict[r] = merge_hole(hole_dict[r])
    return new_hole_dict

def find_adjacent_room(perturb):
    hole_dict = find_hole(perturb)
    con = analyze_hole_connected_nodes(hole_dict)
    adjacent_room = {}
    for pair in con:
        if 'n' in pair[0] and 'n' in pair[1]: #both are rooms
            zmin, zmax, xmin, xmax = con[pair]
            if (pair[1], pair[0]) in adjacent_room:
                z, x = adjacent_room[(pair[1], pair[0])]
                if not (zmin - 1 <= z <= zmax + 1 and xmin - 1 <= x <= xmax + 1): # the two holes from connected rooms cannot merge
                    adjacent_room[pair] = [(zmin + zmax)/2, (xmin + xmax)/2]
            else:
                adjacent_room[pair] = [(zmin + zmax)/2, (xmin + xmax)/2]
    for room in ORIG_ADJACENT_ROOM:
        adjacent_room[room] = ORIG_ADJACENT_ROOM[room]
    return adjacent_room


def perturbed_graph_from_grid(grid_mat, perturb):
    wall_dict = find_wall(perturb)
    hole_dict = find_hole(perturb)
    con = analyze_hole_connected_nodes(hole_dict)
    con_points_hub = ORIG_CON_POINTS_HUB
    for pair in con:
        id1, id2 = pair
        con_points_hub[id1][id2] = con[pair]
        con_points_hub[id2][id1] = con[pair]
    discon = analyze_wall_separated_nodes(wall_dict, con_points_hub)
    nodes_loc, edges = modify_graph(discon, con, con_points_hub)
    return nodes_loc, edges
