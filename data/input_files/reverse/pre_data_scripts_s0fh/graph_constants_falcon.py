import numpy as np
from constants import *

ORIG_NODE_LOC = {'i1': [10, 14, 33, 37],
                 'i2': [10, 14, 55, 59],
                 'i3': [10, 14, 69, 73],
                 'i4': [37, 41, 33, 37],
                 'i5': [37, 41, 55, 59],
                 'i6': [37, 41, 69, 73],
                 'l1': [10, 14, 17, 33],
                 'l2': [10, 14, 37, 55],
                 'l3': [10, 14, 59, 69],
                 'r1': [37, 41, 2, 33],
                 'r2': [37, 41, 37, 55],
                 'r3': [37, 41, 59, 69],
                 'r4': [37, 41, 73, 82],
                 'm1': [14, 37, 33, 37],
                 'm2': [14, 37, 55, 59],
                 'm3': [14, 37, 69, 73],
                 'n01': [2, 10, 15, 21],
                 'n02': [2, 9, 22, 25],
                 'n03': [2, 9, 26, 34],
                 'n04': [2, 9, 35, 51],
                 'n05': [2, 9, 52, 66],
                 'n06': [2, 17, 67, 82],
                 'n07': [2, 17, 83, 90],
                 'n08': [15, 36, 15, 32],
                 'n09': [15, 18, 38, 44],
                 'n10': [19, 27, 38, 44],
                 'n11': [28, 36, 38, 44],
                 'n12': [15, 25, 46, 54],
                 'n13': [26, 36, 46, 54],
                 'n14': [15, 36, 60, 68],
                 'n15': [19, 27, 74, 82],
                 'n16': [28, 36, 74, 82],
                 'n17': [42, 50, 2, 10],
                 'n18': [42, 50, 11, 19],
                 'n19': [42, 50, 20, 28],
                 'n20': [42, 50, 29, 37],
                 'n21': [42, 50, 38, 46],
                 'n22': [42, 50, 47, 55],
                 'n23': [42, 50, 56, 64],
                 'n24': [42, 50, 65, 73],
                 'n25': [42, 50, 74, 82]
                 }

EXPAND_NODE_LOC = {'i1': [10, 14, 33, 37],
                 'i2': [10, 14, 55, 59],
                 'i3': [10, 14, 69, 73],
                 'i4': [37, 41, 33, 37],
                 'i5': [37, 41, 55, 59],
                 'i6': [37, 41, 69, 73],
                 'l1': [10, 14, 17, 33],
                 'l2': [10, 14, 37, 55],
                 'l3': [10, 14, 59, 69],
                 'r1': [37, 41, 2, 33],
                 'r2': [37, 41, 37, 55],
                 'r3': [37, 41, 59, 69],
                 'r4': [37, 41, 73, 82],
                 'm1': [14, 37, 33, 37],
                 'm2': [14, 37, 55, 59],
                 'm3': [14, 37, 69, 73],
                 'n01': [2, 10, 15, 21],
                 'n02': [2, 10, 22, 25],
                 'n03': [2, 10, 26, 34],
                 'n04': [2, 10, 35, 51],
                 'n05': [2, 10, 52, 66],
                 'n06': [2, 18, 67, 82],
                 'n07': [2, 17, 83, 90],
                 'n08': [14, 37, 15, 33],
                 'n09': [14, 18, 37, 45],
                 'n10': [18, 28, 37, 45],
                 'n11': [28, 37, 37, 45],
                 'n12': [14, 26, 45, 55],
                 'n13': [26, 37, 45, 55],
                 'n14': [14, 37, 59, 69],
                 'n15': [18, 28, 73, 82],
                 'n16': [28, 37, 73, 82],
                 'n17': [41, 50, 2, 10],
                 'n18': [41, 50, 10, 19],
                 'n19': [41, 50, 19, 28],
                 'n20': [41, 50, 28, 37],
                 'n21': [41, 50, 37, 46],
                 'n22': [41, 50, 46, 55],
                 'n23': [41, 50, 55, 64],
                 'n24': [41, 50, 64, 73],
                 'n25': [41, 50, 73, 82]
                 }

ORIG_EDGE = [['i1', 'l1'], ['i1', 'n04'], ['i1', 'l2'], ['i1', 'm1'],
            ['i2', 'l2'], ['i2', 'l3'], ['i2', 'm2'],
            ['i3', 'l3'], ['i3', 'm3'],
            ['i4', 'm1'], ['i4', 'r1'], ['i4', 'r2'],
            ['i5', 'm2'], ['i5', 'r2'], ['i5', 'n23'], ['i5', 'r3'],
            ['i6', 'm3'], ['i6', 'r3'], ['i6', 'r4'],
            ['l1', 'n01'], ['l1', 'n02'], ['l1', 'n03'], ['l1', 'n08'],
            ['l2', 'n05'], ['l2', 'n09'],
            ['l3', 'n06'],
            ['m1', 'n10'], ['m1', 'n11'],
            ['m2', 'n12'], ['m2', 'n13'], ['m2', 'n14'],
            ['m3', 'n14'], ['m3', 'n15'],
            ['r1', 'n08'], ['r1', 'n17'], ['r1', 'n18'], ['r1', 'n19'], ['r1', 'n20'],
            ['r2', 'n21'], ['r2', 'n22'],
            ['r3', 'n24'],
            ['r4', 'n16'], ['r4', 'n25'],
            ['n06', 'n07'],
            ] #TODO: could be commeted out and made from the connection data as below


ORIG_INTERSECTION_HALLWAY = {('i1', 'l1'): [10, 14, 33, 34],
                             ('i1', 'l2'): [10, 14, 37, 38],
                             ('i1', 'm1'): [14, 15, 33, 37],
                             ('i2', 'l2'): [10, 14, 54, 55],
                             ('i2', 'l3'): [10, 14, 59, 60],
                             ('i2', 'm2'): [14, 15, 55, 59],
                             ('i3', 'l3'): [10, 14, 69, 70],
                             ('i3', 'm3'): [14, 15, 69, 73],
                             ('i4', 'm1'): [37, 38, 33, 37],
                             ('i4', 'r1'): [37, 41, 33, 34],
                             ('i4', 'r2'): [37, 41, 37, 38],
                             ('i5', 'm2'): [37, 38, 55, 59],
                             ('i5', 'r2'): [37, 41, 55, 56],
                             ('i5', 'r3'): [37, 41, 59, 60],
                             ('i6', 'm3'): [36, 37, 69, 73],
                             ('i6', 'r3'): [37, 41, 69, 70],
                             ('i6', 'r4'): [37, 41, 73, 74]}

ORIG_DOORS = [ # directly copied from the clean map data, some of them are invalid due to perturbation
    [[-2028, 61, 151], [-2028, 60, 151], "wooden_door", 9, 0, ['n06', 'n07']],
    [[-2036, 61, 183], [-2036, 60, 183], "dark_oak_door", 8, 1, ['n25', 'r4']],
    [[-2036, 61, 178], [-2036, 60, 178], "dark_oak_door", 9, 3, ['n16', 'r4']],
    [[-2037, 61, 168], [-2037, 60, 168], "dark_oak_door", 8, 0, ['n15', 'm3']],
    [[-2042, 61, 159], [-2042, 60, 159], "dark_oak_door", 8, 2, ['n14', 'm3']],
    [[-2042, 61, 158], [-2042, 60, 158], "dark_oak_door", 9, 2, ['n14', 'm3']],
    [[-2042, 61, 151], [-2042, 60, 151], "dark_oak_door", 9, 3, ['n06', 'l3']],
    [[-2043, 61, 151], [-2043, 60, 151], "dark_oak_door", 8, 3, ['n06', 'l3']],
    [[-2045, 61, 183], [-2045, 60, 183], "dark_oak_door", 8, 1, ['n24', 'r3']],
    [[-2051, 61, 176], [-2051, 60, 176], "dark_oak_door", 9, 0, ['n14', 'm2']],
    [[-2051, 61, 175], [-2051, 60, 175], "dark_oak_door", 8, 0, ['n14', 'm2']],
    [[-2054, 61, 183], [-2054, 60, 183], "dark_oak_door", 8, 1, ['n23', 'i5']],
    [[-2056, 61, 169], [-2056, 60, 169], "dark_oak_door", 8, 2, ['n13', 'm2']],
    [[-2056, 61, 168], [-2056, 60, 168], "dark_oak_door", 9, 2, ['n13', 'm2']],
    [[-2056, 61, 166], [-2056, 60, 166], "dark_oak_door", 8, 2, ['n12', 'm2']],
    [[-2056, 61, 165], [-2056, 60, 165], "dark_oak_door", 9, 2, ['n12', 'm2']],
    [[-2058, 61, 151], [-2058, 60, 151], "dark_oak_door", 8, 3, ['n05', 'l2']],
    [[-2063, 61, 183], [-2063, 60, 183], "dark_oak_door", 8, 1, ['n22', 'r2']],
    [[-2069, 61, 176], [-2069, 60, 176], "wooden_door", 8, 2, None],
    [[-2069, 61, 172], [-2069, 60, 172], "wooden_door", 9, 2, None],
    [[-2069, 61, 167], [-2069, 60, 167], "wooden_door", 9, 2, None],
    [[-2069, 61, 163], [-2069, 60, 163], "wooden_door", 9, 2, None],
    [[-2071, 61, 156], [-2071, 60, 156], "dark_oak_door", 8, 1, ['n09', 'l2']],
    [[-2072, 61, 183], [-2072, 60, 183], "dark_oak_door", 9, 1, ['n21', 'r2']],
    [[-2073, 61, 170], [-2073, 60, 170], "dark_oak_door", 8, 0, ['n11', 'm1']],
    [[-2073, 61, 161], [-2073, 60, 161], "dark_oak_door", 9, 0, ['n10', 'm1']],
    [[-2075, 61, 151], [-2075, 60, 151], "dark_oak_door", 8, 3, ['n04', 'i1']],
    [[-2081, 61, 183], [-2081, 60, 183], "dark_oak_door", 8, 1, ['n20', 'r1']],
    [[-2087, 61, 178], [-2087, 60, 178], "dark_oak_door", 9, 3, ['n08', 'r1']],
    [[-2087, 61, 156], [-2087, 60, 156], "dark_oak_door", 8, 1, ['n08', 'l1']],
    [[-2087, 61, 151], [-2087, 60, 151], "wooden_door", 8, 3, ['n02', 'l1']],
    [[-2090, 61, 183], [-2090, 60, 183], "dark_oak_door", 8, 1, ['n19', 'r1']],
    [[-2096, 61, 148], [-2096, 60, 148], "acacia_door", 9, 0, None],
    [[-2096, 61, 147], [-2096, 60, 147], "acacia_door", 8, 0, None],
    [[-2096, 61, 145], [-2096, 60, 145], "acacia_door", 9, 0, None],
    [[-2096, 61, 144], [-2096, 60, 144], "acacia_door", 8, 0, None],
    [[-2099, 61, 183], [-2099, 60, 183], "dark_oak_door", 8, 1, ['n18', 'r1']],
    [[-2108, 61, 183], [-2108, 60, 183], "dark_oak_door", 9, 1, ['n17', 'r1']],
    [[-2092, 61, 151], [-2092, 60, 151], 'air_door', 0, 0, ['n01', 'l1']], # not real doors, but record to complete the graph design
    [[-2081, 61, 151], [-2081, 60, 151], 'air_door', 0, 0, ['n03', 'l1']]
  ]


ORIG_ADJACENT_ROOM = {('n06', 'n07'): [9, 82]}


# not real doors, but record to complete the graph design
OPEN_AREA_DOOR = [
    [[-2092, 61, 151], [-2092, 60, 151], ['n01', 'l1']],
    [[-2081, 61, 151], [-2081, 60, 151], ['n03', 'l1']],
    [[-2067, 61, 178], [-2067, 60, 178], ['n11', 'r2']] # only for the hard map
]

ORIG_CON_POINTS_PLAIN = ORIG_INTERSECTION_HALLWAY
for row in ORIG_DOORS:
    s, e, t, f, o, l = row
    if l != None:
        rel_z = s[2] - MAP_ZMIN
        rel_x = s[0] - MAP_XMIN_CROPPED
        if (l[0], l[1]) not in ORIG_CON_POINTS_PLAIN:
            ORIG_CON_POINTS_PLAIN[(l[0], l[1])] = [rel_z, rel_z, rel_x, rel_x]
        else:
            zmin, zmax, xmin, xmax = ORIG_CON_POINTS_PLAIN[(l[0], l[1])]
            ORIG_CON_POINTS_PLAIN[(l[0], l[1])] = [np.min([zmin, rel_z]), np.max([zmax, rel_z]), np.min([xmin, rel_x]), np.max([xmax, rel_x])]

ORIG_CON_POINTS_HUB = {}
for id in ORIG_NODE_LOC:
    ORIG_CON_POINTS_HUB[id] = {}
    for pair in ORIG_CON_POINTS_PLAIN:
        if id in pair:
            if id == pair[0]:
                id2 = pair[1]
            else:
                id2 = pair[0]
            ORIG_CON_POINTS_HUB[id][id2] = ORIG_CON_POINTS_PLAIN[pair]

ID_TO_REGIONNAME = {'i1': 'Left Bottom Intersection',
                    'i2': 'Left Middle Intersection',
                    'i3': 'Left Top Intersection',
                    'i4': 'Right Bottom Intersection',
                    'i5': 'Right Middle Intersection',
                    'i6': 'Right Top Intersection',
                    'l1': 'Left Bottom Hallway',
                    'l2': 'Left Middle Hallway',
                    'l3': 'Left Top Hallway',
                    'r1': 'Right Bottom Hallway',
                    'r2': 'Right Middle Hallway',
                    'r3': 'Right Top1 Hallway',
                    'r4': 'Right Top2 Hallway',
                    'm1': 'Center Bottom Hallway',
                    'm2': 'Center Middle Hallway',
                    'm3': 'Center Top Hallway',
                    'n01': 'Entrance',
                    'n02': 'Secruity Office',
                    'n03': 'Open Break Area',
                    'n04': 'Executive Suite 1',
                    'n05': 'Executive Suite 2',
                    'n06': 'King Chris\'s Office',
                    'n07': 'The King\'s Terrace',
                    'n08': 'The Computer Farm',
                    'n09': 'Janitor',
                    'n10': 'Men\'s Room',
                    'n11': 'Women\'s Room',
                    'n12': 'Amway Conference Room',
                    'n13': 'Mary Kay Conference Room',
                    'n14': 'Herbalife Conference Room',
                    'n15': 'Room 101',
                    'n16': 'Room 102',
                    'n17': 'Room 111',
                    'n18': 'Room 110',
                    'n19': 'Room 109',
                    'n20': 'Room 108',
                    'n21': 'Room 107',
                    'n22': 'Room 106',
                    'n23': 'Room 105',
                    'n24': 'Room 104',
                    'n25': 'Room 103'
                    }

REGIONNAME_TO_ID = {}
for id in ID_TO_REGIONNAME:
    REGIONNAME_TO_ID[ID_TO_REGIONNAME[id]] = id

def LOC_TO_ID(rel_z, rel_x):
    possible_IDs = []
    for id in EXPAND_NODE_LOC:
        area = EXPAND_NODE_LOC[id]
        if len(area) == 4:
            imin, imax, jmin, jmax = area
            if imin <= rel_z <= imax and jmin <= rel_x <= jmax:
                possible_IDs.append(id)
        elif len(area) == 8:
            imin1, imax1, jmin1, jmax1, imin2, imax2, jmin2, jmax2 = area
            if imin1 <= rel_z <= imax1 and jmin1 <= rel_x <= jmax1:
                possible_IDs.append(id)
            elif imin2 <= rel_z <= imax2 and jmin2 <= rel_x <= jmax2:
                possible_IDs.append(id)
    # if len(possible_IDs) == 0:
    #     print ('no id found')
    return possible_IDs


def ID_TO_LOC(id):
    area = ORIG_NODE_LOC[id]
    if len(area) == 4:
        imin, imax, jmin, jmax = area
        return (imin + imax)/2., (jmin + jmax)/2.
    elif len(area) == 8:
        imin1, imax1, jmin1, jmax1, imin2, imax2, jmin2, jmax2 = area
        return (imin1 + imax1 + imin2 + imax2)/2., (jmin1 + jmax1 + jmin2 + jmax2)/2.
