from graph_constants_sparky import *
import csv

distance_path = 'sparky-0_falcon-e/distances/distances_sparky_map0.csv'


def write_distances_matrix(nodes_loc, edges):
    print (nodes_loc)
    print (edges)


    with open(distance_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['from', 'to', 'cost'])

        for e in edges:
            u, v = e[0], e[1]
            uz, ux = sum(nodes_loc[u][0:2])/2, sum(nodes_loc[u][2:4])/2
            vz, vx = sum(nodes_loc[v][0:2])/2, sum(nodes_loc[v][2:4])/2
            dist = abs(uz - vz) + abs(ux - vx)
            # print (u, v, dist)
            # print (v, u, dist)
            # print ()
            spamwriter.writerow([u, v, dist])
            # spamwriter.writerow([v, u, dist]) #TODO: For sparky it is different

write_distances_matrix(NODE_LOC, EDGE)
