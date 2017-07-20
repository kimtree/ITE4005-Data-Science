#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import math
import sys


class Point(object):
    def __init__(self, id, x, y, visited=False):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        self.visited = visited


class DBSCANClusterBuilder(object):
    def __init__(self, input_filename, cluster_count, eps, min_pts):
        self._input_filename = input_filename
        self._cluster_count = int(cluster_count)
        self._eps = int(eps)
        self._min_pts = int(min_pts)
        self.data = set()
        self.noises = set()
        self.clusters = []

    def _load_points_from_input_file(self):
        """Load points from given input file"""
        with open(self._input_filename, 'r') as f:
            for idx, line in enumerate(f):
                # data format [object_id]\t[x_coordinate]\t[y_coordinate]
                data = line.strip().split('\t')
                p = Point(data[0], data[1], data[2])
                self.data.add(p)

    def _get_points_of_neighborhood_radius(self, given_p):
        """Return neighborhood points inside of given radius(eps) for given p"""
        pts = []
        for p in self.data:
            if ((p.x - given_p.x) ** 2 + (p.y - given_p.y) ** 2) <= self._eps ** 2:
                pts.append(p)

        return pts

    def _find_unvisited_point(self):
        """Return unvisited point from all points"""
        for p in self.data:
            if not p.visited:
                return p

        return None

    def _get_avg_points_of_clusters(self):
        """Return all average x-coordinate, y-coordinate point of each cluster"""
        cluster_avg_points = []
        for cluster in self.clusters:
            total_x = 0
            total_y = 0
            for p in cluster:
                total_x += p.x
                total_y += p.y

            p = Point(0, total_x / len(cluster), total_y / len(cluster))
            cluster_avg_points.append(p)

        return cluster_avg_points

    def _sort_clusters_by_count(self):
        self.clusters.sort(key=lambda x: len(x), reverse=True)

    def _adjust(self):
        """Adjust outlier points into nearby cluster"""
        cluster_avg_points = self._get_avg_points_of_clusters()

        # Calculate average dist to average point of cluster
        avg_dist_list = [0] * len(self.clusters)
        for idx, cluster in enumerate(self.clusters):
            for p in cluster:
                avg_point = cluster_avg_points[idx]
                dist = math.sqrt((avg_point.x - p.x) ** 2 + (avg_point.y - p.y) ** 2)
                avg_dist_list[idx] += dist

            avg_dist_list[idx] /= len(cluster)

        # Adjust noises to be included in some cluster
        adjusted_count = 0
        for p in self.noises:
            min_dist = sys.maxsize
            cluster_idx = None
            for idx, avg_point in enumerate(cluster_avg_points):
                dist = math.sqrt((avg_point.x - p.x) ** 2 + (avg_point.y - p.y) ** 2)
                if dist < min_dist and dist <= avg_dist_list[idx] ** 2:
                    min_dist = dist
                    cluster_idx = idx

            if cluster_idx:
                self.clusters[cluster_idx].add(p)
                adjusted_count += 1

        return adjusted_count

    def run(self):
        # Load points from input file
        self._load_points_from_input_file()

        print("Running...")
        while True:
            # Run until there's no unvisited point
            p = self._find_unvisited_point()
            if not p:
                break

            # Mark current point as visited
            p.visited = True

            neighborhood_pts = self._get_points_of_neighborhood_radius(p)
            if len(neighborhood_pts) >= self._min_pts:
                # Trying to generate a new cluster
                new_cluster = set()
                new_cluster.add(p)

                for neighborhood_p in neighborhood_pts:
                    if not neighborhood_p.visited:
                        # Mark neighborhood point as visited
                        neighborhood_p.visited = True

                        # Find all neighborhood points of given neighborhood point
                        neighbor_pts_of_given_p = self._get_points_of_neighborhood_radius(
                            neighborhood_p
                        )

                        if len(neighbor_pts_of_given_p) >= self._min_pts:
                            # Continuously extend points with neighbor points
                            neighborhood_pts.extend(neighbor_pts_of_given_p)

                        # Add a point to cluster if is not a member of another cluster
                        is_member_of_another_cluster = False
                        for cluster in self.clusters:
                            for cluster_p in cluster:
                                if cluster_p.id == neighborhood_p.id:
                                    is_member_of_another_cluster = True
                                    break
                        if not is_member_of_another_cluster:
                            new_cluster.add(neighborhood_p)

                # Adding a new cluster into cluster list
                self.clusters.append(new_cluster)
            else:
                self.noises.add(p)
                # P is noise
                continue

        # Adjusting outlier points into cluster
        adjusted_count = self._adjust()

        print('{} {}'.format('Outlier count: ', len(self.noises)))
        print('{} {}'.format('Adjusted count: ', adjusted_count))
        print('{} {}'.format('Remain count: ', len(self.noises) - adjusted_count))

        # Sort clusters by count of points
        self._sort_clusters_by_count()

        # Export result
        self._export_result()

    def _export_result(self):
        filename_frags = self._input_filename.split('.')

        for idx, cluster in enumerate(self.clusters):
            # Exclude clusters which exceeds given cluster count
            if idx >= self._cluster_count:
                break
            # Write result clusters into file
            with open(filename_frags[0] + '_cluster_' + str(idx) + '.txt', 'w') as f:
                for p in cluster:
                    f.write('{}\n'.format(p.id))


if __name__ == '__main__':
    # eps: Maximum radius of the neighborhood
    # min_pts: Minimum number of points in an eps-neighborhood of a given point
    _, input_filename, cluster_count, eps, min_pts = sys.argv
    builder = DBSCANClusterBuilder(input_filename, cluster_count, eps, min_pts)
    builder.run()
