import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def draw_unit_cells_with_atoms(ax, points, min_coord=0, max_coord=2):
    base_cube_points = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]

    for i in range(min_coord, max_coord):
        for j in range(min_coord, max_coord):
            for k in range(min_coord, max_coord):
                atom_in_cell = any(
                    i <= x < i + 1 and j <= y < j + 1 and k <= z < k + 1
                    for x, y, z in points
                )
                if atom_in_cell:
                    cube_points = [(x + i, y + j, z + k) for x, y, z in base_cube_points]
                    faces = [
                        [cube_points[0], cube_points[1], cube_points[5], cube_points[4]],
                        [cube_points[2], cube_points[3], cube_points[7], cube_points[6]],
                        [cube_points[0], cube_points[4], cube_points[7], cube_points[3]],
                        [cube_points[1], cube_points[5], cube_points[6], cube_points[2]],
                        [cube_points[0], cube_points[1], cube_points[2], cube_points[3]],
                        [cube_points[4], cube_points[5], cube_points[6], cube_points[7]],
                    ]
                    face_colors = 'lightblue'
                    for face in faces:
                        poly = Poly3DCollection([face], alpha=0.5, edgecolor='black', linewidths=0.5, facecolor=face_colors)
                        ax.add_collection3d(poly)

def plot_crystal_structure(ax, structure):
    if structure == "BCC":
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
