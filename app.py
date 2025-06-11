import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ========== Crystal Plotting ==========

def draw_unit_cells(ax, points, min_coord=0, max_coord=2):
    base_cube = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    for i in range(min_coord, max_coord):
        for j in range(min_coord, max_coord):
            for k in range(min_coord, max_coord):
                if any(i <= x < i+1 and j <= y < j+1 and k <= z < k+1 for x, y, z in points):
                    cube = [(x+i, y+j, z+k) for x, y, z in base_cube]
                    faces = [
                        [cube[0], cube[1], cube[5], cube[4]],
                        [cube[2], cube[3], cube[7], cube[6]],
                        [cube[0], cube[3], cube[7], cube[4]],
                        [cube[1], cube[2], cube[6], cube[5]],
                        [cube[0], cube[1], cube[2], cube[3]],
                        [cube[4], cube[5], cube[6], cube[7]],
                    ]
                    poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='skyblue', edgecolor='gray')
                    ax.add_collection3d(poly3d)

def plot_crystal(structure, slip_plane=None, slip_coords=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if structure == "BCC":
        points = [(0,0,0), (1,1,0), (1,0,1), (0,1,1), (1,0,0), (0,1,0), (0,0,1), (1,1,1), (0.5,0.5,0.5)]
    elif structure == "FCC":
        points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1),
                  (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (0.5,0.5,1), (0.5,1,0.5), (1,0.5,0.5)]
    elif structure == "SC":
        points = [(i,j,k) for i in range(2) for j in range(2) for k in range(2)]
    elif structure == "HCP":
        a = 1; c = 1.633 * a
        points = [(0, 0, 0), (1, 0, 0), (0.5, np.sqrt(3)/2, 0), 
                  (0, 0, c), (1, 0, c), (0.5, np.sqrt(3)/2, c),
                  (0.5, np.sqrt(3)/6, c/2), (1.5, np.sqrt(3)/6, c/2)]
    else:
        points = []

    draw_unit_cells(ax, points)

    x, y, z = zip(*points)
    ax.scatter(x, y, z, c='r', s=100)

    if slip_plane and slip_coords:
        poly = Poly3DCollection([slip_coords], alpha=0.5, color='blue')
        ax.add_collection3d(poly)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_title(f"{structure} Crystal Structure")
    st.pyplot(fig)

# ========== Schmid's Law ==========

def visualize_schmid():
    st.subheader("Schmid's Law")
    phi = st.slider("Enter angle between loading direction and slip direction (φ)", 0, 90, 30)
    lam = st.slider("Enter angle between loading direction and slip plane normal (λ)", 0, 90, 60)
    sigma = st.number_input("Enter applied stress σ (in N)", value=1000)

    phi_rad = np.radians(phi)
    lam_rad = np.radians(lam)

    schmid = sigma * np.cos(phi_rad) * np.cos(lam_rad)
    st.latex(r"\text{Resolved Shear Stress } = \sigma \cdot \cos\phi \cdot \cos\lambda")
    st.write(f"→ Resolved Shear Stress = {schmid:.2f} N")

# ========== Main App ==========

st.title("Crystal Structure & Slip System Visualizer")

option = st.sidebar.selectbox("Choose What to Visualize", ["Crystal Structure", "Slip System", "Schmid's Law"])

if option == "Crystal Structure":
    structure = st.selectbox("Select Crystal Structure", ["BCC", "FCC", "SC", "HCP"])
    plot_crystal(structure)

elif option == "Slip System":
    structure = st.selectbox("Structure", ["BCC", "FCC"])
    if structure == "BCC":
        slip_planes = {
            "001": [(0,0,1), (1,0,1), (1,1,1), (0,1,1)],
            "011": [(0,0,0), (0,1,0), (0,1,1), (0,0,1)],
            "112": [(1,0,0), (0,1,0), (0,0,0.5)]
        }
    else:  # FCC
        slip_planes = {
            "111": [(0,0,1), (0,1,0), (1,0,0)],
            "1-11": [(1,0,0), (0,-1,0), (0,0,1)]
        }

    plane = st.selectbox("Slip Plane", list(slip_planes.keys()))
    plot_crystal(structure, slip_plane=plane, slip_coords=slip_planes[plane])

elif option == "Schmid's Law":
    visualize_schmid()
