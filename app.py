import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_unit_cell(structure):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Cube edges
    for s, e in zip([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0],
                     [0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,1]],
                    [[1,0,0],[1,1,0],[0,1,0],[0,0,0],[0,0,1],
                     [1,0,1],[1,1,1],[0,1,1],[0,0,1],[0,0,0]]):
        ax.plot3D(*zip(s,e), color="gray", lw=1)

    # Atom positions
    if structure == "Simple Cubic":
        points = [(0,0,0)]
    elif structure == "BCC":
        points = [(0,0,0), (0.5,0.5,0.5)]
    elif structure == "FCC":
        points = [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]

    for p in points:
        ax.scatter(*p, color='red', s=100)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.set_box_aspect([1,1,1])
    st.pyplot(fig)

# Streamlit app
st.title("Crystal Structure Visualizer")

structure = st.selectbox("Select a crystal structure", ["Simple Cubic", "BCC", "FCC"])
draw_unit_cell(structure)
