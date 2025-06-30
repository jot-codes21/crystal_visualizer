import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ========== CRYSTAL STRUCTURE PLOT ==========

def plot_crystal(structure, slip_coords=None):
    if structure == "BCC":
        points = [(0,0,0), (1,1,0), (1,0,1), (0,1,1), (1,0,0), (0,1,0), (0,0,1), (1,1,1), (0.5,0.5,0.5)]
    elif structure == "FCC":
        points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1),
                  (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (0.5,0.5,1), (0.5,1,0.5), (1,0.5,0.5)]
    elif structure == "SC":
        points = [(i,j,k) for i in range(2) for j in range(2) for k in range(2)]
    elif structure == "HCP":
        a = 1
        c = 1.633 * a
        points = [(0, 0, 0), (1, 0, 0), (0.5, np.sqrt(3)/2, 0),
                  (0, 0, c), (1, 0, c), (0.5, np.sqrt(3)/2, c),
                  (0.5, np.sqrt(3)/6, c/2), (1.5, np.sqrt(3)/6, c/2)]
    else:
        points = []

    if not points:
        st.warning("No points to plot.")
        return

    x, y, z = zip(*points)
    fig = go.Figure()

    # Atom points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=6, color='red'),
        name='Atoms'
    ))

    # Visualize slip plane + direction if provided
    if slip_coords:
        if len(slip_coords) >= 3:
            xs, ys, zs = zip(*slip_coords[:-2])
            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs,
                color='blue', opacity=0.5,
                name='Slip Plane'
            ))

        if len(slip_coords) >= 2:
            p1, p2 = slip_coords[-2], slip_coords[-1]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines', line=dict(color='green', width=5),
                name='Slip Direction'
            ))

    fig.update_layout(
        title=f"{structure} Crystal Structure (Interactive)",
        scene=dict(
            xaxis=dict(range=[0, 2], backgroundcolor="white", gridcolor="lightgrey", title='X'),
            yaxis=dict(range=[0, 2], backgroundcolor="white", gridcolor="lightgrey", title='Y'),
            zaxis=dict(range=[0, 2], backgroundcolor="white", gridcolor="lightgrey", title='Z'),
            bgcolor="white"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, b=10, t=40),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# ========== SCHMID'S LAW (ANGLE BASED) ==========

def visualize_schmid():
    st.subheader("Schmid's Law - Angle Based")
    phi = st.slider("Angle φ (between loading and slip direction)", 0, 90, 30)
    lam = st.slider("Angle λ (between loading and slip plane normal)", 0, 90, 60)
    sigma = st.number_input("Applied Stress σ (N)", value=1000)

    phi_rad = np.radians(phi)
    lam_rad = np.radians(lam)
    schmid = sigma * np.cos(phi_rad) * np.cos(lam_rad)

    st.latex(r"\tau = \sigma \cdot \cos\phi \cdot \cos\lambda")
    st.write(f"→ Resolved Shear Stress = {schmid:.2f} N")

# ========== SCHMID'S LAW (3D VECTOR) ==========

def visualize_schmid_3d():
    st.subheader("Schmid's Law - 3D Vector Visualization")

    loading_vec = st.text_input("Loading Direction L (comma-separated)", "1,0,0")
    slip_plane_vec = st.text_input("Slip Plane Normal N (comma-separated)", "1,1,1")
    slip_direction_vec = st.text_input("Slip Direction D (comma-separated)", "0,1,1")
    sigma = st.number_input("Applied Stress σ (N)", value=1000)

    try:
        L = np.array([float(x) for x in loading_vec.split(",")])
        N = np.array([float(x) for x in slip_plane_vec.split(",")])
        D = np.array([float(x) for x in slip_direction_vec.split(",")])
    except:
        st.error("Invalid vector input. Use comma-separated numbers.")
        return

    L_norm = L / np.linalg.norm(L)
    N_norm = N / np.linalg.norm(N)
    D_norm = D / np.linalg.norm(D)

    cos_phi = np.dot(L_norm, D_norm)
    cos_lambda = np.dot(L_norm, N_norm)
    schmid = sigma * cos_phi * cos_lambda
    phi = np.degrees(np.arccos(cos_phi))
    lam = np.degrees(np.arccos(cos_lambda))

    st.latex(r"\tau = \sigma \cdot \cos\phi \cdot \cos\lambda")
    st.write(f"φ = {phi:.2f}°, λ = {lam:.2f}°")
    st.success(f"Resolved Shear Stress = {schmid:.2f} N")

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0, L_norm[0]], y=[0, L_norm[1]], z=[0, L_norm[2]],
                               mode='lines+text', line=dict(color='red', width=5), name='Loading'))
    fig.add_trace(go.Scatter3d(x=[0, D_norm[0]], y=[0, D_norm[1]], z=[0, D_norm[2]],
                               mode='lines+text', line=dict(color='green', width=5), name='Slip Direction'))
    fig.add_trace(go.Scatter3d(x=[0, N_norm[0]], y=[0, N_norm[1]], z=[0, N_norm[2]],
                               mode='lines+text', line=dict(color='blue', width=5), name='Plane Normal'))

    fig.update_layout(
        title="3D Vector Representation",
        scene=dict(
            xaxis=dict(range=[-1, 1], backgroundcolor="white", gridcolor="lightgrey", title='X'),
            yaxis=dict(range=[-1, 1], backgroundcolor="white", gridcolor="lightgrey", title='Y'),
            zaxis=dict(range=[-1, 1], backgroundcolor="white", gridcolor="lightgrey", title='Z'),
            bgcolor="white"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# ========== MAIN STREAMLIT APP ==========

st.title("🔬 Crystal Plasticity Visualizer")

option = st.sidebar.radio("Select Module", ["Crystal Structure", "Slip System", "Schmid's Law"])

if option == "Crystal Structure":
    structure = st.selectbox("Select Crystal Structure", ["BCC", "FCC", "SC", "HCP"])
    plot_crystal(structure)

elif option == "Slip System":
    structure = st.selectbox("Select Structure", ["BCC", "FCC"])

    if structure == "BCC":
        slip_planes = {
            "110": [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
        }
        slip_directions = {
            "110": {
                "[111]": (1,1,1),
                "[1-11]": (1,-1,1)
            }
        }
    else:
        slip_planes = {
            "111": [(0,0,0), (1,0,0), (0.5,0.5,np.sqrt(2)/2)]
        }
        slip_directions = {
            "111": {
                "[110]": (1,1,0),
                "[-101]": (-1,0,1)
            }
        }

    plane = st.selectbox("Choose Slip Plane", list(slip_planes.keys()))
    direction = st.selectbox("Choose Slip Direction", list(slip_directions[plane].keys()))

    slip_vec = slip_directions[plane][direction]
    plane_coords = slip_planes[plane]

    # Optional: scale direction to fit in cube
    scaled_vec = tuple(0.5 * i for i in slip_vec)
    slip_coords = plane_coords + [(0, 0, 0), scaled_vec]
    plot_crystal(structure, slip_coords=slip_coords)

    st.markdown(f"**Example Slip System:** Plane {plane}, Direction {direction}")

elif option == "Schmid's Law":
    st.markdown("### Choose Mode")
    mode = st.radio("Mode", ["Angle Based", "3D Vector Based"])
    if mode == "Angle Based":
        visualize_schmid()
    else:
        visualize_schmid_3d()
