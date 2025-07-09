import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ======= PLOT FUNCTION =======

def plot_crystal(structure, slip_coords=None):
    if structure == "BCC":
        points = [(0,0,0), (1,1,0), (1,0,1), (0,1,1), (1,0,0), (0,1,0), (0,0,1), (1,1,1), (0.5,0.5,0.5)]
    elif structure == "FCC":
        points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1),
                  (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (0.5,0.5,1), (0.5,1,0.5), (1,0.5,0.5)]
    else:
        points = []

    if not points:
        st.warning("No points to plot.")
        return

    x, y, z = zip(*points)
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=6, color='red'),
        name='Atoms'
    ))

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
        scene=dict(
            xaxis=dict(range=[0, 2], title='X'),
            yaxis=dict(range=[0, 2], title='Y'),
            zaxis=dict(range=[0, 2], title='Z'),
        ),
        margin=dict(l=10, r=10, b=10, t=30),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


# ======= SLIP SYSTEM DATA =======

fcc_slip_planes = {
    "{111}": [(0,0,0), (1,0,0), (0.5,0.5,np.sqrt(2)/2)]
}

fcc_slip_directions = {
    "{111}": {
        "[1 1 0]": (1,1,0),
        "[-1 1 0]": (-1,1,0),
        "[1 0 1]": (1,0,1),
        "[-1 0 1]": (-1,0,1),
        "[0 1 1]": (0,1,1),
        "[0 -1 1]": (0,-1,1),
        "[1 -1 0]": (1,-1,0),
        "[-1 -1 0]": (-1,-1,0),
        "[1 0 -1]": (1,0,-1),
        "[-1 0 -1]": (-1,0,-1),
        "[0 1 -1]": (0,1,-1),
        "[0 -1 -1]": (0,-1,-1)
    }
}

bcc_slip_planes = {
    "{110}": [(0,0,0), (1,0,0), (1,1,0), (0,1,0)],
    "{112}": [(0,0,0), (1,1,0), (0,0,1)],
    "{123}": [(0,0,0), (1,1,1), (1,0,1)]
}

bcc_slip_directions = {
    "{110}": {
        "[1 1 1]": (1,1,1), "[-1 1 1]": (-1,1,1), "[1 -1 1]": (1,-1,1), "[1 1 -1]": (1,1,-1),
        "[-1 -1 1]": (-1,-1,1), "[1 -1 -1]": (1,-1,-1), "[-1 1 -1]": (-1,1,-1), "[-1 -1 -1]": (-1,-1,-1)
    },
    "{112}": {
        "[1 1 1]": (1,1,1), "[-1 1 1]": (-1,1,1), "[1 -1 1]": (1,-1,1), "[1 1 -1]": (1,1,-1),
        "[-1 -1 1]": (-1,-1,1), "[1 -1 -1]": (1,-1,-1), "[-1 1 -1]": (-1,1,-1), "[-1 -1 -1]": (-1,-1,-1)
    },
    "{123}": {
        "[1 1 1]": (1,1,1), "[-1 1 1]": (-1,1,1), "[1 -1 1]": (1,-1,1), "[1 1 -1]": (1,1,-1),
        "[-1 -1 1]": (-1,-1,1), "[1 -1 -1]": (1,-1,-1), "[-1 1 -1]": (-1,1,-1), "[-1 -1 -1]": (-1,-1,-1)
    }
}


# ======= STREAMLIT APP =======

st.title("🔬 Crystal Plasticity Visualizer")

option = st.sidebar.radio(
    "Select Module",
    ["Crystal Structure", "Slip System", "Schmid's Law", "Example", "Tutorials"]
)



if option == "Crystal Structure":
    structure = st.selectbox("Select Crystal Structure", ["FCC", "BCC"])
    plot_crystal(structure)

elif option == "Slip System":
    structure = st.selectbox("Select Structure", ["FCC", "BCC"])

    if structure == "FCC":
        plane = st.selectbox("Select Slip Plane", list(fcc_slip_planes.keys()))
        direction = st.selectbox("Select Slip Direction", list(fcc_slip_directions[plane].keys()))
        slip_plane_coords = fcc_slip_planes[plane]
        slip_vec = fcc_slip_directions[plane][direction]

    else:  # BCC
        plane = st.selectbox("Select Slip Plane", list(bcc_slip_planes.keys()))
        direction = st.selectbox("Select Slip Direction", list(bcc_slip_directions[plane].keys()))
        slip_plane_coords = bcc_slip_planes[plane]
        slip_vec = bcc_slip_directions[plane][direction]

    scaled_vec = tuple(0.5 * i for i in slip_vec)
    slip_coords = slip_plane_coords + [(0, 0, 0), scaled_vec]
    plot_crystal(structure, slip_coords=slip_coords)

elif option == "Schmid's Law":
    st.subheader("Schmid’s Law Visualizer")
    mode = st.radio("Mode", ["Angle Based", "3D Vector Based"])
    
    def visualize_schmid():
        phi = st.slider("Angle φ", 0, 90, 30)
        lam = st.slider("Angle λ", 0, 90, 60)
        sigma = st.number_input("Applied Stress σ", value=1000.0)
        phi_rad = np.radians(phi)
        lam_rad = np.radians(lam)
        tau = sigma * np.cos(phi_rad) * np.cos(lam_rad)
        st.latex(r"\tau = \sigma \cdot \cos\phi \cdot \cos\lambda")
        st.success(f"Resolved Shear Stress: {tau:.2f} N")

    def visualize_schmid_3d():
        sigma = st.number_input("Applied Stress σ", value=1000.0)
        L = np.array([float(x) for x in st.text_input("Loading Direction", "1,0,0").split(",")])
        N = np.array([float(x) for x in st.text_input("Plane Normal", "1,1,1").split(",")])
        D = np.array([float(x) for x in st.text_input("Slip Direction", "0,1,1").split(",")])
        L, N, D = L/np.linalg.norm(L), N/np.linalg.norm(N), D/np.linalg.norm(D)
        cos_phi, cos_lambda = np.dot(L, D), np.dot(L, N)
        tau = sigma * cos_phi * cos_lambda
        st.latex(r"\tau = \sigma \cdot \cos\phi \cdot \cos\lambda")
        st.write(f"φ = {np.degrees(np.arccos(cos_phi)):.2f}°, λ = {np.degrees(np.arccos(cos_lambda)):.2f}°")
        st.success(f"Resolved Shear Stress = {tau:.2f} N")

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[0, L[0]], y=[0, L[1]], z=[0, L[2]], mode='lines', name='Loading', line=dict(width=4, color='red')))
        fig.add_trace(go.Scatter3d(x=[0, D[0]], y=[0, D[1]], z=[0, D[2]], mode='lines', name='Slip Dir', line=dict(width=4, color='green')))
        fig.add_trace(go.Scatter3d(x=[0, N[0]], y=[0, N[1]], z=[0, N[2]], mode='lines', name='Plane Norm', line=dict(width=4, color='blue')))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        st.plotly_chart(fig, use_container_width=True)

    if mode == "Angle Based":
        visualize_schmid()
    else:
        visualize_schmid_3d()
elif option == "Example":
    st.title("📚 Example of Schmid Law")

    st.markdown("""
    Let us consider a **BCC crystal**.
    
    - Chosen slip plane: {110}
    - Slip direction: [1 1 1]
    - Loading direction: [1 0 0]
    - Applied stress σ = 150 MPa
    """)

    sigma = 150.0  # MPa

    # Slip plane normal for {110}
    N = np.array([1, 1, 0])
    # Slip direction
    D = np.array([1, 1, 1])
    # Loading direction
    L = np.array([1, 0, 0])

    # Normalize vectors
    N_unit = N / np.linalg.norm(N)
    D_unit = D / np.linalg.norm(D)
    L_unit = L / np.linalg.norm(L)

    # Compute angles
    cos_phi = np.dot(L_unit, D_unit)
    cos_lambda = np.dot(L_unit, N_unit)
    phi_deg = np.degrees(np.arccos(cos_phi))
    lambda_deg = np.degrees(np.arccos(cos_lambda))

    # Schmid factor and shear stress
    tau = sigma * cos_phi * cos_lambda

    st.subheader("Step-by-Step Calculation")

    st.markdown(f"""
    - **Slip plane normal (N):** {N.tolist()}
    - **Slip direction (D):** {D.tolist()}
    - **Loading direction (L):** {L.tolist()}

    Normalize the vectors:

    - Unit N = {N_unit.round(4).tolist()}
    - Unit D = {D_unit.round(4).tolist()}
    - Unit L = {L_unit.round(4).tolist()}

    Compute:

    - cos(φ) = L ⋅ D = {cos_phi:.4f}
    - cos(λ) = L ⋅ N = {cos_lambda:.4f}
    - φ = {phi_deg:.2f}°
    - λ = {lambda_deg:.2f}°

    Calculate resolved shear stress:

    $$
    \\tau = \\sigma \\times \\cos\\phi \\times \\cos\\lambda
    $$

    Thus:

    $$
    \\tau = {sigma:.1f} \\times {cos_phi:.4f} \\times {cos_lambda:.4f} = {tau:.2f} \\text{{ MPa}}
    $$
    """)

    st.success(f"✅ **Resolved Shear Stress τ = {tau:.2f} MPa**")

    # Optional vector plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0, L_unit[0]], y=[0, L_unit[1]], z=[0, L_unit[2]],
        mode='lines', name='Loading Dir', line=dict(width=4, color='red')
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, D_unit[0]], y=[0, D_unit[1]], z=[0, D_unit[2]],
        mode='lines', name='Slip Dir', line=dict(width=4, color='green')
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, N_unit[0]], y=[0, N_unit[1]], z=[0, N_unit[2]],
        mode='lines', name='Plane Normal', line=dict(width=4, color='blue')
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        )
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Tutorials":
    st.title("🎓 Tutorials")

    st.markdown("""
    Practice computing the resolved shear stress!
    
    In these tutorials, you’ll get:
    - random slip systems (FCC or BCC)
    - loading directions
    - applied stress values
    
    Click “Show Answer” to check the τ value. No steps shown.
    """)

    # ------- Tutorial 1 -------
    st.header("Tutorial 1: Single FCC Example")

    # Fixed FCC tutorial example
    t1_plane = "{111}"
    t1_direction = "[1 1 0]"
    t1_loading = np.array([1, 1, 1])
    t1_sigma = 250.0

    # Compute τ
    D = np.array(fcc_slip_directions[t1_plane][t1_direction])
    N = np.cross(D, [0, 0, 1])
    if np.linalg.norm(N) == 0:
        N = np.array([1, 1, 1])
    N_unit = N / np.linalg.norm(N)
    D_unit = D / np.linalg.norm(D)
    L_unit = t1_loading / np.linalg.norm(t1_loading)
    cos_phi = np.dot(L_unit, D_unit)
    cos_lambda = np.dot(L_unit, N_unit)
    tau_t1 = t1_sigma * cos_phi * cos_lambda

    st.markdown(f"""
    - Crystal: **FCC**
    - Slip plane: **{t1_plane}**
    - Slip direction: **{t1_direction}**
    - Loading direction: **{t1_loading.tolist()}**
    - Applied stress σ = {t1_sigma} MPa
    
    Can you compute the resolved shear stress τ?
    """)

    if st.button("Show Answer for Tutorial 1"):
        st.success(f"✅ Resolved Shear Stress τ = {tau_t1:.2f} MPa")


    # ------- Tutorial 2 -------
    st.header("Tutorial 2: Multiple Systems")

    st.markdown("Below are **3 slip systems** (mix of FCC and BCC). Try calculating τ yourself!")

    # Prepare list of examples
    tutorial_examples = [
        {
            "structure": "FCC",
            "plane": "{111}",
            "direction": "[1 0 1]",
            "loading": np.array([1, 1, 0]),
            "sigma": 300.0
        },
        {
            "structure": "BCC",
            "plane": "{110}",
            "direction": "[1 1 1]",
            "loading": np.array([0, 0, 1]),
            "sigma": 200.0
        },
        {
            "structure": "BCC",
            "plane": "{112}",
            "direction": "[1 -1 1]",
            "loading": np.array([1, 0, 0]),
            "sigma": 100.0
        }
    ]

    answers = []
    for example in tutorial_examples:
        structure = example["structure"]
        plane = example["plane"]
        direction = example["direction"]
        loading = example["loading"]
        sigma = example["sigma"]

        # Slip direction
        if structure == "FCC":
            D = np.array(fcc_slip_directions[plane][direction])
            # Generate dummy plane normal (not strictly accurate but fine for tutorial)
            N = np.cross(D, [0, 0, 1])
            if np.linalg.norm(N) == 0:
                N = np.array([1, 1, 1])
        else:  # BCC
            D = np.array(bcc_slip_directions[plane][direction])
            N = np.array(bcc_slip_planes[plane][1]) - np.array(bcc_slip_planes[plane][0])
            if np.linalg.norm(N) == 0:
                N = np.array([1, 1, 0])

        # Normalize
        D_unit = D / np.linalg.norm(D)
        N_unit = N / np.linalg.norm(N)
        L_unit = loading / np.linalg.norm(loading)

        cos_phi = np.dot(L_unit, D_unit)
        cos_lambda = np.dot(L_unit, N_unit)
        tau = sigma * cos_phi * cos_lambda

        answers.append({
            "system": f"{structure} - Plane {plane}, Dir {direction}",
            "tau": tau
        })

    for i, ans in enumerate(answers, 1):
        st.markdown(f"""
        **System {i}:**
        - Crystal: {ans['system'].split(' - ')[0]}
        - Slip Plane and Direction: {ans['system'].split(' - ')[1]}
        """)
        if st.button(f"Show Answer for System {i}"):
            st.success(f"✅ Resolved Shear Stress τ = {ans['tau']:.2f} MPa")

