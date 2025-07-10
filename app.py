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
    ["Crystal Structure", "Slip System", "Schmid's Law", "Example", "Tutorials", "Strain Hardening"]
)

# --------------------------
# Crystal Structure
# --------------------------

if option == "Crystal Structure":
    structure = st.selectbox("Select Crystal Structure", ["FCC", "BCC"])
    plot_crystal(structure)


# --------------------------
# Slip System
# --------------------------

elif option == "Slip System":
    structure = st.selectbox("Select Structure", ["FCC", "BCC"])

    if structure == "FCC":
        plane = st.selectbox("Select Slip Plane", list(fcc_slip_planes.keys()))
        direction = st.selectbox("Select Slip Direction", list(fcc_slip_directions[plane].keys()))
        slip_plane_coords = fcc_slip_planes[plane]
        slip_vec = fcc_slip_directions[plane][direction]
    else:
        plane = st.selectbox("Select Slip Plane", list(bcc_slip_planes.keys()))
        direction = st.selectbox("Select Slip Direction", list(bcc_slip_directions[plane].keys()))
        slip_plane_coords = bcc_slip_planes[plane]
        slip_vec = bcc_slip_directions[plane][direction]

    scaled_vec = tuple(0.5 * i for i in slip_vec)
    slip_coords = slip_plane_coords + [(0, 0, 0), scaled_vec]
    plot_crystal(structure, slip_coords=slip_coords)


# --------------------------
# Schmid's Law
# --------------------------

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
        st.success(f"Resolved Shear Stress: {tau:.2f} MPa")

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
        st.success(f"Resolved Shear Stress = {tau:.2f} MPa")

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


# --------------------------
# Example Calculation
# --------------------------

elif option == "Example":
    # -------- Example Calculation --------
    st.title("📚 Example of Schmid Law")

    st.markdown("""
    Let us consider a **BCC crystal**.

    - Slip plane: {110}
    - Slip direction: [1 1 1]
    - Loading direction: [1 0 0]
    - Applied stress σ = 150 MPa
    """)

    sigma = 150.0
    N = np.array([1, 1, 0])
    D = np.array([1, 1, 1])
    L = np.array([1, 0, 0])

    N_unit = N / np.linalg.norm(N)
    D_unit = D / np.linalg.norm(D)
    L_unit = L / np.linalg.norm(L)

    cos_phi = np.dot(L_unit, D_unit)
    cos_lambda = np.dot(L_unit, N_unit)
    phi_deg = np.degrees(np.arccos(cos_phi))
    lambda_deg = np.degrees(np.arccos(cos_lambda))
    tau = sigma * cos_phi * cos_lambda

    st.subheader("Step-by-Step Calculation")
    st.markdown(f"""
    - Slip plane normal (N): {N.tolist()}
    - Slip direction (D): {D.tolist()}
    - Loading direction (L): {L.tolist()}
    - Unit N = {N_unit.round(4).tolist()}
    - Unit D = {D_unit.round(4).tolist()}
    - Unit L = {L_unit.round(4).tolist()}
    - cos(φ) = {cos_phi:.4f}
    - cos(λ) = {cos_lambda:.4f}
    - φ = {phi_deg:.2f}°
    - λ = {lambda_deg:.2f}°

    $$\\tau = \\sigma \\times \\cos\\phi \\times \\cos\\lambda$$

    Thus:

    $$\\tau = {sigma:.1f} \\times {cos_phi:.4f} \\times {cos_lambda:.4f} = {tau:.2f}~\\text{{MPa}}$$
    """)
    st.success(f"✅ Resolved Shear Stress τ = {tau:.2f} MPa")

    # Optional vectors plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0, L_unit[0]], y=[0, L_unit[1]], z=[0, L_unit[2]], mode='lines', name='Loading', line=dict(width=4, color='red')))
    fig.add_trace(go.Scatter3d(x=[0, D_unit[0]], y=[0, D_unit[1]], z=[0, D_unit[2]], mode='lines', name='Slip Dir', line=dict(width=4, color='green')))
    fig.add_trace(go.Scatter3d(x=[0, N_unit[0]], y=[0, N_unit[1]], z=[0, N_unit[2]], mode='lines', name='Plane Normal', line=dict(width=4, color='blue')))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig, use_container_width=True)

    # -------- CRSS Activation Example --------
    st.header("💡 Example: Finding Activated Slip Systems")

    st.markdown("""
    Let’s check which slip systems activate under:
    - Crystal: **FCC**
    - Applied stress σ = 400 MPa
    - Loading direction: [1, 1, 0]
    - CRSS = 100 MPa
    Checking:
    - {111} / [1 0 1]
    - {111} / [1 1 0]
    - {111} / [0 1 1]
    """)

    example_systems = [
        {"plane": "{111}", "direction": "[1 0 1]"},
        {"plane": "{111}", "direction": "[1 1 0]"},
        {"plane": "{111}", "direction": "[0 1 1]"}
    ]

    sigma_val = 400.0
    L = np.array([1, 1, 0])
    crss_value = 100.0

    activated_systems = []
    results_text = []

    for s in example_systems:
        plane = s["plane"]
        direction = s["direction"]
        D = np.array(fcc_slip_directions[plane][direction])
        N = np.array([1, 1, 1])
        D_unit = D / np.linalg.norm(D)
        N_unit = N / np.linalg.norm(N)
        L_unit = L / np.linalg.norm(L)
        cos_phi = np.dot(L_unit, D_unit)
        cos_lambda = np.dot(L_unit, N_unit)
        tau = sigma_val * cos_phi * cos_lambda

        if abs(tau) >= crss_value:
            activated_systems.append(f"{plane} / {direction}")
            status = "ACTIVATED"
        else:
            status = "NOT activated"

        results_text.append(f"{plane} / {direction} → τ = {tau:.2f} MPa → {status}")

    st.subheader("Step-by-Step Results")
    for line in results_text:
        st.write(f"- {line}")

    if activated_systems:
        st.success("✅ Activated Slip Systems:\n" + "\n".join(activated_systems))
    else:
        st.info("❌ No slip systems activated under this loading.")


# --------------------------
# Tutorials
# --------------------------

elif option == "Tutorials":
    st.title("🎓 Tutorials")

    st.markdown("""
    Practice computing the resolved shear stress!

    These tutorials:
    - show slip systems from FCC and BCC
    - let you solve τ on your own
    - check whether slip systems activate above a given CRSS
    Click "Show Answer" only when you're ready!
    """)

    # Tutorial 1
    st.header("Tutorial 1: Single FCC Example")

    t1_plane = "{111}"
    t1_direction = "[1 1 0]"
    t1_loading = np.array([1, 1, 1])
    t1_sigma = 250.0

    D = np.array(fcc_slip_directions[t1_plane][t1_direction])
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

    **Compute the resolved shear stress τ.**
    """)

    if st.button("Show Answer for Tutorial 1"):
        st.success(f"✅ τ = {tau_t1:.2f} MPa")

    # Tutorial 2
    st.header("Tutorial 2: Multiple Systems")

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

    for i, ex in enumerate(tutorial_examples, 1):
        structure = ex["structure"]
        plane = ex["plane"]
        direction = ex["direction"]
        loading = ex["loading"]
        sigma = ex["sigma"]

        if structure == "FCC":
            D = np.array(fcc_slip_directions[plane][direction])
            N = np.array([1, 1, 1])
        else:
            D = np.array(bcc_slip_directions[plane][direction])
            N = np.array(bcc_slip_planes[plane][1]) - np.array(bcc_slip_planes[plane][0])
            if np.linalg.norm(N) == 0:
                N = np.array([1, 1, 0])

        D_unit = D / np.linalg.norm(D)
        N_unit = N / np.linalg.norm(N)
        L_unit = loading / np.linalg.norm(loading)

        cos_phi = np.dot(L_unit, D_unit)
        cos_lambda = np.dot(L_unit, N_unit)
        tau = sigma * cos_phi * cos_lambda

        st.markdown(f"""
        **System {i}:**
        - Crystal: {structure}
        - Slip plane: {plane}
        - Slip direction: {direction}
        - Loading direction: {loading.tolist()}
        - Applied stress σ = {sigma} MPa

        **Compute the resolved shear stress τ.**
        """)

        if st.button(f"Show Answer for System {i}"):
            st.success(f"✅ τ = {tau:.2f} MPa")

    # Tutorial 3
    st.header("Tutorial 3: CRSS Activation Check")

    crss_value = 100.0

    st.markdown(f"""
    Let's assume a critical resolved shear stress (CRSS) of **{crss_value} MPa**.
    Check which slip systems below will be activated (τ ≥ CRSS).
    """)

    systems_t3 = [
        {
            "structure": "FCC",
            "plane": "{111}",
            "direction": "[1 0 1]",
            "loading": np.array([1, 1, 0]),
            "sigma": 400.0
        },
        {
            "structure": "BCC",
            "plane": "{110}",
            "direction": "[1 1 1]",
            "loading": np.array([0, 0, 1]),
            "sigma": 150.0
        },
        {
            "structure": "BCC",
            "plane": "{112}",
            "direction": "[1 -1 1]",
            "loading": np.array([1, 0, 0]),
            "sigma": 500.0
        }
    ]

    activated_systems = []
    results_text = []

    for s in systems_t3:
        struct = s["structure"]
        plane = s["plane"]
        direction = s["direction"]
        loading = s["loading"]
        sigma = s["sigma"]

        if struct == "FCC":
            D = np.array(fcc_slip_directions[plane][direction])
            N = np.array([1, 1, 1])
        else:
            D = np.array(bcc_slip_directions[plane][direction])
            N = np.array(bcc_slip_planes[plane][1]) - np.array(bcc_slip_planes[plane][0])
            if np.linalg.norm(N) == 0:
                N = np.array([1, 1, 0])

        D_unit = D / np.linalg.norm(D)
        N_unit = N / np.linalg.norm(N)
        L_unit = loading / np.linalg.norm(loading)
        cos_phi = np.dot(L_unit, D_unit)
        cos_lambda = np.dot(L_unit, N_unit)
        tau = sigma * cos_phi * cos_lambda

        desc = f"{struct} - Plane {plane}, Dir {direction}"
        if abs(tau) >= crss_value:
            activated_systems.append(desc)
            status = "ACTIVATED"
        else:
            status = "NOT activated"

        results_text.append(f"{desc} → τ = {tau:.2f} MPa → {status}")

    st.markdown("""
    The systems to evaluate:
    """)
    for s in systems_t3:
        st.markdown(f"""
        - Crystal: **{s['structure']}**
        - Slip plane: **{s['plane']}**
        - Slip direction: **{s['direction']}**
        - Loading direction: **{s['loading'].tolist()}**
        - Applied stress σ = **{s['sigma']} MPa**
        """)

    if st.button("Show Answer for Tutorial 3"):
        st.success("✅ Here are the activation results:")
        for line in results_text:
            st.write(f"- {line}")
elif option == "Strain Hardening":
    st.title("💪 Strain Hardening")

    st.markdown("""
    When metals deform plastically, dislocations move on slip systems.

    **Strain hardening** happens because as more dislocations form,
    they block each other—making it harder for them to move.

    Think of it like **car traffic**:
    - Cars = dislocations
    - Lanes = slip planes
    - Speed = dislocation velocity
    - Traffic jam = dislocation forest
    - Needing more engine power = higher CRSS

    The more dislocations there are, the higher the critical resolved shear stress (CRSS) becomes.
    """)

    st.subheader("🚗 Traffic Analogy")

    # Traffic photo
    st.image("traffic.jpg", caption="Few cars = fast dislocation motion")

    st.markdown("""
    Imagine a highway:

    - Few cars → fast speed (low CRSS)
    - Lots of cars → traffic jam → cars block each other → need more power to move → high CRSS

    Same in metals:
    - Few dislocations → easy slip → low CRSS
    - Many dislocations → blocked paths → higher CRSS
    """)

    # Dislocation tangle photo
    st.image("dislocations.jpeg", caption="Tangled dislocations create obstacles")

    st.info("Let's calculate strain hardening for a slip system:")

    structure = st.selectbox("Select Crystal Structure", ["FCC", "BCC"])
    if structure == "FCC":
        plane = st.selectbox("Select Slip Plane", list(fcc_slip_planes.keys()))
        direction = st.selectbox("Select Slip Direction", list(fcc_slip_directions[plane].keys()))
    else:
        plane = st.selectbox("Select Slip Plane", list(bcc_slip_planes.keys()))
        direction = st.selectbox("Select Slip Direction", list(bcc_slip_directions[plane].keys()))

    tau = st.number_input("Resolved Shear Stress τ (MPa)", value=150.0)
    rho_0 = st.number_input("Initial Dislocation Density ρ₀ (1/m²)", value=1e12, format="%.2e")
    delta_rho = st.number_input("Increase in Dislocation Density Δρ (1/m²)", value=2e13, format="%.2e")
    alpha = st.number_input("Material Constant α", value=0.5)
    G = st.number_input("Shear Modulus G (MPa)", value=26000.0)
    b = st.number_input("Burgers Vector b (meters)", value=2.5e-10, format="%.2e")

    # Calculate new CRSS
    rho_new = rho_0 + delta_rho
    tau_new = tau + alpha * G * b * np.sqrt(rho_new)

    st.markdown(f"""
    ### 🔎 Results:

    - **New Dislocation Density ρ:** {rho_new:.2e} 1/m²
    - **New CRSS:** {tau_new:.2f} MPa
    """)

    st.success(f"✅ The metal is stronger after strain hardening! New CRSS = {tau_new:.2f} MPa")
