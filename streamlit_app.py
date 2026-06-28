import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from src.section import RCSection
from src.beam import Beam, PushoverAnalysis, PushoverResult

IN_TO_MM = 25.4
FT_TO_MM = 304.8
KSI_TO_MPA = 6.89476
IN2_TO_MM2 = 645.16
MM2_TO_IN2 = 1 / 645.16
KIPFT_TO_NMM = 14.5939
N_TO_KIP = 1 / 4448.22
MM_TO_IN = 1 / 25.4
NMM_TO_KIPFT = 1 / 1.355819e6


def fig_ax(figsize=(5, 4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return fig, ax


def plot_pushover(result: PushoverResult):
    fig, ax = fig_ax()
    ax.plot(result.displacements * MM_TO_IN,
            np.abs(result.axial_forces) * N_TO_KIP, 'b-', linewidth=2)
    if result.failure_index >= 0 and result.failure_index < len(result.displacements):
        idx = result.failure_index
        ax.plot(result.displacements[idx] * MM_TO_IN,
                abs(result.axial_forces[idx]) * N_TO_KIP,
                'ro', markersize=8, label=f'Failure: {result.failure_mode}')
        ax.legend()
    ax.set_xlabel('Axial Displacement (in)')
    ax.set_ylabel('Axial Tension (kip)')
    ax.set_title('Tension Pushover Curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_moment_curvature(section: RCSection, N: float = 0.0):
    fig, ax = fig_ax()
    phis, Ms, _, _, _ = section.moment_curvature(N)
    ax.plot(phis * 25.4, Ms * NMM_TO_KIPFT, 'b-', linewidth=2)
    ax.set_xlabel('Curvature (1/in)')
    ax.set_ylabel('Moment (kip·ft)')
    label = 'Tension' if N < 0 else 'Compression'
    ax.set_title(f'M-φ at N = {abs(N) * N_TO_KIP:.1f} kip ({label})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pm_interaction(section: RCSection):
    fig, ax = fig_ax()
    P_vals, M_vals = section.pm_interaction()
    ax.plot(M_vals * NMM_TO_KIPFT, P_vals * N_TO_KIP, 'r-', linewidth=2)
    ax.set_xlabel('Moment (kip·ft)')
    ax.set_ylabel('Axial Force (kip)')
    ax.set_title('P-M Interaction Diagram')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_midspan_deflection(result: PushoverResult):
    fig, ax = fig_ax()
    ax.plot(result.midspan_deflections * MM_TO_IN,
            abs(result.axial_forces) * N_TO_KIP, 'g-', linewidth=2)
    ax.set_xlabel('Midspan Deflection (in)')
    ax.set_ylabel('Axial Tension (kip)')
    ax.set_title('Axial Tension vs Midspan Deflection')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_moment_diagram(result: PushoverResult):
    fig, ax = fig_ax()
    x_ft = result.x_positions * MM_TO_IN / 12 if result.x_positions is not None else None
    if x_ft is None or not result.moment_profiles:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        return fig

    n_steps = len(result.moment_profiles)
    n_show = min(n_steps, 12)
    idx = np.linspace(0, n_steps - 1, n_show, dtype=int)
    colors = viridis(np.linspace(0.2, 0.9, n_show))

    for i, step_i in enumerate(idx):
        M = np.array(result.moment_profiles[step_i]) * NMM_TO_KIPFT
        label = f'Step {step_i} ({abs(result.axial_forces[step_i]) * N_TO_KIP:.0f} kip)'
        ax.plot(x_ft, M, color=colors[i], linewidth=1.5, label=label)

    ax.set_xlabel('Position along beam (ft)')
    ax.set_ylabel('Total Moment M_sw + P·v (kip·ft)')
    ax.set_title('Moment Distribution (incl. P-Δ effect)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_axial_strain_diagram(result: PushoverResult):
    fig, ax = fig_ax()
    x_ft = result.x_positions * MM_TO_IN / 12 if result.x_positions is not None else None
    if x_ft is None or not result.strain_profiles:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        return fig

    n_steps = len(result.strain_profiles)
    n_show = min(n_steps, 12)
    idx = np.linspace(0, n_steps - 1, n_show, dtype=int)
    colors = viridis(np.linspace(0.2, 0.9, n_show))

    for i, step_i in enumerate(idx):
        eps = np.array(result.strain_profiles[step_i]) * 1e6
        label = f'Step {step_i} ({abs(result.axial_forces[step_i]) * N_TO_KIP:.0f} kip)'
        ax.plot(x_ft, eps, color=colors[i], linewidth=1.5, label=label)

    ax.set_xlabel('Position along beam (ft)')
    ax.set_ylabel('Mid-height Strain (με)')
    ax.set_title('Axial Strain Distribution Along Beam')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_stiffness(result: PushoverResult):
    fig, ax = fig_ax()
    d = result.displacements * MM_TO_IN
    P = abs(result.axial_forces) * N_TO_KIP
    if len(d) < 3:
        ax.text(0.5, 0.5, 'Not enough points', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        return fig

    stiff = np.zeros(len(d))
    stiff[0] = (P[1] - P[0]) / (d[1] - d[0]) if d[1] != d[0] else 0
    for i in range(1, len(d) - 1):
        stiff[i] = (P[i + 1] - P[i - 1]) / (d[i + 1] - d[i - 1])
    stiff[-1] = (P[-1] - P[-2]) / (d[-1] - d[-2]) if d[-1] != d[-2] else 0
    stiff = np.maximum(stiff, 0)

    ax.plot(d, stiff, 'r-', linewidth=2, label='Tangent stiffness')
    ax.fill_between(d, stiff, alpha=0.15, color='red')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Axial Displacement (in)')
    ax.set_ylabel('Tangent Stiffness (kip/in)')
    ax.set_title('Axial Stiffness Along Pushover')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_section(section: RCSection):
    fig, ax = fig_ax()
    b, h = section.b * MM_TO_IN, section.h * MM_TO_IN
    ax.add_patch(plt.Rectangle((-b / 2, 0), b, h, fill=False,
                               edgecolor='black', linewidth=2))
    if section.As_top > 0:
        ax.plot(0, section.d_top * MM_TO_IN, 'ro', markersize=8,
                label=f'Top {section.As_top * MM2_TO_IN2:.2f} in²')
    if section.As_bot > 0:
        ax.plot(0, (h - section.d_bot * MM_TO_IN), 'bs', markersize=8,
                label=f'Bot {section.As_bot * MM2_TO_IN2:.2f} in²')
    ax.set_xlim(-b, b)
    ax.set_ylim(-h * 0.1, h * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('Width (in)')
    ax.set_ylabel('Depth (in)')
    ax.set_title('Cross Section')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def format_results_text(r: PushoverResult, sec: RCSection, L_ft: float, w_kipft: float):
    lines = [
        "=" * 60,
        "  PUSHOVER ANALYSIS RESULTS",
        "=" * 60,
        "",
        f"  Beam: {sec.b * MM_TO_IN:.1f} x {sec.h * MM_TO_IN:.1f} in, Span = {L_ft:.1f} ft",
        f"  Concrete: fc' = {sec.concrete.fc / KSI_TO_MPA:.1f} ksi",
        f"  Steel: fy = {sec.steel.fy / KSI_TO_MPA:.0f} ksi",
        f"  Top reinforcement: {sec.As_top * MM2_TO_IN2:.2f} in² at d' = {sec.d_top * MM_TO_IN:.1f} in",
        f"  Bottom reinforcement: {sec.As_bot * MM2_TO_IN2:.2f} in² at d = {(sec.h - sec.d_bot) * MM_TO_IN:.1f} in (cover = {sec.d_bot * MM_TO_IN:.1f} in)",
        f"  Distributed load: {w_kipft:.1f} kip/ft",
        "",
        "-" * 60,
        "  PUSHOVER RESULT SUMMARY",
        "-" * 60,
        "",
    ]

    if len(r.axial_forces) > 0:
        max_P = np.max(np.abs(r.axial_forces)) * N_TO_KIP
        max_disp = r.displacements[-1] * MM_TO_IN if len(r.displacements) > 0 else 0
        max_M = np.max(np.abs(r.max_moments)) * NMM_TO_KIPFT if len(r.max_moments) > 0 else 0
        max_v = np.max(np.abs(r.midspan_deflections)) * MM_TO_IN if len(r.midspan_deflections) > 0 else 0

        lines += [
            f"  Max axial tension: {max_P:.2f} kip",
            f"  Max axial elongation: {max_disp:.4f} in",
            f"  Max midspan deflection: {max_v:.4f} in",
            f"  Max moment: {max_M:.2f} kip·ft",
            f"  Failure mode: {r.failure_mode}",
            f"  Converged: {r.converged}",
            "",
        ]

    if len(r.axial_forces) > 0:
        lines += [
            "-" * 60,
            "  LOAD-DISPLACEMENT HISTORY (top 20 steps)",
            "-" * 60,
            f"  {'Step':>5} {'Elong (in)':>10} {'Tension (kip)':>12} {'Mid V (in)':>10}",
            "  " + "-" * 42,
        ]
        step = min(len(r.axial_forces), 20)
        idx = np.linspace(0, len(r.axial_forces) - 1, step, dtype=int)
        for i in idx:
            lines.append(
                f"  {i:>5} {r.displacements[i] * MM_TO_IN:>10.4f} "
                f"{abs(r.axial_forces[i]) * N_TO_KIP:>12.2f} "
                f"{r.midspan_deflections[i] * MM_TO_IN:>10.4f}"
            )

    lines += [
        "",
        "=" * 60,
        "  ANALYSIS COMPLETE",
        "=" * 60,
    ]
    return '\n'.join(lines)


def run_analysis(b, h, L, fc, fy, Es, ecu, As_top, d_top, As_bot, d_bot, w_dist, n_seg, n_steps):
    sec = RCSection(b * IN_TO_MM, h * IN_TO_MM, d_top * IN_TO_MM, d_bot * IN_TO_MM,
                    As_top * IN2_TO_MM2, As_bot * IN2_TO_MM2,
                    fc * KSI_TO_MPA, fy * KSI_TO_MPA, Es * KSI_TO_MPA,
                    conc_ecu=ecu)
    beam = Beam(sec, L * FT_TO_MM, w_dist * KIPFT_TO_NMM)
    analysis = PushoverAnalysis(beam, n_seg=n_seg, n_steps=n_steps)
    result = analysis.run()
    return sec, result


st.set_page_config(page_title="RC Beam Pushover Analysis", layout="wide")
st.title("RC Beam Pushover Analysis")

with st.sidebar:
    st.header("Beam Geometry")
    b = st.number_input("Width b (in)", value=12.0, format="%.2f")
    h_in = st.number_input("Height h (in)", value=20.0, format="%.2f")
    L = st.number_input("Span L (ft)", value=20.0, format="%.2f")

    st.header("Material Properties")
    fc = st.number_input("Concrete fc' (ksi)", value=4.0, format="%.2f")
    fy = st.number_input("Steel fy (ksi)", value=60.0, format="%.0f")
    Es = st.number_input("Steel Es (ksi)", value=29000.0, format="%.0f")
    ecu = st.number_input("Concrete ecu", value=0.0035, format="%.4f")

    st.header("Top Reinforcement")
    As_top = st.number_input("Top As (in²)", value=0.62, format="%.2f")
    d_top = st.number_input("Top cover d' (in)", value=2.0, format="%.2f")

    st.header("Bottom Reinforcement")
    As_bot = st.number_input("Bottom As (in²)", value=1.24, format="%.2f")
    d_bot = st.number_input("Bottom cover (in)", value=2.0, format="%.2f")

    st.header("Loading (constant)")
    w_dist = st.number_input("Distributed load w (kip/ft)", value=1.4, format="%.2f")

    st.header("Analysis Settings")
    n_seg = st.number_input("Beam segments", value=50, step=10)
    n_steps = st.number_input("Pushover steps", value=100, step=10)

    run = st.button("Run Analysis", type="primary", use_container_width=True)

if run:
    if b <= 0 or h_in <= 0 or L <= 0:
        st.error("Dimensions must be positive")
    elif fc <= 0 or fy <= 0:
        st.error("Material strengths must be positive")
    else:
        with st.spinner("Running analysis..."):
            try:
                sec, result = run_analysis(b, h_in, L, fc, fy, Es, ecu,
                                           As_top, d_top, As_bot, d_bot,
                                           w_dist, n_seg, n_steps)
                st.success("Analysis complete")

                tabs = st.tabs(["Pushover", "M-φ Curve", "P-M Diagram",
                                "M Diagram", "Strain Diagram", "Stiffness",
                                "Force-Deflection", "Section", "Results"])

                with tabs[0]:
                    st.pyplot(plot_pushover(result))

                with tabs[1]:
                    with st.spinner("Computing M-φ curve..."):
                        try:
                            st.pyplot(plot_moment_curvature(sec))
                        except Exception as e:
                            st.error(f"M-φ curve: {e}")

                with tabs[2]:
                    with st.spinner("Computing P-M diagram..."):
                        try:
                            st.pyplot(plot_pm_interaction(sec))
                        except Exception as e:
                            st.error(f"P-M diagram: {e}")

                with tabs[3]:
                    try:
                        st.pyplot(plot_moment_diagram(result))
                    except Exception as e:
                        st.error(f"Moment diagram: {e}")

                with tabs[4]:
                    try:
                        st.pyplot(plot_axial_strain_diagram(result))
                    except Exception as e:
                        st.error(f"Strain diagram: {e}")

                with tabs[5]:
                    try:
                        st.pyplot(plot_stiffness(result))
                    except Exception as e:
                        st.error(f"Stiffness: {e}")

                with tabs[6]:
                    try:
                        st.pyplot(plot_midspan_deflection(result))
                    except Exception as e:
                        st.error(f"Force-deflection: {e}")

                with tabs[7]:
                    try:
                        st.pyplot(plot_section(sec))
                    except Exception as e:
                        st.error(f"Section: {e}")

                with tabs[8]:
                    text = format_results_text(result, sec, L, w_dist)
                    st.code(text)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
else:
    st.info("Enter parameters in the sidebar and click **Run Analysis**.")
    st.markdown("""
    This app performs nonlinear pushover analysis of a simply supported
    reinforced concrete beam with top and bottom reinforcement layers.

    **Plots:**
    - **Pushover** — Axial tension vs axial displacement
    - **M-φ Curve** — Moment-curvature at zero axial load
    - **P-M Diagram** — Axial force-moment interaction envelope
    - **M Diagram** — Moment distribution along the beam
    - **Strain Diagram** — Mid-height strain along the beam
    - **Stiffness** — Tangent axial stiffness during pushover
    - **Force-Deflection** — Axial tension vs midspan deflection
    - **Section** — Cross-section sketch with reinforcement
    - **Results** — Tabular output
    """)
