import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import viridis
from .section import RCSection
from .beam import PushoverResult


class Plotter:
    def __init__(self, parent, figsize=(5, 4), dpi=100):
        self.parent = parent
        self.figsize = figsize
        self.dpi = dpi

    def new_figure(self):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        return fig, ax

    def embed_figure(self, fig, frame):
        for w in frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        return canvas

    def plot_pushover(self, frame, result: PushoverResult, title: str = "Tension Pushover Curve"):
        fig, ax = self.new_figure()
        ax.plot(result.displacements / 25.4,
                np.abs(np.array(result.axial_forces)) / 4448.22,
                'b-', linewidth=2)
        if result.failure_index >= 0 and result.failure_index < len(result.displacements):
            idx = result.failure_index
            ax.plot(result.displacements[idx] / 25.4,
                    np.abs(result.axial_forces[idx]) / 4448.22,
                    'ro', markersize=8, label=f'Failure: {result.failure_mode}')
            ax.legend()
        ax.set_xlabel('Axial Displacement (in)')
        ax.set_ylabel('Axial Tension (kip)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)

    def plot_moment_curvature(self, frame, section: RCSection, N: float = 0.0):
        fig, ax = self.new_figure()
        phis, Ms, _, _, _ = section.moment_curvature(N)
        ax.plot(phis * 25.4, np.array(Ms) / 1.355819e6, 'b-', linewidth=2)
        ax.set_xlabel('Curvature (1/in)')
        ax.set_ylabel('Moment (kip·ft)')
        label = 'Tension' if N < 0 else 'Compression'
        ax.set_title(f'M-φ at N = {abs(N) / 4448.22:.1f} kip ({label})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)

    def plot_pm_interaction(self, frame, section: RCSection):
        fig, ax = self.new_figure()
        P_vals, M_vals = section.pm_interaction()
        ax.plot(np.array(M_vals) / 1.355819e6, np.array(P_vals) / 4448.22, 'r-', linewidth=2)
        ax.set_xlabel('Moment (kip·ft)')
        ax.set_ylabel('Axial Force (kip)')
        ax.set_title('P-M Interaction Diagram')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)

    def plot_midspan_deflection(self, frame, result: PushoverResult):
        fig, ax = self.new_figure()
        ax.plot(result.midspan_deflections / 25.4,
                np.abs(np.array(result.axial_forces)) / 4448.22,
                'g-', linewidth=2)
        ax.set_xlabel('Midspan Deflection (in)')
        ax.set_ylabel('Axial Tension (kip)')
        ax.set_title('Axial Tension vs Midspan Deflection')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)

    def plot_moment_diagram(self, frame, result: PushoverResult):
        fig, ax = self.new_figure()
        x_ft = result.x_positions / 304.8 if result.x_positions is not None else None
        if x_ft is None or not result.moment_profiles:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
            fig.tight_layout()
            return self.embed_figure(fig, frame)

        n_steps = len(result.moment_profiles)
        n_show = min(n_steps, 12)
        idx = np.linspace(0, n_steps - 1, n_show, dtype=int)
        colors = viridis(np.linspace(0.2, 0.9, n_show))

        for i, step_i in enumerate(idx):
            M = np.array(result.moment_profiles[step_i]) / 1.355819e6
            label = f'Step {step_i} ({abs(result.axial_forces[step_i]) / 4448.22:.0f} kip)'
            ax.plot(x_ft, M, color=colors[i], linewidth=1.5, label=label)

        ax.set_xlabel('Position along beam (ft)')
        ax.set_ylabel('Total Moment M_sw + P·v (kip·ft)')
        ax.set_title('Moment Distribution (incl. P-Δ effect)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)

    def plot_axial_strain_diagram(self, frame, result: PushoverResult):
        fig, ax = self.new_figure()
        x_ft = result.x_positions / 304.8 if result.x_positions is not None else None
        if x_ft is None or not result.strain_profiles:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
            fig.tight_layout()
            return self.embed_figure(fig, frame)

        n_steps = len(result.strain_profiles)
        n_show = min(n_steps, 12)
        idx = np.linspace(0, n_steps - 1, n_show, dtype=int)
        colors = viridis(np.linspace(0.2, 0.9, n_show))

        for i, step_i in enumerate(idx):
            eps = np.array(result.strain_profiles[step_i]) * 1e6
            label = f'Step {step_i} ({abs(result.axial_forces[step_i]) / 4448.22:.0f} kip)'
            ax.plot(x_ft, eps, color=colors[i], linewidth=1.5, label=label)

        ax.set_xlabel('Position along beam (ft)')
        ax.set_ylabel('Mid-height Strain (με)')
        ax.set_title('Axial Strain Distribution Along Beam')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)

    def plot_stiffness(self, frame, result: PushoverResult):
        fig, ax = self.new_figure()
        d = result.displacements / 25.4
        P = np.abs(np.array(result.axial_forces)) / 4448.22
        if len(d) < 3:
            ax.text(0.5, 0.5, 'Not enough points', transform=ax.transAxes, ha='center')
            fig.tight_layout()
            return self.embed_figure(fig, frame)

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
        return self.embed_figure(fig, frame)

    def plot_section(self, frame, section: RCSection):
        fig, ax = self.new_figure()
        b, h = section.b / 25.4, section.h / 25.4
        ax.add_patch(plt.Rectangle((-b / 2, 0), b, h, fill=False,
                                   edgecolor='black', linewidth=2))

        if section.As_top > 0:
            ax.plot(0, section.d_top / 25.4, 'ro', markersize=8,
                    label=f'Top {section.As_top / 645.16:.2f} in²')
        if section.As_bot > 0:
            ax.plot(0, (h - section.d_bot / 25.4), 'bs', markersize=8,
                    label=f'Bot {section.As_bot / 645.16:.2f} in²')

        ax.set_xlim(-b, b)
        ax.set_ylim(-h * 0.1, h * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('Width (in)')
        ax.set_ylabel('Depth (in)')
        ax.set_title('Cross Section')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.embed_figure(fig, frame)
