import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np

from .section import RCSection
from .beam import Beam, PushoverAnalysis
from .plotting import Plotter


class PushoverApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RC Beam Pushover Analysis")
        self.root.geometry("1200x750")

        self.plotter = Plotter(self.root)
        self.result = None
        self.section = None

        self._build_ui()

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=0)

        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=1)

        self._build_input_panel(left_frame)
        self._build_output_panel(right_paned)

    def _build_input_panel(self, parent):
        canvas = tk.Canvas(parent, width=350)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._bind_mousewheel(canvas)

        f = scroll_frame

        info = ttk.Label(f, text="RC Beam Pushover Analysis",
                         font=('Helvetica', 12, 'bold'))
        info.pack(pady=(10, 5))

        sep = ttk.Separator(f, orient='horizontal')
        sep.pack(fill='x', padx=10, pady=5)

        self._add_section(f, "Beam Geometry")
        self.b = self._add_field(f, "Width b (in):", 12)
        self.h = self._add_field(f, "Height h (in):", 20)
        self.L = self._add_field(f, "Span L (ft):", 20)

        self._add_section(f, "Material Properties")
        self.fc = self._add_field(f, "Concrete fc' (ksi):", 4)
        self.fy = self._add_field(f, "Steel fy (ksi):", 60)
        self.Es = self._add_field(f, "Steel Es (ksi):", 29000)
        self.ecu = self._add_field(f, "Concrete ecu:", 0.0035)

        self._add_section(f, "Top Reinforcement")
        self.As_top = self._add_field(f, "Top As (in²):", 0.62)
        self.d_top = self._add_field(f, "Top cover d' (in):", 2)

        self._add_section(f, "Bottom Reinforcement")
        self.As_bot = self._add_field(f, "Bottom As (in²):", 1.24)
        self.d_bot = self._add_field(f, "Bottom cover (in):", 2)

        self._add_section(f, "Loading (constant)")
        self.w_dist = self._add_field(f, "Distributed load w (kip/ft):", 1.4)

        self._add_section(f, "Analysis Settings")
        self.n_seg = self._add_field(f, "Beam segments:", 50)
        self.n_steps = self._add_field(f, "Pushover steps:", 100)

        self._add_section(f, "Actions")
        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill='x', padx=10, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Run Analysis",
                                  command=self._run_analysis)
        self.run_btn.pack(fill='x', pady=2)

        self.status_lbl = ttk.Label(f, text="Ready", font=('Helvetica', 9))
        self.status_lbl.pack(pady=(0, 10))

    def _bind_mousewheel(self, canvas):
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _add_section(self, parent, title):
        lbl = ttk.Label(parent, text=title, font=('Helvetica', 10, 'bold'))
        lbl.pack(anchor='w', padx=10, pady=(10, 2))

    def _add_field(self, parent, label, default):
        f = ttk.Frame(parent)
        f.pack(fill='x', padx=10, pady=2)
        lbl = ttk.Label(f, text=label, width=22, anchor='w')
        lbl.pack(side='left')
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(f, textvariable=var, width=12)
        ent.pack(side='right')
        return var

    def _build_output_panel(self, parent):
        nb = ttk.Notebook(parent)
        parent.add(nb, weight=1)

        self.plot_frame_pushover = ttk.Frame(nb)
        self.plot_frame_mc = ttk.Frame(nb)
        self.plot_frame_pm = ttk.Frame(nb)
        self.plot_frame_moment_diag = ttk.Frame(nb)
        self.plot_frame_strain_diag = ttk.Frame(nb)
        self.plot_frame_stiffness = ttk.Frame(nb)
        self.plot_frame_deflection = ttk.Frame(nb)
        self.plot_frame_section = ttk.Frame(nb)
        self.results_frame = ttk.Frame(nb)

        nb.add(self.plot_frame_pushover, text="Pushover")
        nb.add(self.plot_frame_mc, text="M-φ Curve")
        nb.add(self.plot_frame_pm, text="P-M Diagram")
        nb.add(self.plot_frame_moment_diag, text="M Diagram")
        nb.add(self.plot_frame_strain_diag, text="Strain Diagram")
        nb.add(self.plot_frame_stiffness, text="Stiffness")
        nb.add(self.plot_frame_deflection, text="Force-Deflection")
        nb.add(self.plot_frame_section, text="Section")
        nb.add(self.results_frame, text="Results")

        self._build_results_tab(self.results_frame)

    def _build_results_tab(self, parent):
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.results_text = tk.Text(text_frame, wrap='word', font=('Courier', 10))
        scroll = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        self.results_text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

    def _get_float(self, var, name):
        try:
            return float(var.get())
        except ValueError:
            raise ValueError(f"Invalid input for {name}: '{var.get()}'")

    def _get_int(self, var, name):
        try:
            return int(var.get())
        except ValueError:
            raise ValueError(f"Invalid input for {name}: '{var.get()}'")

    def _run_analysis(self):
        self.run_btn.configure(state='disabled')
        self.status_lbl.configure(text="Running analysis...")
        t = threading.Thread(target=self._do_analysis, daemon=True)
        t.start()

    def _do_analysis(self):
        try:
            b = self._get_float(self.b, "Width") * 25.4
            h = self._get_float(self.h, "Height") * 25.4
            L = self._get_float(self.L, "Span") * 304.8
            fc = self._get_float(self.fc, "fc'") * 6.89476
            fy = self._get_float(self.fy, "fy") * 6.89476
            Es = self._get_float(self.Es, "Es") * 6.89476
            ecu = self._get_float(self.ecu, "ecu")
            As_top = self._get_float(self.As_top, "Top As") * 645.16
            d_top = self._get_float(self.d_top, "Top cover") * 25.4
            As_bot = self._get_float(self.As_bot, "Bottom As") * 645.16
            d_bot = self._get_float(self.d_bot, "Effective depth") * 25.4
            w_dist = self._get_float(self.w_dist, "Dist load") * 14.5939
            n_seg = self._get_int(self.n_seg, "Segments")
            n_steps = self._get_int(self.n_steps, "Steps")

            if b <= 0 or h <= 0 or L <= 0:
                raise ValueError("Dimensions must be positive")
            if fc <= 0 or fy <= 0:
                raise ValueError("Material strengths must be positive")

            sec = RCSection(b, h, d_top, d_bot, As_top, As_bot, fc, fy, Es,
                            conc_ecu=ecu)
            self.section = sec

            beam = Beam(sec, L, w_dist)
            analysis = PushoverAnalysis(beam, n_seg=n_seg, n_steps=n_steps)
            result = analysis.run()

            self.result = result
            self.root.after(0, self._update_results)

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            self.root.after(0, lambda: self.run_btn.configure(state='normal'))

    def _update_results(self):
        self.status_lbl.configure(text="Analysis complete")

        self.plotter.plot_pushover(self.plot_frame_pushover, self.result)

        try:
            self.plotter.plot_moment_curvature(self.plot_frame_mc, self.section)
        except Exception:
            pass

        try:
            self.plotter.plot_pm_interaction(self.plot_frame_pm, self.section)
        except Exception:
            pass

        try:
            self.plotter.plot_section(self.plot_frame_section, self.section)
        except Exception:
            pass

        try:
            self.plotter.plot_midspan_deflection(
                self.plot_frame_deflection, self.result)
        except Exception:
            pass

        try:
            self.plotter.plot_moment_diagram(
                self.plot_frame_moment_diag, self.result)
        except Exception:
            pass

        try:
            self.plotter.plot_axial_strain_diagram(
                self.plot_frame_strain_diag, self.result)
        except Exception:
            pass

        try:
            self.plotter.plot_stiffness(
                self.plot_frame_stiffness, self.result)
        except Exception:
            pass

        self._update_results_text()

    def _update_results_text(self):
        r = self.result
        sec = self.section
        txt = self.results_text
        txt.delete('1.0', tk.END)

        lines = [
            "=" * 60,
            "  PUSHOVER ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Beam: {sec.b / 25.4:.1f} x {sec.h / 25.4:.1f} in, Span = {self.beam_L():.1f} ft",
            f"  Concrete: fc' = {sec.concrete.fc / 6.89476:.1f} ksi",
            f"  Steel: fy = {sec.steel.fy / 6.89476:.0f} ksi",
            f"  Top reinforcement: {sec.As_top / 645.16:.2f} in² at d' = {sec.d_top / 25.4:.1f} in",
            f"  Bottom reinforcement: {sec.As_bot / 645.16:.2f} in² at d = {(sec.h - sec.d_bot) / 25.4:.1f} in (cover = {sec.d_bot / 25.4:.1f} in)",
            f"  Distributed load: {self.beam_w():.1f} kip/ft",
            "",
            "-" * 60,
            "  PUSHOVER RESULT SUMMARY",
            "-" * 60,
            "",
        ]

        if len(r.axial_forces) > 0:
            max_P = np.max(np.abs(r.axial_forces)) / 4448.22
            max_disp = r.displacements[-1] / 25.4 if len(r.displacements) > 0 else 0
            max_M = np.max(np.abs(r.max_moments)) / 1.355819e6 if len(r.max_moments) > 0 else 0
            max_v = np.max(np.abs(r.midspan_deflections)) / 25.4 if len(r.midspan_deflections) > 0 else 0

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
                    f"  {i:>5} {r.displacements[i] / 25.4:>10.4f} "
                    f"{abs(r.axial_forces[i]) / 4448.22:>12.2f} "
                    f"{r.midspan_deflections[i] / 25.4:>10.4f}"
                )

        lines += [
            "",
            "=" * 60,
            "  ANALYSIS COMPLETE",
            "=" * 60,
        ]

        txt.insert('1.0', '\n'.join(lines))

    def beam_L(self):
        if self.section and hasattr(self, 'L'):
            return float(self.L.get())
        return 0

    def beam_w(self):
        if hasattr(self, 'w_dist'):
            return float(self.w_dist.get())
        return 0

    def _show_error(self, msg):
        self.status_lbl.configure(text="Error")
        messagebox.showerror("Analysis Error", msg)
        self.run_btn.configure(state='normal')

    def run(self):
        self.root.mainloop()
