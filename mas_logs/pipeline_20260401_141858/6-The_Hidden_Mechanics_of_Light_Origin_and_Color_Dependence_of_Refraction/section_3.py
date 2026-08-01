from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        # Configuration and Colors
        COLOR_VACUUM = "#00FFFF"   # Cyan
        COLOR_MEDIUM = "#FFFFFF"   # White
        COLOR_INDUCED = "#00FF00"  # Green
        COLOR_RESULTANT = "#FFFFFF" # White
        
        lecture_lines = [
            "Driving and induced waves exhibit a relative phase shift.",
            "Rotating phasors represent these electric field components.",
            "Vector addition yields a resultant with a retarded phase.",
            "Permittivity scales the magnitude of the induced response.",
            "Refractive index n depends on the material's relative permittivity."
        ]
        
        self.setup_layout("Mathematical Visualization: The Phase Velocity", lecture_lines)
        
        # Trackers for dynamics
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda dt: time_tracker.increment_value(dt))
        
        # Physics Parameters
        k_val = 0.6            # Induced strength factor
        phi_val = np.arctan(k_val) # Phase lag in radians
        omega_val = 3.5        # Angular frequency
        wave_k = 2.5           # Spatial frequency
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_VACUUM)
        
        vac_label = Text("Vacuum Wave", font_size=18, color=COLOR_VACUUM)
        self.place_at_grid(vac_label, "A3")
        
        def get_vac_wave():
            t = time_tracker.get_value()
            g = FunctionGraph(
                lambda x: 0.35 * np.sin(wave_k * x - omega_val * t),
                x_range=[-2.2, 2.2], color=COLOR_VACUUM
            )
            return self.place_in_area(g, "B1", "B6")

        vac_wave = always_redraw(get_vac_wave)
        
        self.play(Write(vac_label), Create(vac_wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_MEDIUM)
        
        med_label = Text("Medium Wave", font_size=18, color=COLOR_MEDIUM)
        self.place_at_grid(med_label, "C3")
        
        def get_med_wave():
            t = time_tracker.get_value()
            # Resultant wave with phase lag phi
            g = FunctionGraph(
                lambda x: 0.35 * 0.9 * np.sin(wave_k * x - omega_val * t - phi_val),
                x_range=[-2.2, 2.2], color=COLOR_MEDIUM
            )
            return self.place_in_area(g, "D1", "D6")

        med_wave = always_redraw(get_med_wave)
        
        self.play(Write(med_label), Create(med_wave))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_VACUUM)
        
        phasor_origin = self.grid["E2"]
        phasor_circle = Circle(radius=0.7, color=GRAY, stroke_opacity=0.3)
        phasor_circle.move_to(phasor_origin)
        self.add(phasor_circle)

        def get_driving_phasor():
            t = time_tracker.get_value()
            v = 0.7 * np.array([np.cos(omega_val * t), np.sin(omega_val * t), 0])
            return Arrow(start=phasor_origin, end=phasor_origin + v, buff=0, color=COLOR_VACUUM, stroke_width=4)
            
        driving_phasor = always_redraw(get_driving_phasor)
        driving_lbl = Text("E_dr", font_size=16, color=COLOR_VACUUM)
        self.place_at_grid(driving_lbl, "E1")

        self.play(Create(driving_phasor), Write(driving_lbl))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_INDUCED)

        def get_induced_phasor():
            t = time_tracker.get_value()
            v = (k_val * 0.7) * np.array([np.cos(omega_val * t - PI/2), np.sin(omega_val * t - PI/2), 0])
            return Arrow(start=phasor_origin, end=phasor_origin + v, buff=0, color=COLOR_INDUCED, stroke_width=4)

        induced_phasor = always_redraw(get_induced_phasor)
        ind_lbl = Text("E_ind", font_size=16, color=COLOR_INDUCED)
        self.place_at_grid(ind_lbl, "F1")

        self.play(Create(induced_phasor), Write(ind_lbl))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_RESULTANT)

        def get_resultant_phasor():
            t = time_tracker.get_value()
            vec_dr = 0.7 * np.array([np.cos(omega_val * t), np.sin(omega_val * t), 0])
            vec_ind = (k_val * 0.7) * np.array([np.cos(omega_val * t - PI/2), np.sin(omega_val * t - PI/2), 0])
            return Arrow(start=phasor_origin, end=phasor_origin + vec_dr + vec_ind, buff=0, color=COLOR_RESULTANT, stroke_width=6)

        resultant_phasor = always_redraw(get_resultant_phasor)
        res_lbl = Text("E_res", font_size=16, color=COLOR_RESULTANT)
        self.place_at_grid(res_lbl, "E3")
        
        # Unicode: n approx sqrt(epsilon_r) -> n \u2248 \u221a\u03b5\u1d63
        formula = Text("n \u2248 \u221a\u03b5\u1d63", font_size=28, color=COLOR_RESULTANT)
        self.place_at_grid(formula, "F5")

        self.play(Create(resultant_phasor), Write(res_lbl))
        self.play(Write(formula))
        self.wait(3)
