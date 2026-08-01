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
        # Setup the layout
        title = "The Physics: Energy to Velocity"
        lines = [
            "In gravity, potential energy converts to kinetic energy.",
            "Velocity depends on the vertical distance fallen.",
            "This links depth directly to the speed of travel."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_WIRE = "#888888"
        COLOR_BEAD = "#FFFFFF"
        COLOR_Y = "#FF00FF"
        COLOR_V = "#00FFFF"
        
        # Positions
        start_pt = self.grid["B2"]
        end_pt = self.grid["E5"]
        cp1 = self.grid["D2"]
        cp2 = self.grid["E3"]
        
        # Wire
        wire = CubicBezier(start_pt, cp1, cp2, end_pt, color=COLOR_WIRE)
        
        # Pre-sampled points for the curve to avoid expensive calculations in updaters
        NUM_SAMPLES = 60 
        samples = [wire.point_from_proportion(i/(NUM_SAMPLES-1)) for i in range(NUM_SAMPLES)]
        
        def get_pos(alpha_val):
            alpha_val = np.clip(alpha_val, 0, 1)
            idx = alpha_val * (NUM_SAMPLES - 1)
            i0 = int(idx)
            i1 = min(i0 + 1, NUM_SAMPLES - 1)
            frac = idx - i0
            return (1-frac)*samples[i0] + frac*samples[i1]

        # Mobjects
        bead = Dot(color=COLOR_BEAD, radius=0.1)
        bead.move_to(start_pt)
        
        y_bar = Line(start_pt, start_pt, color=COLOR_Y, stroke_width=4)
        # Using Text for the simple label 'y' to reduce MathTex rendering overhead
        y_label = Text("y", color=COLOR_Y, font_size=20)
        y_bar.set_opacity(0)
        y_label.set_opacity(0)
        
        formula = MathTex(r"v = \sqrt{2gy}", color=COLOR_V, font_size=28)
        self.place_at_grid(formula, "A5")
        formula.set_opacity(0)
        
        # Velocity arrow - reusing persistent object
        v_arrow = Arrow(start_pt, start_pt + RIGHT*0.1, color=COLOR_V, buff=0, 
                        stroke_width=3, max_tip_length_to_length_ratio=0.25)
        v_arrow.set_opacity(0)
        
        # Add objects to scene
        self.add(y_bar, y_label, v_arrow, formula)
        
        alpha = ValueTracker(0)
        
        # Optimized updater for all dynamic elements
        def update_elements(m):
            a = alpha.get_value()
            p = get_pos(a)
            bead.move_to(p)
            
            y_drop = max(0, start_pt[1] - p[1])
            if y_drop > 0.05:
                y_top = np.array([p[0], start_pt[1], 0])
                y_bar.put_start_and_end_on(y_top, p)
                y_bar.set_opacity(1)
                y_label.move_to(y_top + DOWN * (y_drop / 2) + LEFT * 0.25)
                y_label.set_opacity(1)
            else:
                y_bar.set_opacity(0)
                y_label.set_opacity(0)
            
            if a > 0.02:
                # Calculate direction from a slightly earlier point
                p_prev = get_pos(a - 0.02)
                dir_vec = p - p_prev
                dist = np.linalg.norm(dir_vec)
                if dist > 1e-4:
                    unit_v = dir_vec / dist
                    v_mag = np.sqrt(y_drop) * 1.2
                    if v_mag > 0.1:
                        v_arrow.put_start_and_end_on(p, p + unit_v * v_mag)
                        v_arrow.set_opacity(1)
                    else:
                        v_arrow.set_opacity(0)
                else:
                    v_arrow.set_opacity(0)
            else:
                v_arrow.set_opacity(0)

        bead.add_updater(update_elements)

        # === Animation for Lecture Line 1 ===
        # In gravity, potential energy converts to kinetic energy.
        self.lecture[0].set_color(WHITE)
        self.play(Create(wire), FadeIn(bead), run_time=1.0)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Velocity depends on the vertical distance fallen.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(COLOR_Y)
        # Drop a little bit to show 'y'
        self.play(alpha.animate.set_value(0.3), run_time=1.2, rate_func=smooth)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # This links depth directly to the speed of travel.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(COLOR_V)
        self.play(formula.animate.set_opacity(1), run_time=0.5)
        # Full descent with linear progress along the curve
        self.play(alpha.animate.set_value(1.0), run_time=2.5, rate_func=linear)
        self.wait(1.5)
        
        bead.clear_updaters()
