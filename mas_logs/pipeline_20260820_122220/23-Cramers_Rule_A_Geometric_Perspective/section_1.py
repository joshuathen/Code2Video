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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Determinants represent the area of a parallelogram.",
            "Vectors forming the parallelogram determine this area.",
            "Zero area means vectors are linearly dependent."
        ]
        self.setup_layout("Prerequisite Review: Determinants as Area", lecture_lines)

        # --- Assets ---
        # Note: Using SVGMobject for asset references
        parallelogram_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")

        # --- Animation Components ---
        axes = Axes(x_range=[-1, 3, 1], y_range=[-1, 3, 1], axis_config={"include_tip": False}, tips=False).scale(0.5)
        self.place_in_area(axes, "C2", "F6", scale_factor=0.85)
        
        v1_val = np.array([2, 1, 0])
        v2_val = np.array([0.5, 1.5, 0])
        
        v1 = Vector(v1_val, color="#FFD700")
        v2 = Vector(v2_val, color="#FFD700")
        
        # Parallelogram using polygon (using asset icon for placeholder if needed, but keeping polygon for physics)
        poly = Polygon(ORIGIN, v1_val, v1_val + v2_val, v2_val, color="#00BFFF", fill_opacity=0.3)
        poly.set_stroke(width=2)
        
        # Tethered labels (B011, B020)
        l1 = MathTex(r"\\vec{v}_1", color=WHITE).scale(0.7).next_to(v1.get_end(), UP, buff=0.1)
        l2 = MathTex(r"\\vec{v}_2", color=WHITE).scale(0.7).next_to(v2.get_end(), RIGHT, buff=0.1)
        det_label = MathTex(r"det(A)", color=WHITE).scale(0.7)
        self.place_at_grid(det_label, "D5", scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.add(axes, poly, v1, v2, l1, l2)
        self.play(FadeIn(poly), FadeIn(v1), FadeIn(v2), FadeIn(l1), FadeIn(l2), Write(det_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00BFFF")
        v1_tracker = ValueTracker(2.0)
        v2_tracker = ValueTracker(1.5)
        
        # Updater logic (B010)
        def update_poly(p):
            new_v1 = v1_tracker.get_value() * RIGHT + 0.5 * UP
            new_v2 = 0.5 * RIGHT + v2_tracker.get_value() * UP
            p.become(Polygon(ORIGIN, new_v1, new_v1 + new_v2, new_v2, color="#00BFFF", fill_opacity=0.3).set_stroke(width=2))
        
        poly.add_updater(update_poly)
        self.play(v1_tracker.animate.set_value(2.5), v2_tracker.animate.set_value(1.0), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.play(v1_tracker.animate.set_value(1.0), v2_tracker.animate.set_value(0.5), run_time=2)
        poly.set_fill(color="#FF4500", opacity=0.5)
        self.play(Flash(poly, color="#32CD32", line_length=0.2, num_lines=12))
