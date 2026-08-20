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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Fractal dimension measures geometric roughness and complexity.",
            "Lightning bolts possess fractional dimensions between one and two.",
            "This tool quantifies shapes beyond Euclidean limitations."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)

        # Assets/Objects
        val = ValueTracker(1.0)
        
        # Persistent mobject to represent complexity
        curve = VMobject(color=WHITE)
        curve.set_points_smoothly([np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0])])
        self.place_in_area(curve, 'B4', 'D6', scale_factor=0.7)
        self.add(curve)

        def update_curve(m):
            d = val.get_value()
            points = []
            for i in range(11):
                x = -1 + i * 0.2
                y = np.sin(x * 5 * d) * 0.5 * (d - 1)
                points.append(np.array([x, y, 0]))
            m.set_points_smoothly(points)
        
        curve.add_updater(update_curve)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(val.animate.set_value(1.5), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Load asset and add label
        lightning = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightning.svg")
        self.place_in_area(lightning, "B2", "D4", scale_factor=0.5)
        d_label = Text("D ≈ 1.3", color=YELLOW, font_size=24)
        self.place_at_grid(d_label, "E3", scale_factor=0.8)
        
        self.play(FadeIn(lightning), Write(d_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
