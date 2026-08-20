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
        self.setup_layout("Mechanism: The Projection and Activation Cycle", [
            "Input vectors project into high-dimensional space.",
            "Non-linear functions activate these neurons.",
            "Facts exist as distinct vector directions.",
            "Expansion helps isolate stored information.",
            "Projection back completes the cycle."
        ])
        
        # Elements
        input_dot = Dot(color="#FF9900")
        grid_visual = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        activation_pulse = Circle(radius=0.3, color="#FFFF00")
        output_label = Text("Satoshi Nakamoto", font_size=24, color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF9900")
        self.place_at_grid(input_dot, 'B3', scale_factor=0.6)
        self.play(FadeIn(input_dot))
        self.play(input_dot.animate.move_to(self.grid['C4']))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_at_grid(grid_visual, 'C4', scale_factor=1.5)
        self.play(FadeIn(grid_visual))
        self.play(grid_visual.animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(activation_pulse, 'B4', scale_factor=0.7)
        self.play(Flash(activation_pulse))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        trail = TracedPath(input_dot.get_center, stroke_width=2, stroke_color="#FFFFFF")
        self.add(trail)
        self.play(input_dot.animate.move_to(self.grid['D5']))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.place_in_area(output_label, 'C5', 'D6', scale_factor=0.8)
        self.play(Write(output_label))
        self.wait(2)
