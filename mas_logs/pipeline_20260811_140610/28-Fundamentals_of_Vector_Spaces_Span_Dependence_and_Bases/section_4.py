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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "A basis is a minimal spanning set.",
            "It provides an efficient coordinate system.",
            "No redundancy, yet spans the space perfectly."
        ]
        self.setup_layout("Bases: The Minimalist Framework", lecture_lines)
        
        # Setup grid for visual
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": True, "color": GRAY}
        )
        # Using place_in_area to fit the grid - Fixed per VideoCritic
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.4)
        self.add(axes)

        # Asset: grid.svg
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_at_grid(grid_asset, 'C4', scale_factor=0.5)
        self.add(grid_asset)

        # Basis vectors i and j
        i_vec = Vector(RIGHT, color=YELLOW)
        j_vec = Vector(UP, color=YELLOW)
        
        i_label = MathTex(r"\\hat{i}", color=YELLOW)
        j_label = MathTex(r"\\hat{j}", color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(i_vec), Create(j_vec))
        
        # Fixed labels per VideoCritic
        self.place_at_grid(i_label, 'C6', scale_factor=0.8)
        i_label.next_to(i_vec.get_end(), DOWN, buff=0.1)
        self.place_at_grid(j_label, 'B5', scale_factor=0.8)
        j_label.next_to(j_vec.get_end(), LEFT, buff=0.1)
        
        self.play(Write(i_label), Write(j_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Grid lines representation
        grid_lines = VGroup()
        for k in range(-2, 3):
            line_h = Line(axes.c2p(-2, k), axes.c2p(2, k), stroke_width=1, color=BLUE_D)
            line_v = Line(axes.c2p(k, -2), axes.c2p(k, 2), stroke_width=1, color=BLUE_D)
            grid_lines.add(line_h, line_v)
        
        self.play(Create(grid_lines))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.05) for x in range(-2, 3) for y in range(-2, 3)])
        self.play(FadeIn(dots))
        self.wait(2)
