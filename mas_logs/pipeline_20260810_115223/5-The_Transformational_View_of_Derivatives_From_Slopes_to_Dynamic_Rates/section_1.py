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
        lecture_lines = ["Constant speed means a constant slope.", "Rise over run gives the steepness.", "Steady progress on a grid."]
        self.setup_layout("Prerequisite: The Static Slope", lecture_lines)
        
        # Pre-build objects
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], axis_config={"include_numbers": False}).scale(0.6)
        line = Line(axes.c2p(0, 0), axes.c2p(4, 3), color=WHITE)
        triangle = Polygon(axes.c2p(1, 0.75), axes.c2p(3, 0.75), axes.c2p(3, 2.25), color=WHITE, fill_opacity=0.2)
        dy = Line(axes.c2p(3, 0.75), axes.c2p(3, 2.25), color="#FF00FF")
        dx = Line(axes.c2p(1, 0.75), axes.c2p(3, 0.75), color="#FF00FF")
        dy_label = Text("Δy", font_size=20, color="#FF00FF").next_to(dy, RIGHT)
        dx_label = Text("Δx", font_size=20, color="#FF00FF").next_to(dx, DOWN)
        
        # Asset integration
        turtle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dot.svg", color="#00FF00").scale(0.2)
        slope_text = Text("Slope = Δy/Δx", font_size=24, color="#FFFF00")
        
        group = VGroup(axes, line, triangle, dy, dx, dy_label, dx_label, turtle)
        self.place_in_area(group, 'A3', 'F6', scale_factor=0.6)
        self.place_at_grid(slope_text, 'B2', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Create(axes), Create(line))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        self.play(Create(triangle), Create(dy), Create(dx), Write(dy_label), Write(dx_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        self.add(turtle)
        turtle.move_to(line.get_start())
        self.play(MoveAlongPath(turtle, line), run_time=2)
        self.play(Write(slope_text))
