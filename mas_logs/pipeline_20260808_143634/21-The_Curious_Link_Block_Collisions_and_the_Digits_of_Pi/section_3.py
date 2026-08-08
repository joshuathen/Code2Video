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
        self.setup_layout("The Geometric Transformation", [
            "Collision paths map to geometric reflections.",
            "Trajectories bounce inside a wedge shape.",
            "Wedge angles depend on mass ratios."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset
        wedge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wedge.svg")
        self.place_at_grid(wedge, "E5", scale_factor=0.5)
        self.play(FadeIn(wedge))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Visualizing trajectory bouncing
        path = VGroup(
            Line(self.grid["B3"], self.grid["E2"], color=WHITE),
            Line(self.grid["E2"], self.grid["E5"], color=WHITE),
            Line(self.grid["E5"], self.grid["B3"], color=WHITE)
        )
        self.play(Create(path))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        # Indicating the wedge angle dependence
        angle_arc = Arc(radius=0.5, start_angle=0, angle=PI/6, color=RED)
        angle_arc.move_to(self.grid["E4"])
        angle_label = MathTex(r"\\theta", color=RED).next_to(angle_arc, RIGHT, buff=0.1)
        
        self.play(Create(angle_arc), Write(angle_label))
        self.lecture[2].set_color(RED)
        
        self.wait(2)
