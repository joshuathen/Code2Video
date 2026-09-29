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
        lecture_lines = ["Vectors are arrows from origin to a point.", "We define space with standard basis vectors.", "Unit vectors i and j form the grid."]
        self.setup_layout("Prerequisite Review: Vectors as Points in Space", lecture_lines)
        
        # Grid setup
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'B4', 'F6', scale_factor=0.6)
        
        # Labels for axes
        i_label = Text("i", color="#FFFF00", font_size=24)
        j_label = Text("j", color="#FFFF00", font_size=24)
        origin_label = Text("O", color="#FFFF00", font_size=24)
        self.place_at_grid(i_label, 'E5', scale_factor=0.8)
        self.place_at_grid(j_label, 'C4', scale_factor=0.8)
        self.place_at_grid(origin_label, 'E4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        v_coords = axes.c2p(2, 3)
        vector_v = Arrow(start=axes.c2p(0, 0), end=v_coords, color="#00FF00", buff=0)
        
        # Mock asset loading - using a simple circle for the robot scout
        robot_scout = Circle(radius=0.1, color=WHITE, fill_opacity=1)
        robot_scout.move_to(v_coords)
        
        self.play(Create(axes), Create(vector_v), FadeIn(robot_scout))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(Write(i_label), Write(j_label), Write(origin_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        i_vec = Arrow(start=axes.c2p(0, 0), end=axes.c2p(1, 0), color="#FF00FF", buff=0)
        j_vec = Arrow(start=axes.c2p(0, 0), end=axes.c2p(0, 1), color="#FF00FF", buff=0)
        self.play(Create(i_vec), Create(j_vec))
        self.wait(2)
