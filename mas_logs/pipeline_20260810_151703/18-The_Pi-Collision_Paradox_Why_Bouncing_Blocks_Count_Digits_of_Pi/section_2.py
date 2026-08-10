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

class Section2Scene(TeachingScene):
    def construct(self):
        lines = [
            "Map velocities as coordinates on a plane.",
            "Wall impacts look like reflections across axes.",
            "Block collisions reflect across a constant line."
        ]
        self.setup_layout("Prerequisite: The Mapping to Velocity Space", lines)
        
        # Axes and visual space
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": True}).scale(0.5)
        
        # Assets
        wall_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(axes, 'B3', 'E5', scale_factor=1.0)
        
        wedge_line1 = Line(axes.c2p(0, 0), axes.c2p(4, 0), color=BLUE)
        wedge_line2 = Line(axes.c2p(0, 0), axes.c2p(4, 4), color=RED)
        circle1 = Circle(radius=0.4, color=GREEN).move_to(axes.c2p(2, 2))
        
        self.play(Create(axes), Create(wedge_line1), Create(wedge_line2), Create(circle1))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        v_vector = Vector(axes.c2p(1, 2) - axes.c2p(0, 0), color=YELLOW)
        self.place_at_grid(v_vector, 'C4', scale_factor=0.9)
        self.place_at_grid(wall_icon, 'A3', scale_factor=0.5)
        
        self.play(GrowArrow(v_vector), FadeIn(wall_icon))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        v1_label = Tex(r"$v_1 = 0$").scale(0.5)
        v2_label = Tex(r"$v_2 = v_1$").scale(0.5)
        
        self.place_at_grid(v1_label, 'E3', scale_factor=0.7)
        self.place_at_grid(v2_label, 'B6', scale_factor=0.7)
        self.place_at_grid(block_icon, 'F3', scale_factor=0.5)
        
        self.play(Write(v1_label), Write(v2_label), FadeIn(block_icon))
        self.wait(2)
