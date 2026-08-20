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
        lecture_lines = [
            "Linear combinations combine vectors: c1v1 plus c2v2.",
            "Span is the set of all reachable points.",
            "Varying coefficients covers the entire plane."
        ]
        self.setup_layout("Linear Combinations and Span", lecture_lines)
        
        # Asset: origin
        origin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg")
        self.place_at_grid(origin_icon, 'C4', scale_factor=0.5)

        # Animation Elements
        v1 = Vector([1, 0.5], color="#FF5733")
        v2 = Vector([0, 1], color="#33FF57")
        self.place_at_grid(v1, 'C4')
        self.place_at_grid(v2, 'C4')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        v1_label = MathTex("v_1", color="#FF5733").scale(0.7)
        v2_label = MathTex("v_2", color="#33FF57").scale(0.7)
        
        # Fix: Positioning labels using the grid
        self.place_at_grid(v1_label, 'C2')
        self.place_at_grid(v2_label, 'B3')
        
        self.play(FadeIn(origin_icon), Create(v1), Create(v2), Write(v1_label), Write(v2_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00CED1")
        
        # Parallelogram group for v1 + v2
        parallelogram = Polygon(
            ORIGIN, v1.get_end(), v1.get_end() + v2.get_end(), v2.get_end(),
            color="#00CED1", fill_opacity=0.3
        )
        v1_plus_v2 = Vector(v1.get_end() + v2.get_end(), color="#00CED1")
        v1_plus_v2_label = MathTex("v_1 + v_2", color="#00CED1").scale(0.7)
        vector_sum_group = VGroup(parallelogram, v1_plus_v2, v1_plus_v2_label)
        
        self.place_in_area(vector_sum_group, 'B4', 'D6', scale_factor=0.8)
        
        self.play(Create(parallelogram), Create(v1_plus_v2), Write(v1_plus_v2_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Grid/Span visualization
        span_dots_grid = VGroup()
        for i in range(-2, 3):
            for j in range(-2, 3):
                point = i * v1.get_end() + j * v2.get_end()
                dot = Dot(point, color="#FFFF00", radius=0.05)
                span_dots_grid.add(dot)
        
        # Fix: Placing span grid in area E1-F6
        self.place_in_area(span_dots_grid, 'E1', 'F6', scale_factor=0.7)
                
        self.play(FadeIn(span_dots_grid))
        self.wait(2)
