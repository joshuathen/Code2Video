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
            "Scaling a basis vector scales the area.", 
            "Slide vector b along the span.", 
            "Area changes linearly with x.", 
            "The variable x becomes geometrically isolated.", 
            "This completes our geometric proof."
        ]
        self.setup_layout("Visual Proof: Areas in Ratio", lecture_lines)
        
        # Create visual elements
        axes = Axes(x_range=[0, 3], y_range=[0, 3], axis_config={"include_tip": False}).scale(0.5)
        square = Polygon([0,0,0], [1,0,0], [1,1,0], [0,1,0], color=BLUE).scale(0.5)
        para = Polygon([0,0,0], [2,0,0], [3,1,0], [1,1,0], color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(square, 'C2')
        self.play(Create(square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(para, 'C4', 'D5', scale_factor=0.9)
        self.play(Transform(square.copy(), para))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        area_label = MathTex(r"\text{Area} \propto x").set_color(GREEN)
        self.place_at_grid(area_label, 'B5', scale_factor=0.8)
        self.play(Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        dot = Dot(color=RED)
        self.place_at_grid(dot, 'D3')
        self.play(FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        check = Tex(r"$\checkmark$", color=PURPLE)
        self.place_at_grid(check, 'E4', scale_factor=0.7)
        self.play(Write(check))
        self.wait(2)
