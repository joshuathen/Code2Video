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
        # Initialize layout
        self.setup_layout("Prerequisite: Determinant as Area", [
            "The determinant represents the area of a parallelogram.",
            "Columns v1 and v2 define the base area.",
            "This area is the determinant of matrix A."
        ])

        # Define key points using the grid for geometry construction
        # Origin at E2, v1=(2,1)->D4, v2=(1,2)->C3, Corner=(3,3)->B5
        origin = self.grid["E2"]
        v1_end = self.grid["D4"]
        v2_end = self.grid["C3"]
        v_sum = self.grid["B5"]

        # === Animation for Lecture Line 1 ===
        # Line 1: "The determinant represents the area of a parallelogram."
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        
        v1_color = "#87CEEB" # Sky Blue
        v2_color = "#98FB98" # Pale Green
        
        v1_vec = Arrow(origin, v1_end, buff=0, color=v1_color)
        v2_vec = Arrow(origin, v2_end, buff=0, color=v2_color)
        
        v1_label = MathTex(r"\vec{v}_1", color=v1_color, font_size=24).next_to(v1_end, RIGHT, buff=0.1)
        v2_label = MathTex(r"\vec{v}_2", color=v2_color, font_size=24).next_to(v2_end, LEFT, buff=0.1)
        
        self.play(Create(v1_vec), Write(v1_label))
        self.play(Create(v2_vec), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Columns v1 and v2 define the base area."
        self.play(self.lecture[1].animate.set_color("#98FB98"))
        
        # Parallelogram boundary
        l1 = DashedLine(v1_end, v_sum, color=GRAY)
        l2 = DashedLine(v2_end, v_sum, color=GRAY)
        
        # Parallelogram fill
        parallelogram = Polygon(
            origin, v1_end, v_sum, v2_end,
            stroke_width=0,
            fill_color="#FFFF00",
            fill_opacity=0.3
        )
        
        self.play(Create(l1), Create(l2))
        self.play(FadeIn(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "This area is the determinant of matrix A."
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Display the label 'Area = Det(A) = 3'
        area_label = MathTex(r"\text{Area} = \text{Det}(A) = 3", color="#FFFF00", font_size=32)
        
        # Fix for Issue 25: Place in area B2 to B5 for better centering and visibility
        self.place_in_area(area_label, 'B2', 'B5', scale_factor=1.0)
        
        self.play(Write(area_label))
        self.wait(2)
