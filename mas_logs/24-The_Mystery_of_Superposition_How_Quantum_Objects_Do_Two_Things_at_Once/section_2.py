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
        # Initialize Scene
        title = "Prerequisite: The Language of Vectors"
        lines = [
            "Physicists use arrows, or vectors, to represent states.",
            "Vertical is State A; horizontal is State B.",
            "A diagonal arrow represents a mixture of both."
        ]
        self.setup_layout(title, lines)

        # Positions from Grid
        origin_pt = self.grid["E2"]
        state_a_pt = self.grid["B2"]
        state_b_pt = self.grid["E5"]
        super_pt = self.grid["B5"]

        # === Animation for Lecture Line 1 ===
        # Physicists use arrows, or vectors, to represent states.
        self.lecture[0].set_color("#FFFFFF")
        
        # Basis Vectors
        arrow_a = Arrow(origin_pt, state_a_pt, color="#00FF00", buff=0, stroke_width=4)
        arrow_b = Arrow(origin_pt, state_b_pt, color="#FFFF00", buff=0, stroke_width=4)
        
        self.play(
            GrowArrow(arrow_a),
            GrowArrow(arrow_b),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vertical is State A; horizontal is State B.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00") # Highlighting State definitions
        
        # Labels for Basis Vectors using Unicode for Braket notation
        label_a = Text("|0⟩", font_size=24, color="#00FF00")
        label_b = Text("|1⟩", font_size=24, color="#FFFF00")
        
        self.place_at_grid(label_a, "A2", scale_factor=0.8)
        self.place_at_grid(label_b, "F6", scale_factor=0.8)
        
        self.play(
            FadeIn(label_a),
            FadeIn(label_b),
            run_time=1
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # A diagonal arrow represents a mixture of both.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        # Superposition Vector
        arrow_psi = Arrow(origin_pt, super_pt, color="#FFFFFF", buff=0, stroke_width=6)
        label_psi = Text("|ψ⟩", font_size=28, color="#FFFFFF")
        self.place_at_grid(label_psi, "A5", scale_factor=0.8)
        
        # Projection lines
        proj_h = DashedLine(super_pt, [origin_pt[0], super_pt[1], 0], color="#888888")
        proj_v = DashedLine(super_pt, [super_pt[0], origin_pt[1], 0], color="#888888")
        
        self.play(
            GrowArrow(arrow_psi),
            FadeIn(label_psi),
            run_time=1.5
        )
        self.play(
            Create(proj_h),
            Create(proj_v),
            run_time=1
        )
        self.wait(3)
