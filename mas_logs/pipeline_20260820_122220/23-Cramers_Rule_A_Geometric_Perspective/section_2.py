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
        self.setup_layout("Framing the Linear System", [
            "A linear system Ax equals b.",
            "This is a combination of column vectors.",
            "We find scalars to reach point b."
        ])

        # --- Assets ---
        # Vectors
        vec_v1 = Arrow(ORIGIN, [0, 1.5, 0], color="#FFD700")
        vec_v2 = Arrow(ORIGIN, [1.5, 0.5, 0], color="#FFD700")
        vec_b = Arrow(ORIGIN, [1.5, 2.0, 0], color="#00BFFF")
        
        # Labels
        label_v1 = MathTex(r"v_1", color="#FFD700")
        label_v2 = MathTex(r"v_2", color="#FFD700")
        label_b = MathTex(r"b", color="#00BFFF")
        
        # Equation
        eq = MathTex(r"A", r"x", r"=", r"b", font_size=32)
        eq.set_color_by_tex("A", WHITE)
        eq.set_color_by_tex("x", "#FF4500")
        eq.set_color_by_tex("b", "#00BFFF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00BFFF")
        self.place_in_area(eq, 'B2', 'B3', scale_factor=1.2)
        self.play(Write(eq))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        
        # Vector visualization (B019, B004: use grid columns 4-6)
        self.place_at_grid(vec_v1, 'C4', scale_factor=0.9)
        self.place_at_grid(label_v1, 'C4') # Tethered
        label_v1.next_to(vec_v1, UP, buff=0.1).scale(0.7)
        
        self.place_at_grid(vec_v2, 'D4', scale_factor=0.9)
        self.place_at_grid(label_v2, 'D4') # Tethered
        label_v2.next_to(vec_v2, UP, buff=0.1).scale(0.7)
        
        self.play(Create(vec_v1), Write(label_v1), Create(vec_v2), Write(label_v2))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00BFFF")
        
        # Vector sum
        self.place_at_grid(vec_b, 'E4', scale_factor=1.0)
        self.place_at_grid(label_b, 'F4') # Tethered
        label_b.next_to(vec_b, UP, buff=0.1).scale(0.7)
        
        self.play(Create(vec_b), Write(label_b))
        self.wait(1)
