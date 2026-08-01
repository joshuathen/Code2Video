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
        # Setup the scene layout with updated prompt text
        title = "Defining a New Perspective (The New Basis)"
        lines = [
            "We can choose any two independent vectors as basis.",
            "These new vectors define a slanted coordinate grid.",
            "We call this new coordinate system Basis B."
        ]
        self.setup_layout(title, lines)
        
        # Color mapping for lines and vectors
        COLOR_B1 = "#FF8C00"
        COLOR_B2 = "#1E90FF"
        COLOR_GRID = "#8B4513"
        
        # Visual Origin and Vectors
        origin_pos = self.grid["D3"]
        b1_vec_raw = np.array([2, 1, 0])
        b2_vec_raw = np.array([-1, 1, 0])
        
        # Standard basis for transition
        i_vec = Arrow(origin_pos, origin_pos + RIGHT, buff=0, color=WHITE)
        j_vec = Arrow(origin_pos, origin_pos + UP, buff=0, color=WHITE)
        
        # New basis vectors
        b1_vec = Arrow(origin_pos, origin_pos + b1_vec_raw, buff=0, color=COLOR_B1)
        b2_vec = Arrow(origin_pos, origin_pos + b2_vec_raw, buff=0, color=COLOR_B2)
        
        # Labels - Using Text instead of MathTex to avoid 'latex' dependency
        label_b1 = Text("b1", color=COLOR_B1)
        label_b2 = Text("b2", color=COLOR_B2)
        label_basis = Text("Basis B", color=COLOR_GRID)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_B1))
        
        # Show standard basis first
        self.play(Create(i_vec), Create(j_vec))
        self.wait(1)
        
        # Replace i and j with b1 and b2
        self.play(
            ReplacementTransform(i_vec, b1_vec),
            ReplacementTransform(j_vec, b2_vec),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line, dim first
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_B2)
        )
        
        # Label the new vectors - Addressing Issue 35 and 36
        self.place_at_grid(label_b1, "A5", scale_factor=0.8)
        self.place_at_grid(label_b2, "A3", scale_factor=0.8)
        
        self.play(Write(label_b1), Write(label_b2))
        
        # Create skewed grid
        grid_lines = VGroup()
        # Lines parallel to b1
        for k in range(-2, 3):
            line_b1 = Line(
                origin_pos + k * b2_vec_raw - 1.2 * b1_vec_raw,
                origin_pos + k * b2_vec_raw + 1.5 * b1_vec_raw,
                color=COLOR_GRID,
                stroke_opacity=0.4,
                stroke_width=2
            )
            grid_lines.add(line_b1)
        
        # Lines parallel to b2
        for k in range(-1, 3):
            line_b2 = Line(
                origin_pos + k * b1_vec_raw - 2.0 * b2_vec_raw,
                origin_pos + k * b1_vec_raw + 2.0 * b2_vec_raw,
                color=COLOR_GRID,
                stroke_opacity=0.4,
                stroke_width=2
            )
            grid_lines.add(line_b2)
            
        # Ensure grid is behind vectors
        self.add_foreground_mobjects(b1_vec, b2_vec, label_b1, label_b2)
        
        self.play(Create(grid_lines), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line, dim second
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_GRID)
        )
        
        # Label Basis B - Addressing Issue 37
        self.place_in_area(label_basis, "E5", "F6", scale_factor=0.9)
        
        self.play(Write(label_basis))
        self.wait(2)
        
        # Clean up highlights for final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
