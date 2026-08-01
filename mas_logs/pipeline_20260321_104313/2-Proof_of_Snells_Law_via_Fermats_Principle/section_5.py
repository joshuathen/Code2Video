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

class Section5Scene(TeachingScene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset grid to the right half of the screen
                x = 1.5 + j * 0.8
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        title_text = "Section 5: Final Derivation of Snell's Law"
        lecture_lines = [
            "- Set the derivative of time to zero:",
            "- dT/dx = 0",
            "- (sin(theta_1) / v_1) - (sin(theta_2) / v_2) = 0",
            "- Rearrange the terms:",
            "- sin(theta_1) / v_1 = sin(theta_2) / v_2",
            "- Multiply by c to use refractive indices:",
            "- n_1 sin(theta_1) = n_2 sin(theta_2)"
        ]

        self.setup_layout(title_text, lecture_lines)

        # Fixed: Replaced MathTex with Text using Unicode characters to bypass FileNotFoundError for 'latex' executable.
        # This allows the scene to render in environments where a LaTeX distribution is not installed.
        formula = Text(
            "n₁ sin(θ₁) = n₂ sin(θ₂)",
            color=YELLOW,
            font_size=48
        )
        self.place_at_grid(formula, "C3", scale_factor=1.2)
        
        box = SurroundingRectangle(formula, color=BLUE, buff=0.3)
        
        # Animations
        # Note: self.title and self.lecture were added in setup_layout, 
        # but re-playing their creation/fade-in to follow original logic.
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.wait(1)
        self.play(Write(formula))
        self.play(Create(box))
        self.wait(3)
