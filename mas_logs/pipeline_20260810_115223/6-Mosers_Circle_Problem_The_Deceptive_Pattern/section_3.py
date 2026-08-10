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
        self.setup_layout("The Trap: Testing the Hypothesis", [
            "The sequence 1, 2, 4, 8, 16 suggests 2^(n-1).",
            "Testing n equals six, we expect thirty-two.",
            "Carefully counting, we find only thirty-one regions."
        ])

        # === Animation for Lecture Line 1 ===
        # Display a prompt: 'Is the pattern 2^(n-1)?' in bold white. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        prompt = Text("Is the pattern 2^(n-1)?", font_size=32, color=WHITE)
        self.place_in_area(prompt, 'B2', 'B5')
        self.play(Write(prompt))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw the case for n=6, highlighting all chord intersections.
        circle = Circle(radius=1.5, color=BLUE)
        self.place_at_grid(circle, 'D2', scale_factor=1.2)
        self.play(Create(circle))
        
        points = [Dot(point=circle.point_at_angle(2 * PI * i / 6), color=YELLOW) for i in range(6)]
        self.add(*points)
        self.play(*[FadeIn(p) for p in points])
        
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Count the regions/Result 31 vs 32
        result_text = Text("Result: 31", font_size=40, color=GREEN)
        expected_text = Text("Expected: 32", font_size=30, color=RED)
        cross = Cross(expected_text, color=RED)
        
        self.place_at_grid(result_text, 'D4', scale_factor=0.9)
        self.place_at_grid(expected_text, 'E4', scale_factor=0.9)
        self.place_at_grid(cross, 'E4', scale_factor=1.0)
        
        self.play(Write(result_text))
        self.play(Write(expected_text), Create(cross))
        
        # Conclude that the intuitive pattern has failed. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        self.lecture[2].set_color(GREEN)
        self.wait(2)
