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
    def construct(self):
        self.setup_layout("Application & Wrap-up", [
            "In real systems, we sum many noise sources.", 
            "This rule simplifies complex error analysis.", 
            "We treat the total noise as one Gaussian."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Representing many noise sources
        dots = VGroup(*[Dot(color=BLUE) for _ in range(10)])
        dots.arrange_in_grid(2, 5, buff=0.3)
        self.place_at_grid(dots, 'B4', scale_factor=0.8)
        self.play(FadeIn(dots))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Symbolizing simplification (arrow + Gaussian)
        arrow = Arrow(start=UP, end=DOWN, color=YELLOW)
        gaussian = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2], color=YELLOW)
        
        group = VGroup(arrow, gaussian).arrange(DOWN)
        self.place_at_grid(group, 'C4', scale_factor=0.7)
        self.play(Create(arrow), Create(gaussian))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Final Gaussian
        final_gaussian = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-3, 3], color=GREEN)
        self.place_in_area(final_gaussian, 'D4', 'F6', scale_factor=0.9)
        self.play(ReplacementTransform(group, final_gaussian))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
