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
        # 1. Setup Layout
        title_text = "Normalizing the Space (The Posterior)"
        lecture_lines = [
            "Let's combine the remaining areas to form a new space.",
            "This total area represents all possible glinting events.",
            "We normalize this space back into a unit square.",
            "The gold region's relative size gives the new probability.",
            "This updated value is called the posterior probability."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        GOLD_CLR = "#FFD700"
        GREY_CLR = "#888888"

        # Create components
        # Proportional sizes: Gold is 0.08, Other is 0.18.
        # We start with them at a visible scale before normalizing.
        gold_rect = Rectangle(width=0.8, height=2.0, fill_color=GOLD_CLR, fill_opacity=0.8, stroke_width=2)
        grey_rect = Rectangle(width=1.8, height=2.0, fill_color=GREY_CLR, fill_opacity=0.5, stroke_width=2)
        grey_rect.next_to(gold_rect, RIGHT, buff=0)
