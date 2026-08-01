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

class Section1Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        """
        Standard layout setup for the teaching series.
        """
        # Set background color
        self.camera.background_color = "#000000"
        
        # Title Setup
        self.title_mobj = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        
        # Lecture content on the left side
        lecture_texts = [Text(line, font_size=24, color=WHITE) for line in lecture_lines]
        self.lecture_vgroup = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture_vgroup.to_edge(LEFT, buff=0.8)

        # Define a 3x3 grid system for the right side of the screen
        self.grid_points = {}
        rows = ["A", "B", "C"]
        cols = ["1", "2", "3"]
        for i, row_label in enumerate(rows):
            for j, col_label in enumerate(cols):
                # Positions objects in the right hemisphere of the coordinate plane
                x_val = 2.0 + (j * 2.0)
                y_val = 1.5 - (i * 1.5)
                self.grid_points[f"{row_label}{col_label}"] = np.array([x_val, y_val, 0])

    def get_grid_pos(self, key):
        return self.grid_points.get(key, ORIGIN)

    def construct(self):
        # Configuration
        title = "The Mystery of 0!"
        bullets = [
            "- Factorial Rule: n! = n * (n-1)!",
            "- Inverse Rule: (n-1)! = n! / n",
            "- Case n = 1: 0! = 1! / 1",
            "- Therefore: 0! = 1"
        ]
        
        # Initialize Layout
        self.setup_layout(title, bullets)
        
        # Display Title and Bullets
        self.play(Write(self.title_mobj))
        self.play(FadeIn(self.lecture_vgroup, shift=RIGHT))
        self.wait(1)

        # Mathematical Proof visualization using the grid
        # Replacing MathTex with Text to avoid the dependency on a local LaTeX installation
        step1 = Text("3! = 6", font_size=36).move_to(self.get_grid_pos("A1"))
        step2 = Text("2! = 6 / 3 = 2", font_size=36).move_to(self.get_grid_pos("A2"))
        step3 = Text("1! = 2 / 2 = 1", font_size=36).move_to(self.get_grid_pos("B1"))
        step4 = Text("0! = 1 / 1 = 1", font_size=40, color=YELLOW).move_to(self.get_grid_pos("B2"))

        # Animations
        self.play(Write(step1))
        self.wait(0.5)
        self.play(FadeIn(step2, shift=UP))
        self.wait(0.5)
        self.play(Write(step3))
        self.wait(0.5)
        
        # Highlighting the conclusion
        self.play(
            ReplacementTransform(step3.copy(), step4),
            run_time=1.5
        )
        
        # Final visual emphasis
        box = SurroundingRectangle(step4, color=BLUE, buff=0.2)
        self.play(Create(box))
        
        self.wait(3)
