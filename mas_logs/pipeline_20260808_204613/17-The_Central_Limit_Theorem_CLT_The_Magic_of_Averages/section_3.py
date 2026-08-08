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
        lecture_lines = [
            "The CLT links sample means to normalcy.",
            "Larger samples lead to a normal distribution.",
            "It works regardless of the original population shape.",
            "Sample size increases converge toward the bell.",
            "This magic occurs reliably with large N."
        ]
        self.setup_layout("The Mechanism: The Convergence Principle", lecture_lines)
        
        # Create Axes for the visualization
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 1, 0.2],
            axis_config={"include_tip": False},
            x_length=5,
            y_length=3
        )
        # Applying the fix from issue 29 as it incorporates 27/28 feedback logically
        self.place_in_area(axes, 'C3', 'D4', scale_factor=0.35)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        
        # Simple animation: morphing histogram (conceptual representation)
        hist = VGroup(*[Rectangle(height=np.random.rand()*2, width=0.5, color=WHITE).set_fill(WHITE, 0.5) for _ in range(6)])
        hist.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        # Applying the fix from issue 29
        self.place_in_area(hist, 'E3', 'F4', scale_factor=0.35)
        self.add(hist)
        
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)
