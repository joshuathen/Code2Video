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
        lecture_lines = [
            "Why does the cycloid win the race?",
            "Steep drops accelerate motion rapidly.",
            "Flat sections leverage that high velocity."
        ]
        self.setup_layout("The 'Brachistochrone' Principle in Action", lecture_lines)
        
        # Define paths
        cycloid = ParametricFunction(lambda t: np.array([t - np.sin(t), - (1 - np.cos(t)), 0]), t_range=[0, 2*PI], color=BLUE)
        line = Line(start=np.array([0, 0, 0]), end=np.array([2*PI, -2, 0]), color=RED)
        
        # Combine paths for easier placement
        paths = VGroup(cycloid, line)
        
        # Apply positioning fix from Issue 30 (B4-E6, 0.55 scale)
        self.place_in_area(paths, 'B4', 'E6', scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Create(cycloid), Create(line))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(Indicate(cycloid), run_time=2)
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(Flash(cycloid.get_end(), color=YELLOW, line_length=0.2, num_lines=12))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
