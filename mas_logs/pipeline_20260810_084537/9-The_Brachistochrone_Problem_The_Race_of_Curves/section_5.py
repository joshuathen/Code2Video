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
        lecture_lines = [
            "Nature often chooses the path of least time.",
            "The cycloid outperforms the straight path.",
            "Geometry reveals nature's secret efficiency."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # Visual Elements
        straight_line = Line(start=self.grid["B2"], end=self.grid["D5"], color=BLUE)
        
        # Cycloid approximation
        curve = ParametricFunction(
            lambda t: np.array([t - np.sin(t), - (1 - np.cos(t)), 0]),
            t_range=[0, 2*PI],
            color=YELLOW
        )
        # Apply fix for Issue 31 and 33: shifted to B3-D6
        self.place_in_area(curve, "B3", "D6", scale_factor=0.4)

        # Labels
        label_straight = Text("Straight", font_size=20, color=BLUE)
        label_cycloid = Text("Cycloid", font_size=20, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(straight_line), Write(label_straight.next_to(straight_line, UP, buff=0.1)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(curve), Write(label_cycloid.next_to(curve, DOWN, buff=0.1)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        final_text = Text("Nature optimizes time.", font_size=28, color=WHITE)
        # Apply fix for Issue 32 and 33: E3-F6
        self.place_in_area(final_text, "E3", "F6", scale_factor=0.8)
        self.play(FadeIn(final_text))
        self.wait(2)
