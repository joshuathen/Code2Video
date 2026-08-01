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

class Section4Scene(Scene):
    def construct(self):
        # Configuration
        self.camera.background_color = "#000000"

        # Title
        title = Text("The Mystery of 0! = 1", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.5)

        # Lecture Bullets (Left Side)
        lecture_lines = [
            "- Pattern Analysis",
            "- Combinatorial Logic",
            "- Empty Product Rule",
            "- Gamma Function Connection"
        ]
        
        lecture_vgroup = VGroup(*[
            Text(line, font_size=24, color=WHITE) for line in lecture_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        lecture_vgroup.to_edge(LEFT, buff=1.0).shift(DOWN * 0.5)

        # Right Side - Mathematical Logic
        # Replacing MathTex with Text to resolve FileNotFoundError: 'latex'
        logic_group = VGroup(
            Text("4! = 24", font_size=34),
            Text("3! = 24 / 4 = 6", font_size=34),
            Text("2! = 6 / 3 = 2", font_size=34),
            Text("1! = 2 / 2 = 1", font_size=34),
            Text("0! = 1 / 1 = 1", color=YELLOW, font_size=42)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        logic_group.to_edge(RIGHT, buff=1.5).shift(DOWN * 0.5)

        # Animations
        self.play(Write(title))
        self.wait(0.5)

        self.play(
            FadeIn(lecture_vgroup, shift=RIGHT),
            run_time=1.5
        )
        self.wait(1)

        # Sequential appearance of the logic
        for i, math_line in enumerate(logic_group):
            if i == len(logic_group) - 1:
                # Highlight the 0! line
                self.play(Create(SurroundingRectangle(math_line, color=YELLOW, buff=0.1)))
                self.play(Write(math_line), run_time=1.2)
            else:
                self.play(Write(math_line), run_time=0.8)
            self.wait(0.3)

        # Visual Grid/Circle for flair (Right background)
        bg_circle = Circle(radius=2.5, color=BLUE_E, stroke_opacity=0.3).move_to(logic_group.get_center())
        self.play(FadeIn(bg_circle))
        
        self.wait(3)
