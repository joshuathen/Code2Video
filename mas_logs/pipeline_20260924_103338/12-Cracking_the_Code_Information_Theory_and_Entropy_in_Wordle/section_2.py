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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Rare outcomes surprise us, providing more information.",
            "Information is the negative log of probability.",
            "Frequent events carry very little surprise value."
        ]
        self.setup_layout("Defining Information: The 'Surprise' Factor", lecture_lines)
        
        # Elements
        event_icon = Circle(radius=0.5, color="#00FFFF", fill_opacity=0.5)
        label_x = Text("Event X", font_size=24, color="#00FFFF")
        event_group = VGroup(event_icon, label_x).arrange(DOWN)
        
        # Initial position setup (using B2-B4 area)
        self.place_in_area(event_group, 'B2', 'B4', scale_factor=0.5)
        
        formula = MathTex(r"I(x) = -\log_2(P(x))", color=WHITE)
        # Using E3 for formula to avoid clutter
        self.place_at_grid(formula, 'E3', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(event_group))
        self.lecture[0].set_color("#00FFFF")
        self.wait(4)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(event_group), FadeIn(formula))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(4)

        # === Animation for Lecture Line 3 ===
        self.play(formula.animate.set_color(YELLOW))
        self.lecture[2].set_color("#FF00FF")
        self.wait(4)
