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
            "Almost all hypersphere volume resides near the surface.",
            "Imagine an orange with a massive, heavy peel.",
            "The inner fruit becomes mathematically negligible.",
            "Peel represents concentrated surface volume.",
            "Interior density fades as dimensions increase."
        ]
        self.setup_layout("The Crusty Shell: Concentration of Mass", lecture_lines)
        
        # Define elements
        circle_shell = Circle(radius=1.5, color=ORANGE, fill_opacity=0.3)
        inner_fruit = Circle(radius=0.5, color=YELLOW, fill_opacity=0.6)
        orange = VGroup(circle_shell, inner_fruit)
        peel_label = Text("Peel", font_size=16, color=ORANGE)
        fade_label = Text("Empty", font_size=16, color=GREY)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        self.place_in_area(orange, 'B4', 'E6', scale_factor=0.6)
        self.add(orange)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        self.place_at_grid(peel_label, 'B4', scale_factor=0.8)
        self.add(peel_label)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREY)
        self.place_at_grid(fade_label, 'D5', scale_factor=0.8)
        self.add(fade_label)
        self.wait(2)
