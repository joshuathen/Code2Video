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
            "For one point, we get one region.",
            "Two points give two regions.",
            "Three points result in four.",
            "Four points yield eight.",
            "Five points double to sixteen."
        ]
        self.setup_layout("Empirical Observation: The Trap of Induction", lecture_lines)
        
        # Colors for the lecture lines
        colors = ["#00FFFF", "#00FFFF", "#00FFFF", "#00FFFF", "#00FFFF"]
        
        def draw_circle_scenario(n):
            circle = Circle(radius=1.2, color=WHITE)
            if n == 0: return VGroup(circle)
            points = [
                (1.2 * np.cos(2 * PI * k / n + PI/2), 1.2 * np.sin(2 * PI * k / n + PI/2), 0)
                for k in range(n)
            ]
            dots = VGroup(*[Dot(p, color=WHITE) for p in points])
            lines = VGroup()
            for i in range(n):
                for j in range(i + 1, n):
                    lines.add(Line(points[i], points[j], color=GRAY))
            return VGroup(circle, dots, lines)

        # === Animation for Lecture Line 1 ===
        s1 = self.place_in_area(draw_circle_scenario(1), "B4", "E6", scale_factor=0.7)
        self.play(Create(s1))
        self.play(self.lecture[0].animate.set_color(colors[0]))

        # === Animation for Lecture Line 2 ===
        s2 = self.place_at_grid(draw_circle_scenario(2), "C4", scale_factor=0.6)
        self.play(FadeOut(s1), Create(s2))
        self.play(self.lecture[1].animate.set_color(colors[1]))

        # === Animation for Lecture Line 3 ===
        s3 = self.place_at_grid(draw_circle_scenario(3), "D4", scale_factor=0.6)
        self.play(FadeOut(s2), Create(s3))
        self.play(self.lecture[2].animate.set_color(colors[2]))

        # === Animation for Lecture Line 4 ===
        s4 = self.place_at_grid(draw_circle_scenario(4), "D4", scale_factor=0.6)
        self.play(FadeOut(s3), Create(s4))
        self.play(self.lecture[3].animate.set_color(colors[3]))

        # === Animation for Lecture Line 5 ===
        s5 = self.place_at_grid(draw_circle_scenario(5), "D4", scale_factor=0.6)
        self.play(FadeOut(s4), Create(s5))
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.wait(1)
