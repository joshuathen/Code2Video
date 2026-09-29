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
            "Time is the sum of times in each medium.",
            "Distance uses the Pythagorean theorem for both.",
            "Set the derivative of time to zero.",
            "This finds the point of minimum time.",
            "Calculus reveals the path light must choose."
        ]
        self.setup_layout("Mathematical Derivation", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw a grid and interface
        interface = Line(start=[-2, 0, 0], end=[2, 0, 0], color=GRAY)
        self.place_at_grid(interface, 'D2', scale_factor=0.6)
        self.play(Create(interface))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Place two points, 'A' and 'B'
        point_a = Dot(color="#00FF00").move_to(self.grid['B4'])
        point_b = Dot(color="#00FF00").move_to(self.grid['E6'])
        label_a = Text("A", font_size=20, color="#00FF00").next_to(point_a, UP, buff=0.1)
        label_b = Text("B", font_size=20, color="#00FF00").next_to(point_b, DOWN, buff=0.1)
        self.play(FadeIn(point_a, label_a, point_b, label_b))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Animate path and label segments
        path = Line(point_a.get_center(), point_b.get_center(), color="#FFFF00")
        self.play(Create(path))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        # Label path segments
        d1_label = Text("d1", font_size=20, color="#00FFFF").move_to(self.grid['C4'])
        d2_label = Text("d2", font_size=20, color="#00FFFF").move_to(self.grid['D5'])
        label_group = VGroup(d1_label, d2_label)
        self.place_in_area(label_group, 'A4', 'B6', scale_factor=0.75)
        self.play(FadeIn(d1_label, d2_label))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        # Show formula
        formula = MathTex(r"T = \frac{d_1}{v_1} + \frac{d_2}{v_2}", font_size=30, color="#FF00FF")
        self.place_in_area(formula, 'E3', 'F5', scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[4].set_color("#FF00FF")
        
        self.wait(2)
