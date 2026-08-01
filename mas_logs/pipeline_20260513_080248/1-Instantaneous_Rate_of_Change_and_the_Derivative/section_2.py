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
        # Initialize Scene
        lecture_lines = [
            'Consider a curve with two distinct points plotted.',
            'A secant line connects these two points directly.',
            'Its slope represents the average rate of change.'
        ]
        self.setup_layout("Prerequisite: The Slope of a Secant Line", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#3498DB"))

        # Setup Graph Area
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 10, 2],
            x_length=4.5,
            y_length=4.0,
            axis_config={"include_tip": True, "font_size": 18}
        )
        self.place_in_area(axes, 'A1', 'E5')
        
        curve = axes.plot(lambda x: x**2, x_range=[0, 3.2], color="#3498DB")
        
        # Points A(1,1) and B(3,9)
        point_a_coord = axes.c2p(1, 1)
        point_b_coord = axes.c2p(3, 9)
        
        dot_a = Dot(point_a_coord, color=WHITE, radius=0.08)
        dot_b = Dot(point_b_coord, color=WHITE, radius=0.08)
        
        label_a = Text("A(1, 1)", font_size=16, color=WHITE).next_to(dot_a, DOWN + LEFT, buff=0.1)
        label_b = Text("B(3, 9)", font_size=16, color=WHITE).next_to(dot_b, LEFT, buff=0.1)

        self.play(Create(axes), Create(curve))
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(self.lecture[1].animate.set_color("#F1C40F"))

        # Yellow Secant Line
        # We extend the line slightly beyond the points for visual effect
        secant_line = Line(
            axes.c2p(0.5, -1), 
            axes.c2p(3.2, 9.8), 
            color="#F1C40F"
        )
        
        secant_label = Text("Secant Line", font_size=18, color="#F1C40F")
        self.place_at_grid(secant_label, "B5")
        # Numerical angle calculation for label rotation
        secant_label.rotate(np.arctan2(point_b_coord[1] - point_a_coord[1], point_b_coord[0] - point_a_coord[0]))

        self.play(Create(secant_line))
        self.play(FadeIn(secant_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color("#F1C40F"))

        # Slope Formula display
        formula_text = Text("Slope (m) = (y2 - y1) / (x2 - x1)", font_size=18, color=WHITE)
        self.place_in_area(formula_text, 'F1', 'F2', scale_factor=0.55)
        
        calc_text = Text("m = (9 - 1) / (3 - 1)", font_size=18, color=WHITE)
        self.place_in_area(calc_text, 'F3', 'F5', scale_factor=0.55)
        
        result_text = Text("m = 4", font_size=22, color="#F1C40F")
        self.place_at_grid(result_text, "F6", scale_factor=0.7)

        self.play(Write(formula_text))
        self.wait(0.5)
        self.play(Write(calc_text))
        self.wait(0.5)
        self.play(Write(result_text))
        self.wait(2)
