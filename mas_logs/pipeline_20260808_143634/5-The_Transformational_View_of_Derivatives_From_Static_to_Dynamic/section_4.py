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

class Section4Scene(TeachingScene, MovingCameraScene):
    def construct(self):
        lecture_lines = [
            "Curves look linear when zoomed in.",
            "Derivative provides local linear approximation.",
            "Use linear math for complex paths.",
            "Predict position using velocity vectors.",
            "Motion behaves linearly at any instant."
        ]
        self.setup_layout("Application: The Derivative as a Local Linearization", lecture_lines)
        
        # Create objects
        curve = FunctionGraph(lambda x: 0.5 * np.sin(3*x) + 0.2 * x**2, x_range=[-2, 2], color=BLUE)
        self.place_in_area(curve, 'B3', 'E6', scale_factor=0.6)
        
        dot = Dot(color=YELLOW)
        dot.move_to(curve.point_from_proportion(0.5))
        self.place_at_grid(dot, 'D4', scale_factor=0.4)
        
        # Tangent line (initially invisible)
        tangent_line = Line(start=LEFT, end=RIGHT, color=RED)
        self.place_at_grid(tangent_line, 'D4', scale_factor=0.5)
        
        self.play(Create(curve), FadeIn(dot))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(self.camera.frame.animate.set(zoom=2).move_to(dot.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        vector = Arrow(start=dot.get_center(), end=dot.get_center() + RIGHT*0.5 + UP*0.2, color=ORANGE)
        self.play(GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)
