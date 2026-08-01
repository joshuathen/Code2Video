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
        # Setup layout
        title_text = "Visualizing the Slope on a Curve"
        lecture_lines = [
            "The derivative represents the tangent slope at any point.",
            "Consider the point three comma four on the circle.",
            "Plug these values into our derivative formula.",
            "The resulting slope is negative three-fourths.",
            "The tangent line perfectly matches this calculated slope."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        circle = Circle(radius=1.4, color="#FFFFFF")
        self.place_in_area(circle, 'B2', 'E5')
        
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        circle_label = Text("x² + y² = 25", font_size=24, color="#FFFFFF")
        self.place_at_grid(circle_label, 'A2')
        
        self.play(Create(circle), Write(circle_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        tl_pos = self.grid['B2']
        br_pos = self.grid['E5']
        center = (tl_pos + br_pos) / 2
        radius = 1.4
        
        dot_pos = center + radius * np.array([0.6, 0.8, 0])
        dot = Dot(point=dot_pos, color="#00FF00", radius=0.08)
        
        dot_label = Text("(3, 4)", font_size=24, color="#00FF00")
        self.place_at_grid(dot_label, 'D2') 

        self.play(FadeIn(dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        formula = Text("dy/dx = -x/y", font_size=24, color=WHITE)
        self.place_at_grid(formula, 'A5')
        
        formula_sub = Text("dy/dx = -3/4", font_size=24, color=WHITE)
        self.place_at_grid(formula_sub, 'A5')
        
        self.play(Write(formula))
        self.wait(1)
        self.play(Transform(formula, formula_sub))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        slope_val = Text("m = -3/4", font_size=24, color="#FFFF00")
        self.place_at_grid(slope_val, 'E3')
        
        self.play(FadeIn(slope_val))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        tan_dir = np.array([0.8, -0.6, 0])
        tangent_line = Line(
            dot_pos - tan_dir * 1.5,
            dot_pos + tan_dir * 1.5,
            color="#FFD700",
            stroke_width=4
        )
        
        self.play(Create(tangent_line))
        self.wait(2)