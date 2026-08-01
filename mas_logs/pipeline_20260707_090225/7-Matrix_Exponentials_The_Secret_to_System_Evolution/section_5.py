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
        # Initial Setup
        title = "Geometric Intuition: Rotations and Scaling"
        lines = [
            "Diagonal matrices result in pure scaling along axes.",
            "Skew-symmetric matrices create continuous rotations in space.",
            "The exponential maps local rates to global transformations."
        ]
        self.setup_layout(title, lines)

        # Utility to build a matrix manually (Avoiding LaTeX)
        def get_matrix_group(vals, label_text="A =", color=WHITE):
            m_vals = VGroup(*[Text(v, font_size=24, color=color) for v in vals]).arrange_in_grid(rows=2, cols=2, buff=0.4)
            l_bracket = Text("[", font_size=42, color=color).stretch_to_fit_height(m_vals.height + 0.2).next_to(m_vals, LEFT, buff=0.1)
            r_bracket = Text("]", font_size=42, color=color).stretch_to_fit_height(m_vals.height + 0.2).next_to(m_vals, RIGHT, buff=0.1)
            matrix = VGroup(m_vals, l_bracket, r_bracket)
            label = Text(label_text, font_size=24, color=color).next_to(matrix, LEFT, buff=0.2)
            return VGroup(label, matrix)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Matrix A (Diagonal)
        matrix_diag = get_matrix_group(["1", "0", "0", "0.5"], color=YELLOW)
        self.place_in_area(matrix_diag, "D2", "D4", scale_factor=0.8) # Issue 46: Area placement
        
        circle = Circle(radius=1.0, color=WHITE)
        self.place_in_area(circle, "B2", "B4", scale_factor=0.8)
        
        self.play(FadeIn(matrix_diag), Create(circle))
        self.wait(1)
        
        ellipse = circle.copy().stretch(0.5, dim=1)
        self.play(Transform(circle, ellipse), run_time=2)
        self.wait(1)

        # Cleanup for next transition
        self.play(FadeOut(matrix_diag), FadeOut(circle))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # Matrix A (Skew-symmetric)
        matrix_skew = get_matrix_group(["0", "-1", "1", "0"], color=BLUE)
        self.place_in_area(matrix_skew, "D2", "D4", scale_factor=0.8) # Issue 46: Area placement
        
        unit_circle = Circle(radius=1.0, color=WHITE, stroke_opacity=0.3)
        self.place_in_area(unit_circle, "B2", "B4", scale_factor=0.8)
        
        # Character 'Vector' (Arrow)
        vector = Arrow(start=unit_circle.get_center(), 
                       end=unit_circle.get_center() + RIGHT * unit_circle.width/2, 
                       color="#00FFFF", buff=0)
        
        self.play(FadeIn(matrix_skew), FadeIn(unit_circle), FadeIn(vector))
        
        # Rotate Vector
        self.play(Rotate(vector, angle=2*PI, about_point=unit_circle.get_center()), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Formula for Exponential (Maps local rates to global transformations)
        formula = Text("e^At = I + At + (At)^2/2! + ...", font_size=24, color=GREEN)
        self.place_in_area(formula, "B1", "B5", scale_factor=0.8) # Issue 44: Area placement
        
        evolution_text = Text("Evolution Operator", font_size=22, color=GREEN)
        self.place_in_area(evolution_text, "F2", "F4", scale_factor=0.8) # Issue 45: Area placement
        
        # Visualizing the path
        path = unit_circle.copy().set_color(GREEN).set_stroke(opacity=1.0, width=4)
        
        self.play(
            FadeIn(formula),
            FadeIn(evolution_text),
            Create(path),
            Rotate(vector, angle=2*PI, about_point=unit_circle.get_center()),
            run_time=4
        )
        self.wait(2)
