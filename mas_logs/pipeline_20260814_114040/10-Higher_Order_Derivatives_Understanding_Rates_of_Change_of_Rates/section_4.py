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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We can continue the derivative process further.",
            "Third derivative, f'''(x), represents jerk.",
            "Higher derivatives offer refined information about smoothness."
        ]
        self.setup_layout("Generalizing to Higher Orders (n-th derivative)", lecture_lines)
        
        # Define functions for visual
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        f_x = axes.plot(lambda x: 0.5 * x**3, color="#FFFFFF")
        f_prime = axes.plot(lambda x: 1.5 * x**2, color="#FFFFFF")
        
        f_double_prime = axes.plot(lambda x: 3 * x, color="#FFFF00")
        f_triple_prime = axes.plot(lambda x: 3, color="#FFFF00")
        
        n_th_deriv = MathTex(r"f^{(n)}(x) = \frac{d^n}{dx^n}f(x)", color="#FF00FF")

        # Layout group
        graphs = VGroup(axes, f_x, f_prime, f_double_prime, f_triple_prime)
        # Fix for issue 28: offset to avoid lecture overlap
        self.place_in_area(graphs, 'C3', 'F6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(graphs[0]), Create(graphs[1]), Create(graphs[2]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(Create(graphs[3]), Create(graphs[4]))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        # Fix for issue 29: adjust position for hierarchy
        self.place_at_grid(n_th_deriv, 'B2', scale_factor=0.9)
        self.play(Write(n_th_deriv))
        self.wait(2)
