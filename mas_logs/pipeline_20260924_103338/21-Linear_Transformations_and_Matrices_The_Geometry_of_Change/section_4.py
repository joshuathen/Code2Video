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
            "Chaining transformations is equivalent to matrix multiplication.",
            "Order matters: rotations before scaling differ by result.",
            "Matrix multiplication is generally non-commutative."
        ]
        self.setup_layout("Composition: Chaining Transformations", lecture_lines)
        
        # Assets
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Grid visual
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_opacity": 0.5}).scale(0.5)
        self.place_in_area(grid, 'C4', 'F6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        
        matrix_a = MathTex(r"A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}").scale(0.6) # Rotation
        matrix_b = MathTex(r"B = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}").scale(0.6) # Scaling
        
        self.place_at_grid(matrix_a, 'A4', scale_factor=0.6)
        self.place_at_grid(matrix_b, 'A6', scale_factor=0.6)
        
        self.play(FadeIn(grid), Write(matrix_a), Write(matrix_b))
        
        # Add Airplane
        self.place_at_grid(airplane, 'B5', scale_factor=0.3)
        self.play(FadeIn(airplane))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Animate Rotation then Scale
        self.play(grid.animate.apply_matrix([[0, -1], [1, 0]]), run_time=1)
        self.play(grid.animate.apply_matrix([[2, 0], [0, 1]]), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        
        composite_label = MathTex(r"C = B \times A").scale(0.8)
        self.place_at_grid(composite_label, 'C5', scale_factor=0.8)
        
        self.play(Write(composite_label))
        
        # Highlight order with car
        car.set_color("#FF4500")
        self.place_at_grid(car, 'E5', scale_factor=0.4)
        self.play(FadeIn(car))
        
        self.wait(2)
