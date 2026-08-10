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
        lecture_lines = ["High-dimensional data behaves like hyperspheres.", "Machine learning maps vectors on spheres.", "We use cosine similarity to measure distance."]
        self.setup_layout("Real-World Application: High-Dimensional Data", lecture_lines)
        
        # Create visual elements
        # 1. Dataset Matrix & Globe
        matrix = Matrix([[1, 0.2], [0.1, 0.9], [0.8, 0.3]], h_buff=0.8, v_buff=0.8)
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        data_viz = VGroup(matrix, globe).arrange(DOWN)
        
        # 2. Vector Field / Manifold
        dots = VGroup(*[Dot(color=BLUE) for _ in range(10)])
        dots.arrange_in_grid(2, 5, buff=0.2)
        
        # 3. Compass and Formula
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        formula = MathTex(r"\frac{A \cdot B}{\|A\|\|B\|}")
        similarity_viz = VGroup(compass, formula).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(data_viz, 'A4', 'C6', scale_factor=0.6)
        self.play(Create(data_viz))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(dots, 'D1', 'F3', scale_factor=0.6)
        self.play(Create(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(similarity_viz, 'B4', scale_factor=1.0)
        self.play(FadeIn(similarity_viz))
        self.wait(2)
