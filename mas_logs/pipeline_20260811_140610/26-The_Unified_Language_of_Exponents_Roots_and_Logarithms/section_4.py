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
        self.setup_layout("The Transformation Triangle", [
            "Exponents, logs, roots are linked.",
            "They rearrange three fixed numbers.",
            "These are inverse operations."
        ])
        
        # Triangle nodes
        exp_node = Text("Exponential\n(b^x = y)", font_size=24, color="#00FFFF")
        log_node = Text("Logarithmic\n(log_b y = x)", font_size=24, color="#00FFFF")
        root_node = Text("Roots/Inverse\n(y^(1/x) = b)", font_size=24, color="#00FFFF")
        
        # Optimized positions based on constraints (columns 4-6)
        self.place_at_grid(exp_node, 'B5', scale_factor=0.8)
        self.place_at_grid(log_node, 'D4', scale_factor=0.8)
        self.place_at_grid(root_node, 'D6', scale_factor=0.8)

        tri = Polygon(
            self.grid['B5'], self.grid['D4'], self.grid['D6'],
            color=WHITE
        )
        
        # Arrows 
        arrow1 = Arrow(self.grid['B5'], self.grid['D4'], color=YELLOW, buff=0.5)
        arrow2 = Arrow(self.grid['D4'], self.grid['D6'], color=YELLOW, buff=0.5)
        arrow3 = Arrow(self.grid['D6'], self.grid['B5'], color=YELLOW, buff=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Create(tri), FadeIn(exp_node), FadeIn(log_node), FadeIn(root_node))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(Create(arrow1), Create(arrow2), Create(arrow3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(Rotate(VGroup(tri, exp_node, log_node, root_node, arrow1, arrow2, arrow3), angle=PI/3, about_point=np.array([2.5, 0, 0])))
        self.wait(1)
