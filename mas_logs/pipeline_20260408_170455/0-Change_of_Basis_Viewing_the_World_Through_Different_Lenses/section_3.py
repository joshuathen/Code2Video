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
        self.setup_layout("Introducing a New Basis", [
            'We can choose an entirely different set of basis vectors.', 
            'These new vectors, b1 and b2, form a new basis.', 
            'They define a tilted and stretched coordinate grid.', 
            'A point at [1, 1] here means one of each.', 
            'We reach it by following the new basis vectors.'
        ])

        # Grid area constants
        # Area B2-E6 centered at [3.5, -0.3]
        grid_center = np.array([3.5, -0.3, 0])
        unit_scale = 0.7  # 1 unit in coordinate space = 0.7 Manim units

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Standard Grid (Gray) - Resolved Issue 33
        std_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#555555", "stroke_opacity": 0.6},
            axis_config={"stroke_color": "#555555", "stroke_width": 2}
        )
        self.place_in_area(std_grid, "B2", "E6", scale_factor=unit_scale)
        
        # Unit vectors i (#00FF00) and j (#0000FF)
        i_vec = Vector([unit_scale, 0], color="#00FF00").shift(grid_center)
        j_vec = Vector([0, unit_scale], color="#0000FF").shift(grid_center)
        
        # Labels using Text with unicode
        i_label = Text("î", color="#00FF00", font_size=20)
        j_label = Text("ĵ", color="#0000FF", font_size=20)
        self.place_at_grid(i_label, "D5") 
        self.place_at_grid(j_label, "B3") 

        self.play(Create(std_grid), run_time=1.5)
        self.play(GrowArrow(i_vec), GrowArrow(j_vec), FadeIn(i_label), FadeIn(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # New basis vectors b1 = [2, 1] (#FF0000) and b2 = [-1, 1] (#FF8C00)
        b1_raw = np.array([2 * unit_scale, 1 * unit_scale, 0])
        b2_raw = np.array([-1 * unit_scale, 1 * unit_scale, 0])
        b1_vec = Vector(b1_raw, color="#FF0000").shift(grid_center)
        b2_vec = Vector(b2_raw, color="#FF8C00").shift(grid_center)
        
        b1_label = Text("b₁", color="#FF0000", font_size=20)
        b2_label = Text("b₂", color="#FF8C00", font_size=20)
        # Resolved Issue 34
        self.place_at_grid(b1_label, "B6", scale_factor=0.8) 
        self.place_at_grid(b2_label, "B2") 

        self.play(GrowArrow(b1_vec), GrowArrow(b2_vec), FadeIn(b1_label), FadeIn(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        basis_matrix = [[2, -1], [1, 1]]
        
        tilted_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#FF0000", "stroke_opacity": 0.4},
            axis_config={"stroke_color": "#FF0000", "stroke_opacity": 0.6}
        )
        tilted_grid.apply_matrix(basis_matrix)
        # Resolved Issue 32
        self.place_in_area(tilted_grid, "B2", "E6", scale_factor=unit_scale)

        self.play(
            ReplacementTransform(std_grid, tilted_grid),
            FadeOut(i_vec), FadeOut(j_vec), FadeOut(i_label), FadeOut(j_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Point [1, 1] in basis B. Standard coord = 1*b1 + 1*b2 = [1, 2]
        dot_pos = grid_center + b1_raw + b2_raw
        target_dot = Dot(dot_pos, color="#FFFFFF", radius=0.08)
        dot_label = Text("[1, 1]", font_size=18, color=WHITE).next_to(target_dot, UR, buff=0.1)
        
        self.play(Create(target_dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Draw arrows along b1 then b2 to reach the white dot
        path1 = Arrow(grid_center, b1_vec.get_end(), color="#FF0000", buff=0, stroke_width=4, tip_length=0.15)
        path2 = Arrow(b1_vec.get_end(), target_dot.get_center(), color="#FF8C00", buff=0, stroke_width=4, tip_length=0.15)
        
        self.play(Create(path1))
        self.play(Create(path2))
        self.wait(2)
