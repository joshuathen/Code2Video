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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with updated 3-line script
        self.setup_layout(
            "The Zero Case: Collapsing Dimensions", 
            [
                'Some transformations land both basis vectors on one line.', 
                'This collapses the entire plane, losing all area.', 
                'A zero determinant means the transformation is not invertible.'
            ]
        )
        
        # Create Coordinate system mobjects
        grid_lines = VGroup()
        for x in range(-2, 3):
            grid_lines.add(Line([x, -2, 0], [x, 2, 0], stroke_width=1, color=GRAY))
        for y in range(-2, 3):
            grid_lines.add(Line([-2, y, 0], [2, y, 0], stroke_width=1, color=GRAY))
        
        i_hat = Arrow([0, 0, 0], [1, 0, 0], buff=0, color="#00FF00")
        j_hat = Arrow([0, 0, 0], [0, 1, 0], buff=0, color="#00FFFF")
        basis_group = VGroup(grid_lines, i_hat, j_hat)
        
        # Place the grid system (Issue 49: scale_factor=0.7)
        self.place_in_area(basis_group, 'A1', 'F6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Line 1: 'Some transformations land both basis vectors on one line.'
        self.lecture[0].set_color("#ADD8E6") # Light Blue
        self.play(Create(grid_lines), GrowArrow(i_hat), GrowArrow(j_hat))
        
        # Vectors move to (1,1). We use the visual scaling factor (0.7) to match the group.
        start_pos = i_hat.get_start()
        vec = np.array([1, 1, 0]) * 0.7
        target_i = Arrow(start_pos, start_pos + vec, buff=0, color="#00FF00")
        target_j = Arrow(start_pos, start_pos + vec, buff=0, color="#00FFFF")
        
        self.play(Transform(i_hat, target_i), Transform(j_hat, target_j), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: 'This collapses the entire plane, losing all area.'
        self.lecture[1].set_color("#FFFF00") # Yellow
        
        # Collapse grid lines relative to the visual origin
        origin = basis_group.get_center()
        collapsed_lines = VGroup()
        for line in grid_lines:
            p1_rel = line.get_start() - origin
            p2_rel = line.get_end() - origin
            
            # Map point (x,y) to (x+y, x+y) relative to center
            # This represents both basis vectors becoming (1,1)
            new_p1 = origin + np.array([p1_rel[0] + p1_rel[1], p1_rel[0] + p1_rel[1], 0])
            new_p2 = origin + np.array([p2_rel[0] + p2_rel[1], p2_rel[0] + p2_rel[1], 0])
            
            collapsed_lines.add(Line(new_p1, new_p2, stroke_width=2, color=GRAY_B))

        self.play(Transform(grid_lines, collapsed_lines), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: 'A zero determinant means the transformation is not invertible.'
        self.lecture[2].set_color("#FFA500") # Orange
        
        # Labels (Issue 47 & 48 fixes)
        txt_area = Text("Area = 0", font_size=28, color="#FF0000")
        txt_inv = Text("Not Invertible", font_size=28, color="#FF0000")
        
        self.place_at_grid(txt_area, 'B6', scale_factor=0.9)
        self.place_at_grid(txt_inv, 'E6', scale_factor=0.9)
        
        self.play(FadeIn(txt_area), FadeIn(txt_inv))
        self.wait(2)
