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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vectors are arrows in a 2D plane.", "Point (x, y) uses basis vectors i-hat and j-hat.", "Any vector is a linear combination of basis vectors."]
        self.setup_layout("Prerequisites: Vectors as Coordinates", lecture_lines)

        # Prepare objects
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        
        i_hat = Vector([1, 0], color="#FF0000")
        j_hat = Vector([0, 1], color="#00FF00")
        i_label = MathTex(r"\\hat{i}", color="#FF0000", font_size=24).next_to(i_hat.get_end(), RIGHT)
        j_label = MathTex(r"\\hat{j}", color="#00FF00", font_size=24).next_to(j_hat.get_end(), UP)
        
        v = Vector([2, 3], color="#FFFF00")
        v_label = MathTex(r"\\vec{v} = 2\\hat{i} + 3\\hat{j}", font_size=28)
        
        coord_group = VGroup(grid_asset, axes, i_hat, j_hat, i_label, j_label)

        # === Animation for Lecture Line 1 ===
        # Position coord_plane with vector using place_in_area as requested
        self.place_in_area(coord_group, 'A4', 'F6', scale_factor=0.6)
        self.play(FadeIn(grid_asset), FadeIn(axes))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(i_hat), Write(i_label), FadeIn(j_hat), Write(j_label))
        self.lecture[1].set_color("#FF0000") 

        # === Animation for Lecture Line 3 ===
        # Formula at E3, grid area A3-F6
        self.place_at_grid(v_label, 'E3', scale_factor=0.9)
        self.play(FadeIn(v), Write(v_label))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
