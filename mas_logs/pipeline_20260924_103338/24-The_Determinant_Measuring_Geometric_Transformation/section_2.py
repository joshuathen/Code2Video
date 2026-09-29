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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Concept of Basis Vectors", [
            "Define basis vectors i and j.", 
            "Matrices are sets of transformation instructions.", 
            "Watch where i and j land."
        ])
        
        # Assets
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Setup Axes
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        self.add(axes)
        
        # Vectors
        i_vec = Vector(RIGHT, color="#33FF57")
        j_vec = Vector(UP, color="#33FF57")
        i_label = MathTex(r"\\hat{i}", color="#33FF57")
        j_label = MathTex(r"\\hat{j}", color="#33FF57")
        
        self.place_at_grid(i_label, 'B4', scale_factor=0.8)
        self.place_at_grid(j_label, 'C3', scale_factor=0.8)

        self.add(i_vec, j_vec, i_label, j_label, vector_icon)
        self.place_at_grid(vector_icon, 'A4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#33FF57"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF5733"))
        
        # Create transform effect
        new_i = np.array([2, 1, 0])
        new_j = np.array([0, 2, 0])
        
        i_vec_target = Vector(new_i, color="#FF5733")
        j_vec_target = Vector(new_j, color="#FF5733")
        
        triangle_shape = Polygon(ORIGIN, new_i, (new_i + new_j), new_j, 
                                 fill_opacity=0.3, color="#FF5733")
        self.place_in_area(triangle_shape, 'C4', 'E5', scale_factor=0.5)
        
        self.play(
            i_vec.animate.put_start_and_end_on(ORIGIN, axes.c2p(*new_i)),
            j_vec.animate.put_start_and_end_on(ORIGIN, axes.c2p(*new_j)),
            i_label.animate.set_color("#FF5733"),
            j_label.animate.set_color("#FF5733"),
            i_vec.animate.set_color("#FF5733"),
            j_vec.animate.set_color("#FF5733"),
            FadeIn(grid_icon)
        )
        self.place_at_grid(grid_icon, 'E2', scale_factor=0.5)
        
        self.play(Create(triangle_shape))
        self.wait(2)
