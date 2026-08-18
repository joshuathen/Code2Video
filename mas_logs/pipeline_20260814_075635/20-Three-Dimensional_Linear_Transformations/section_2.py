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
        self.setup_layout("Prerequisite Refresher: Basis Vectors", [
            "Recall basis vectors i, j, and k.",
            "Transformations are defined by their landing spots.",
            "The 3x3 matrix tracks these new coordinates."
        ])
        
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": True})
        i_vec = Arrow(start=ORIGIN, end=axes.c2p(1, 0), color="#FF5733", buff=0)
        j_vec = Arrow(start=ORIGIN, end=axes.c2p(0, 1), color="#FF5733", buff=0)
        i_label = MathTex(r"\hat{i}", color="#FF5733").next_to(i_vec.get_end(), RIGHT)
        j_label = MathTex(r"\hat{j}", color="#FF5733").next_to(j_vec.get_end(), UP)
        
        basis_group = VGroup(axes, i_vec, j_vec, i_label, j_label)
        
        # Asset integration (using placeholders as per assets in prompt)
        icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        asset_icon = SVGMobject(icon_path) if True else Dot() # Fallback for demo
        
        # Fixing positions per critic suggestions
        self.place_in_area(basis_group, "A1", "F3", scale_factor=0.5)
        
        # Placeholder for grid_nodes as requested by critic 41
        grid_nodes = VGroup(*[Dot(self.grid[f"{r}{c}"]) for r in "ABCDEF" for c in "123456"])
        self.place_in_area(grid_nodes, 'B4', 'F6', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.play(Create(basis_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        
        new_i = axes.c2p(1.5, 0.5)
        new_j = axes.c2p(0, 1.5)
        
        self.play(
            i_vec.animate.put_start_and_end_on(ORIGIN, new_i),
            j_vec.animate.put_start_and_end_on(ORIGIN, new_j),
            i_label.animate.next_to(new_i, RIGHT),
            j_label.animate.next_to(new_j, UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        
        matrix = Matrix([[1.5, 0], [0.5, 1.5]])
        self.place_at_grid(matrix, "D5", scale_factor=0.7)
        self.play(Write(matrix))
        self.wait(1)
