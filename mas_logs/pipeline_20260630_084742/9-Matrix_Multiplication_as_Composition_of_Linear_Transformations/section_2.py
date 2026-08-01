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
        # Initial layout setup
        title = "Prerequisite Check: The Basis Vectors"
        lines = [
            "We track transformation using unit basis vectors.",
            "Watch i-hat and j-hat land at new coordinates.",
            "These new locations form the matrix columns."
        ]
        self.setup_layout(title, lines)

        # Coordinate System Setup
        axes = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": False}
        )
        # Fix for Issue 31 and 33: Move axes away from lecture notes and utilize full width
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.9)
        
        # Basis Vectors
        i_color = "#FF0000"
        j_color = "#00FF00"
        
        # Helper to create vector and label relative to axes
        def get_basis_vec(coords, color, label_text):
            vec = Vector(axes.c2p(*coords), color=color, buff=0)
            label = Text(label_text, color=color, font_size=24)
            label.next_to(vec.get_end(), direction=vec.get_vector(), buff=0.1)
            return VGroup(vec, label)

        i_vec_grp = get_basis_vec([1, 0], i_color, "i")
        j_vec_grp = get_basis_vec([0, 1], j_color, "j")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes))
        self.play(GrowArrow(i_vec_grp[0]), Write(i_vec_grp[1]))
        self.play(GrowArrow(j_vec_grp[0]), Write(j_vec_grp[1]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Define targets for 90-degree rotation
        i_target_coords = [0, 1]
        j_target_coords = [-1, 0]

        new_i_vec = Vector(axes.c2p(*i_target_coords), color=i_color, buff=0)
        new_j_vec = Vector(axes.c2p(*j_target_coords), color=j_color, buff=0)
        
        new_i_label = Text("i'", color=i_color, font_size=24)
        new_i_label.next_to(new_i_vec.get_end(), UP, buff=0.1)
        
        new_j_label = Text("j'", color=j_color, font_size=24)
        new_j_label.next_to(new_j_vec.get_end(), LEFT, buff=0.1)

        self.play(
            Transform(i_vec_grp[0], new_i_vec),
            Transform(i_vec_grp[1], new_i_label),
            Transform(j_vec_grp[0], new_j_vec),
            Transform(j_vec_grp[1], new_j_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Create matrix manually using Text to avoid LaTeX dependency (Matrix class uses MathTex for brackets)
        # Row 1: 0, -1 | Row 2: 1, 0
        m_elements = VGroup(
            Text("0", font_size=30, color=i_color), Text("-1", font_size=30, color=j_color),
            Text("1", font_size=30, color=i_color), Text("0", font_size=30, color=j_color)
        ).arrange_in_grid(rows=2, cols=2, buff=0.5)
        
        l_bracket = Text("[", font_size=60).next_to(m_elements, LEFT, buff=0.1)
        r_bracket = Text("]", font_size=60).next_to(m_elements, RIGHT, buff=0.1)
        
        matrix_obj = VGroup(l_bracket, m_elements, r_bracket)
        
        # Fix for Issue 32: Position matrix further right to avoid crowding lecture notes
        self.place_at_grid(matrix_obj, 'A5', scale_factor=0.8)
        
        self.play(FadeIn(matrix_obj, shift=DOWN))
        self.wait(2)
        
        # Cleanup highlight
        self.lecture[2].set_color(WHITE)
        self.wait(1)
