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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'This formula converts basis-B coordinates into standard ones.',
            "Consider a vector at position one-two in the Drone's grid.",
            'Multiplying by P computes the equivalent standard coordinates.',
            "The result shows the mouse's position for the Owl.",
            'The grid shifts, but the physical location remains fixed.'
        ]
        self.setup_layout("Mathematical Formula & Graphical Walkthrough", lecture_lines)

        # Define Matrix P values from example
        p_val = np.array([[2, -1], [1, 1]])

        # Helper to create a matrix using Text
        def create_text_matrix(values, color=WHITE):
            rows = VGroup(*[
                VGroup(*[Text(str(v), font_size=24, color=color) for v in row]).arrange(RIGHT, buff=0.5)
                for row in values
            ]).arrange(DOWN, buff=0.4)
            l_br = Text("[", font_size=42, color=color).stretch_to_fit_height(rows.height + 0.2).next_to(rows, LEFT, buff=0.2)
            r_br = Text("]", font_size=42, color=color).stretch_to_fit_height(rows.height + 0.2).next_to(rows, RIGHT, buff=0.2)
            return VGroup(l_br, rows, r_br)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        formula = Text("[x]_std = P * [x]_B", color=WHITE, font_size=32)
        # Fix Issue 42: scale factor 0.9 to be consistent with calc_group
        self.place_in_area(formula, "A1", "A6", scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Basis B visualization setup (Skewed Grid)
        skewed_grid = NumberPlane(
            x_range=[-4, 4], y_range=[-4, 4],
            background_line_style={"stroke_color": "#555555", "stroke_width": 2},
            axis_config={"stroke_color": "#555555"}
        ).apply_matrix(p_val)
        
        # Vector at [1, 2] in basis B. Standard coordinates = P @ [1, 2] = [0, 3]
        vec_target = np.array([0, 3, 0])
        vector = Vector(vec_target, color="#FFD700")
        dot = Dot(vec_target, color="#FFD700")
        mouse_label = Text("Mouse (Static Point)", font_size=20, color="#FFD700").next_to(dot, UR, buff=0.1)
        
        # Fix Issue 41: Group point/vector/grid and position in area D1-F6
        viz_group = VGroup(skewed_grid, vector, dot, mouse_label)
        self.place_in_area(viz_group, "D1", "F6", scale_factor=0.4)
        
        self.play(Create(skewed_grid))
        self.play(GrowArrow(vector), FadeIn(dot), Write(mouse_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        p_matrix = create_text_matrix(p_val.tolist())
        v_b_matrix = create_text_matrix([[1], [2]])
        equals_sign = Text("=", font_size=32)
        v_std_matrix = create_text_matrix([[0], [3]])
        
        calc_group = VGroup(p_matrix, v_b_matrix, equals_sign, v_std_matrix).arrange(RIGHT, buff=0.3)
        self.place_in_area(calc_group, "B1", "C6", scale_factor=0.9)
        
        self.play(FadeIn(p_matrix), FadeIn(v_b_matrix))
        self.play(Write(equals_sign))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(FadeIn(v_std_matrix))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Standard Grid Morphing
        std_grid = NumberPlane(
            x_range=[-4, 4], y_range=[-4, 4],
            background_line_style={"stroke_color": "#444444", "stroke_width": 2},
            axis_config={"stroke_color": "#444444"}
        )
        # Position standard grid same as the skewed grid assembly
        tl_pos = self.grid["D1"]
        br_pos = self.grid["F6"]
        center = (tl_pos + br_pos) / 2
        std_grid.scale(0.4).move_to(center)
        
        # Morph grid while vector tip stays stationary at [0, 3] in standard space
        self.play(ReplacementTransform(skewed_grid, std_grid))
        self.wait(2)
