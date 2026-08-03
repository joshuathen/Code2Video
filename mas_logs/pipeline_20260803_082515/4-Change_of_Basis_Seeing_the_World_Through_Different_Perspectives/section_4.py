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
        # Setup Layout
        title = "The Translation Problem"
        lines = [
            "How do we translate Pixel's coordinates to ours?",
            "We follow his tilted arrows on our standard grid.",
            "This calculation bridges the two different perspectives."
        ]
        self.setup_layout(title, lines)

        # Colors
        V1_COLOR = "#FFFF00"  # Yellow (Basis vector 1)
        V2_COLOR = "#00FFFF"  # Cyan (Basis vector 2)
        VECTOR_COLOR = "#FFFFFF" # White (Pixel's vector)
        RESULT_COLOR = "#00FF00" # Green (Standard coordinates result)
        SKEWED_GRID_COLOR = "#555555" # Dimmed skewed grid
        STD_GRID_COLOR = "#333333" # Dimmed standard grid

        # Basis vectors defined in Section 3
        v1 = np.array([2, 1, 0])
        v2 = np.array([-1, 1, 0])
        
        # 1. Coordinate System Mobjects
        # Standard Grid
        std_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            background_line_style={"stroke_color": STD_GRID_COLOR, "stroke_opacity": 0.5}
        ).set_z_index(-1)
        
        # Skewed Grid (Pixel's basis)
        skewed_grid = NumberPlane(
            x_range=[-1, 2.5, 1],
            y_range=[-1, 2.5, 1],
            background_line_style={"stroke_color": SKEWED_GRID_COLOR, "stroke_opacity": 0.8}
        )
        # Columns of matrix are the basis vectors v1 and v2
        matrix = [[2, -1], [1, 1]]
        skewed_grid.apply_matrix(matrix)
        
        # Main Vector in Pixel's basis: [2, 1]_B
        # This maps to 2*v1 + 1*v2 = [3, 3] in standard coordinates
        pixel_point = 2 * v1 + 1 * v2
        main_vector = Arrow(ORIGIN, pixel_point, buff=0, color=VECTOR_COLOR)
        label_b = MathTex(r"[2, 1]_B", color=VECTOR_COLOR).scale(0.8)
        label_b.next_to(main_vector.get_end(), UR, buff=0.1)
        
        # Chained vectors for visual addition: 2*v1 then 1*v2
        arrow_v1_1 = Arrow(ORIGIN, v1, buff=0, color=V1_COLOR, stroke_width=4)
        arrow_v1_2 = Arrow(v1, 2*v1, buff=0, color=V1_COLOR, stroke_width=4)
        arrow_v2_add = Arrow(2*v1, 2*v1 + v2, buff=0, color=V2_COLOR, stroke_width=4)
        path_arrows = VGroup(arrow_v1_1, arrow_v1_2, arrow_v2_add)
        
        label_2v1 = MathTex("2 \\cdot v_1", color=V1_COLOR).scale(0.7)
        label_2v1.next_to(arrow_v1_2.get_center(), DR, buff=0.1)
        label_v2 = MathTex("1 \\cdot v_2", color=V2_COLOR).scale(0.7)
        label_v2.next_to(arrow_v2_add.get_center(), UL, buff=0.1)
        
        # Result coordinates label
        res_point = Dot(pixel_point, color=RESULT_COLOR)
        label_std = MathTex("(3, 3)", color=RESULT_COLOR).scale(0.9)
        label_std.next_to(res_point, RIGHT, buff=0.2)
        
        # Positioning Logic
        # Focus on the area where the transformation occurs (approx origin to (4,3))
        focus_center = np.array([1.5, 1.5, 0])
        all_visuals = VGroup(
            std_grid, skewed_grid, main_vector, label_b, 
            path_arrows, label_2v1, label_v2, res_point, label_std
        )
        all_visuals.shift(-focus_center)
        # Resolved issues 41, 42, 43 by adjusting area to B2:F6 and scale to 0.4
        self.place_in_area(all_visuals, 'B2', 'F6', scale_factor=0.4)
        
        # Initially hide components for progressive reveal
        std_grid.set_opacity(0)
        path_arrows.set_opacity(0)
        label_2v1.set_opacity(0)
        label_v2.set_opacity(0)
        res_point.set_opacity(0)
        label_std.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        # "How do we translate Pixel's coordinates to ours?"
        self.lecture[0].set_color(WHITE)
        self.play(Create(skewed_grid), run_time=1)
        self.play(GrowArrow(main_vector), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We follow his tilted arrows on our standard grid."
        self.play(
            self.lecture[0].animate.set_color(GRAY), 
            self.lecture[1].animate.set_color(V1_COLOR)
        )
        self.play(
            Succession(
                GrowArrow(arrow_v1_1),
                GrowArrow(arrow_v1_2),
                GrowArrow(arrow_v2_add)
            ),
            Write(label_2v1),
            Write(label_v2),
            path_arrows.animate.set_opacity(1),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This calculation bridges the two different perspectives."
        self.play(
            self.lecture[1].animate.set_color(GRAY), 
            self.lecture[2].animate.set_color(RESULT_COLOR)
        )
        self.play(
            std_grid.animate.set_opacity(1),
            FadeIn(res_point),
            Write(label_std)
        )
        self.play(main_vector.animate.set_color(RESULT_COLOR))
        self.wait(2)
