from manim import *
import numpy as np

# Set configuration to prevent FileNotFoundError during TeX cleanup race conditions
config.no_latex_cleanup = True

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
        self.setup_layout("The Translation Matrix (The 'Dictionary')", [
            "Alice needs to understand Bob's unique steps.",
            "Bob's basis vectors are columns in a matrix.",
            "This matrix P acts as a translation dictionary."
        ])

        # Colors
        COLOR_B1 = BLUE_B
        COLOR_B2 = GREEN_B
        COLOR_P = "#FFFF00" # Yellow
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_B1))
        
        # Asset: Alice Icon
        alice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/alice.svg")
        self.place_at_grid(alice_icon, "A1", scale_factor=0.6)
        
        # Alice's grid - Fix for Issue 35: A2 to F6
        alice_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": WHITE, "stroke_width": 1, "stroke_opacity": 0.3},
            axis_config={"stroke_color": WHITE, "stroke_width": 2}
        ).add_coordinates()
        self.place_in_area(alice_grid, "A2", "F6", scale_factor=0.8)
        
        # Bob's basis vectors b1=(2,1) and b2=(-1,1)
        grid_origin = alice_grid.c2p(0, 0)
        b1_end = alice_grid.c2p(2, 1)
        b2_end = alice_grid.c2p(-1, 1)
        
        b1 = Arrow(grid_origin, b1_end, buff=0, color=COLOR_B1, stroke_width=4)
        b2 = Arrow(grid_origin, b2_end, buff=0, color=COLOR_B2, stroke_width=4)
        
        b1_label = MathTex("\\vec{b}_1", color=COLOR_B1, font_size=24).next_to(b1_end, UR, buff=0.1)
        b2_label = MathTex("\\vec{b}_2", color=COLOR_B2, font_size=24).next_to(b2_end, UL, buff=0.1)

        self.play(Create(alice_grid), FadeIn(alice_icon), run_time=1)
        self.play(GrowArrow(b1), Write(b1_label), run_time=1)
        self.play(GrowArrow(b2), Write(b2_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_B2)
        )
        
        # Matrix P = [[2, -1], [1, 1]]
        matrix_p = Matrix([[2, -1], [1, 1]], 
                         left_bracket="[", 
                         right_bracket="]",
                         element_alignment_corner=ORIGIN).scale(0.8)
        # Highlight columns with matching vector colors
        matrix_p.get_columns()[0].set_color(COLOR_B1)
        matrix_p.get_columns()[1].set_color(COLOR_B2)
        
        p_label = MathTex("P = ", font_size=36)
        matrix_group = VGroup(p_label, matrix_p).arrange(RIGHT, buff=0.2)
        
        # Fix for Issue 33: B3 to D5
        self.place_in_area(matrix_group, "B3", "D5", scale_factor=1.1)
        
        # Values for b1 and b2 to morph into matrix
        b1_coords = MathTex("\\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix}", color=COLOR_B1, font_size=24).next_to(b1_label, DOWN, buff=0.1)
        b2_coords = MathTex("\\begin{bmatrix} -1 \\\\ 1 \\end{bmatrix}", color=COLOR_B2, font_size=24).next_to(b2_label, DOWN, buff=0.1)

        self.play(Write(b1_coords), Write(b2_coords))
        
        # Morphing: Fade grid/vectors while transforming coord labels into matrix elements
        self.play(
            FadeOut(alice_grid), FadeOut(b1), FadeOut(b2), FadeOut(b1_label), FadeOut(b2_label), FadeOut(alice_icon),
            ReplacementTransform(b1_coords, matrix_p.get_columns()[0]),
            ReplacementTransform(b2_coords, matrix_p.get_columns()[1]),
            Write(p_label),
            Write(matrix_p.get_brackets()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_P)
        )
        
        # Flash matrix P in yellow and label it "Dictionary"
        dictionary_label = Text("Dictionary", color=COLOR_P, font_size=24)
        
        # Fix for Issue 34: E4
        self.place_at_grid(dictionary_label, "E4", scale_factor=1.0)
        
        self.play(
            matrix_group.animate.set_color(COLOR_P),
            Flash(matrix_group, color=COLOR_P, flash_radius=1.5, num_lines=12),
            Write(dictionary_label),
            run_time=1.5
        )
        self.wait(2)
