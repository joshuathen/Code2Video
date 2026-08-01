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
        # Fetch data from storyboard
        title_text = "The Building Blocks: Basis Vectors i and j"
        lecture_lines = [
            "Basis vectors i and j are unit-length building blocks.",
            "Every vector is a combination of these fundamental steps.",
            "Vector three-two is simply three i plus two j."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_I = "#FFFF00" # Yellow
        COLOR_J = "#800080" # Purple
        COLOR_V = "#00FF00" # Green for the final vector
        HIGHLIGHT = YELLOW
        
        # Define grid points relative to origin at F2 (aligned with critic feedback in Issue 33 & 34)
        p_origin = self.grid['F2']
        p_i = self.grid['F3']
        p_j = self.grid['E2']
        p_3i = self.grid['F5']
        p_3i_plus_j = self.grid['E5']
        p_3i_plus_2j = self.grid['D5']
        
        # === Animation for Lecture Line 1 ===
        # "Basis vectors i and j are unit-length building blocks."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        # Show a background grid to provide context
        # NumberPlane aligned to the 6x6 grid points
        back_grid = NumberPlane(
            x_range=[0, 4, 1], y_range=[0, 5, 1],
            x_length=4, y_length=5,
            background_line_style={"stroke_opacity": 0.2},
            axis_config={"stroke_opacity": 0.4}
        )
        self.place_in_area(back_grid, 'A2', 'F6')
        
        i_vec = Arrow(p_origin, p_i, buff=0, color=COLOR_I)
        j_vec = Arrow(p_origin, p_j, buff=0, color=COLOR_J)
        
        # Labels for basis vectors
        i_label = MathTex(r"\mathbf{i}", color=COLOR_I)
        # Positioned above the unit vector i's tip
        self.place_at_grid(i_label, 'E3', scale_factor=0.8)
        
        j_label = MathTex(r"\mathbf{j}", color=COLOR_J)
        # Fix for Issue 33: Label j at E1 (proximity to tip E2)
        self.place_at_grid(j_label, 'E1', scale_factor=0.8)
        
        self.play(Create(back_grid))
        self.play(GrowArrow(i_vec), Write(i_label))
        self.play(GrowArrow(j_vec), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Every vector is a combination of these fundamental steps."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )
        
        # Show linear combination 3*i and 2*j components
        # Constructing 3i
        i1 = Arrow(p_origin, self.grid['F3'], buff=0, color=COLOR_I, stroke_width=2)
        i2 = Arrow(self.grid['F3'], self.grid['F4'], buff=0, color=COLOR_I, stroke_width=2)
        i3 = Arrow(self.grid['F4'], self.grid['F5'], buff=0, color=COLOR_I, stroke_width=2)
        
        # Constructing 2j from the tip of 3i
        j1 = Arrow(p_3i, self.grid['E5'], buff=0, color=COLOR_J, stroke_width=2)
        j2 = Arrow(self.grid['E5'], self.grid['D5'], buff=0, color=COLOR_J, stroke_width=2)
        
        self.play(Create(i1), Create(i2), Create(i3))
        self.play(Create(j1), Create(j2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Vector three-two is simply three i plus two j."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )
        
        # The resulting vector [3, 2]
        res_vec = Arrow(p_origin, p_3i_plus_2j, buff=0, color=COLOR_V, stroke_width=6)
        res_label = MathTex(r"3\mathbf{i} + 2\mathbf{j}", color=COLOR_V)
        # Fix for Issue 34: res_label at C5 (proximity to tip D5)
        self.place_at_grid(res_label, 'C5', scale_factor=0.9)
        
        self.play(GrowArrow(res_vec))
        self.play(Write(res_label))
        self.wait(3)
        
        # Final cleanup: reset lecture line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
