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
        # Section title and lecture lines
        title_text = "The Reverse Journey (P-Inverse)"
        lecture_lines = [
            "• What if we want to translate coordinates back?",
            "• We need the inverse of our transition matrix, P.",
            "• Multiplying by P-inverse converts our view.",
            "• This operation 'undoes' the grid's tilt.",
            "• Now we can communicate back into Pixel's language."
        ]
        
        # Colors for alignment
        c1 = "#88C0D0" # Aqua
        c2 = "#EBCB8B" # Yellow
        c3 = "#A3BE8C" # Green
        c4 = "#B48EAD" # Purple
        c5 = "#D08770" # Orange

        # 1. Initialize layout
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        # Start with the equation x = P * [x]_B
        eq1 = MathTex("x", "=", "P", "[x]_B", color=c1)
        self.place_at_grid(eq1, "A3", scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        # Multiply both sides by P^-1 to get [x]_B = P^-1 * x
        eq2 = MathTex("[x]_B", "=", "P^{-1}", "x", color=c2)
        self.place_at_grid(eq2, "A3", scale_factor=1.0)
        self.play(Transform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        # Show the inverse matrix P^-1 appearing on screen
        p_inv_matrix = Matrix([[1, -1], [0, 1]], color=c3).scale(0.7)
        p_inv_label = MathTex("P^{-1} =", color=c3).scale(0.8)
        p_inv_group = VGroup(p_inv_label, p_inv_matrix).arrange(RIGHT, buff=0.2)
        self.place_at_grid(p_inv_group, "B3")
        self.play(FadeIn(p_inv_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(c4))
        
        # Setup the grid and "un-skew" it
        # [Issue 47 Fix]: Plane positioning in A2-F6
        plane = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'A2', 'F6', scale_factor=0.8)
        
        # Initial tilt state (Basis B seen from standard basis)
        p_matrix_vals = [[1, 1], [0, 1]]
        p_inv_matrix_vals = [[1, -1], [0, 1]]
        plane.apply_matrix(p_matrix_vals)
        
        # [Issue 49 Fix]: Basis vector labels positioning
        b1_arrow = Arrow(plane.get_origin(), plane.c2p(1, 0), buff=0, color=RED)
        b2_arrow = Arrow(plane.get_origin(), plane.c2p(0, 1), buff=0, color=GREEN)
        
        b1_label = MathTex("\\vec{b}_1", color=RED)
        b2_label = MathTex("\\vec{b}_2", color=GREEN)
        self.place_at_grid(b1_label, 'C4', scale_factor=0.6)
        self.place_at_grid(b2_label, 'B3', scale_factor=0.6)
        
        self.play(Create(plane))
        self.play(GrowArrow(b1_arrow), GrowArrow(b2_arrow))
        self.play(Write(b1_label), Write(b2_label))
        self.wait(1)
        
        # Undoing the tilt (Applying P^-1)
        # [Issue 48 Fix]: Reposition v2_label (b2_label here) to avoid title overlap
        self.play(
            plane.animate.apply_matrix(p_inv_matrix_vals),
            b1_arrow.animate.apply_matrix(p_inv_matrix_vals),
            b2_arrow.animate.apply_matrix(p_inv_matrix_vals),
            b1_label.animate.move_to(self.grid['D4']),
            b2_label.animate.move_to(self.grid['A2']).scale(1.16), # Scale up to 0.7 total
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(c5))
        
        # [Issue 31 Fix]: Asset Integration
        pixel_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg")
        pixel_asset.set_height(0.5)
        
        self.play(FadeOut(b1_label), FadeOut(b2_label), FadeOut(b1_arrow), FadeOut(b2_arrow))
        
        # Vector x = [5, 4] in standard view
        vec_x = Arrow(plane.get_origin(), plane.c2p(5, 4), buff=0, color=c5)
        vec_label = MathTex("x = \\begin{bmatrix} 5 \\\\ 4 \\end{bmatrix}", color=c5).scale(0.6)
        self.place_at_grid(vec_label, "E5")
        
        pixel_asset.move_to(plane.c2p(5, 4) + 0.3 * UR)
        
        self.play(GrowArrow(vec_x), FadeIn(pixel_asset), Write(vec_label))
        self.wait(1)
        
        # Show coordinate conversion: [x]_B = P^-1 * x = [1, 4]
        self.play(
            vec_x.animate.apply_matrix(p_inv_matrix_vals),
            pixel_asset.animate.move_to(plane.c2p(1, 4) + 0.3 * UR),
            vec_label.animate.become(
                MathTex("[x]_B = \\begin{bmatrix} 1 \\\\ 4 \\end{bmatrix}", color=c5).scale(0.6).move_to(self.grid["E5"])
            ),
            run_time=2
        )
        
        self.wait(2)
