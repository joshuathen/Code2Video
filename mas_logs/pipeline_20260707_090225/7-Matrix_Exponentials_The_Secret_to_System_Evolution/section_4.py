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
        # Initialization
        lecture_lines = [
            "Computing infinite series directly is often difficult.",
            "Diagonalization simplifies the matrix A into P D P-inverse.",
            "The exponential of a diagonal matrix is easy to compute.",
            "Change basis, scale the components, then change back.",
            "This shortcut makes matrix exponentials computationally practical."
        ]
        self.setup_layout("The Computational Shortcut: Diagonalization", lecture_lines)

        # Colors
        P_COLOR = "#FF00FF"
        D_COLOR = "#FFFF00"
        V_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        # Using MarkupText instead of MathTex to avoid Dependency on LaTeX binary
        messy_series = MarkupText(
            'e<sup>At</sup> = I + At + (At)<sup>2</sup>/2! + ...', 
            font_size=32, color=BLUE_B
        )
        self.place_in_area(messy_series, "A1", "B6")
        self.play(Write(messy_series))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        
        # Using MarkupText for matrices and formulas
        p_mat = MarkupText(f'<span color="{P_COLOR}">P</span>', font_size=48)
        d_mat = MarkupText(f'<span color="{D_COLOR}">D</span>', font_size=48)
        p_inv_mat = MarkupText(f'<span color="{P_COLOR}">P<sup>-1</sup></span>', font_size=48)
        
        self.place_at_grid(p_mat, "C2", scale_factor=1.1)
        self.place_at_grid(d_mat, "C4", scale_factor=1.1)
        self.place_at_grid(p_inv_mat, "C6", scale_factor=1.1)

        diag_formula = MarkupText(
            f'A = <span color="{P_COLOR}">P</span> <span color="{D_COLOR}">D</span> <span color="{P_COLOR}">P<sup>-1</sup></span>', 
            font_size=36
        )
        self.place_in_area(diag_formula, "B1", "B6")

        self.play(
            messy_series.animate.set_opacity(0.3),
            FadeIn(diag_formula),
            FadeIn(p_mat),
            FadeIn(d_mat),
            FadeIn(p_inv_mat)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        final_formula = MarkupText(
            f'e<sup>At</sup> = <span color="{P_COLOR}">P</span> e<sup><span color="{D_COLOR}">Dt</span></sup> <span color="{P_COLOR}">P<sup>-1</sup></span>', 
            font_size=36
        )
        self.place_in_area(final_formula, "B1", "B6")
        
        e_dt = MarkupText(f'<span color="{D_COLOR}">e<sup>Dt</sup></span>', font_size=44)
        self.place_at_grid(e_dt, "C4", scale_factor=1.0) 

        self.play(
            Transform(diag_formula, final_formula),
            Transform(d_mat, e_dt)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        
        # Coordinate System for visualization
        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.3}
        )
        vec = Vector([1.0, 0.5], color=V_COLOR)
        plane_group = VGroup(plane, vec)
        self.place_in_area(plane_group, "D1", "F6", scale_factor=0.6)
        
        self.play(Create(plane), GrowArrow(vec))
        self.wait(0.5)

        # Process: P^-1 (basis change) -> Scale -> P (basis back)
        p_inv_matrix = [[0.5, 0.5], [-0.5, 0.5]]
        self.play(
            p_inv_mat.animate.scale(1.2).set_color(WHITE),
            vec.animate.apply_matrix(p_inv_matrix),
            run_time=1.5
        )
        self.play(p_inv_mat.animate.scale(1/1.2).set_color(P_COLOR))

        # Scaling
        self.play(
            d_mat.animate.scale(1.2).set_color(WHITE),
            vec.animate.scale(2.2),
            run_time=1.5
        )
        self.play(d_mat.animate.scale(1/1.2).set_color(D_COLOR))

        # Applying P
        p_matrix = [[1, -1], [1, 1]]
        self.play(
            p_mat.animate.scale(1.2).set_color(WHITE),
            vec.animate.apply_matrix(p_matrix),
            run_time=1.5
        )
        self.play(p_mat.animate.scale(1/1.2).set_color(P_COLOR))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        
        final_box = SurroundingRectangle(diag_formula, color=PURPLE)
        self.play(Create(final_box))
        self.wait(2)
