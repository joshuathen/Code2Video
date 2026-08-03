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
        title = "The Core Formula: Mapping Coordinates"
        lines = [
            "The formula maps Bob's coordinates to Alice's.",
            "Multiplying by P translates Bob's view to Alice's.",
            "Bob's grid morphing to align with Alice's system.",
            "Calculation confirms the point's position in Alice's grid.",
            "The transformation links their two different perspectives."
        ]
        self.setup_layout(title, lines)

        # Colors
        formula_color = "#FFFFFF"
        matrix_color = "#00FFFF"
        grid_color = "#FFFF00"
        calc_color = "#00FF00"
        final_color = "#FFA500"

        # === Animation for Lecture Line 1 ===
        # Formula [v]_Alice = P * [v]_Bob
        formula = MathTex(
            r"[\vec{v}]_{\text{Alice}} = P [\vec{v}]_{\text{Bob}}", 
            font_size=36, 
            color=formula_color
        )
        # Fix for Issue 38: Adjusted area and scale to avoid top-heavy look.
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(formula_color),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show matrix P multiplying a generic vector [v]_Bob
        matrix_p = MathTex(
            r"P = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}",
            font_size=32,
            color=matrix_color
        )
        # Fix for Issue 36: Adjusted position to prevent overlap with calculation.
        self.place_in_area(matrix_p, 'B1', 'B2', scale_factor=0.7)

        self.play(
            self.lecture[1].animate.set_color(matrix_color),
            FadeIn(matrix_p)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate Bob's grid morphing and stretching into Alice's square grid
        grid_center = self.grid["D4"]
        
        # Standard Grid (Alice)
        alice_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE, "stroke_opacity": 0.3},
            axis_config={"include_tip": False, "stroke_opacity": 0.5}
        ).scale(0.5).move_to(grid_center)

        # Skewed Grid (Bob) - Columns of P are the basis vectors
        bob_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": grid_color, "stroke_opacity": 0.6},
            axis_config={"include_tip": False, "stroke_opacity": 0.5}
        ).scale(0.5).apply_matrix([[2, -1], [1, 1]]).move_to(grid_center)

        self.play(
            self.lecture[2].animate.set_color(grid_color),
            Create(bob_grid),
            FadeOut(matrix_p)
        )
        self.wait(1)
        
        self.play(
            bob_grid.animate.become(alice_grid),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Plug in Bob's coordinates (1, 1) into the formula on screen
        calc = MathTex(
            r"\begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}",
            font_size=32,
            color=calc_color
        )
        # Fix for Issue 37: Adjusted position to prevent overlap with previous matrix and crowding.
        self.place_in_area(calc, 'B3', 'B6', scale_factor=0.7)

        self.play(
            self.lecture[3].animate.set_color(calc_color),
            Write(calc)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show result (1, 2) and highlight point on the Alice grid
        dot = Dot(point=alice_grid.coords_to_point(1, 2), color=final_color)
        dot_label = MathTex(r"(1, 2)_{\text{Alice}}", font_size=24, color=final_color)
        dot_label.next_to(dot, UR, buff=0.1)

        self.play(
            self.lecture[4].animate.set_color(final_color),
            FadeIn(dot),
            Write(dot_label)
        )
        self.wait(2)
