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
        # Setup basic layout using storyboard lecture lines
        self.setup_layout(
            "The Inverse Operation: Going Backwards", 
            [
                "The inverse matrix P inverse performs the reverse translation.",
                "It converts standard coordinates back into Basis B.",
                "This is vital for local navigation and sensor data.",
                "Onboard computers use inverses to understand the world.",
                "We move between perspectives by applying matrix transformations."
            ]
        )

        # Color palette
        COLOR_P = WHITE
        COLOR_P_INV = "#FFD700"  # Gold
        COLOR_BASIS = "#FFA500"  # Orange
        COLOR_STD = "#ADD8E6"    # Light Blue

        # === Animation for Lecture Line 1 ===
        # Display the Inverse Matrix P⁻¹ in gold next to Matrix P.
        self.lecture[0].set_color(COLOR_P_INV)
        
        p_mat_val = [[1, -1], [1, 1]]
        p_inv_val = [[0.5, 0.5], [-0.5, 0.5]]
        
        p_matrix = Matrix(p_mat_val).scale(0.5)
        p_label = MathTex("P =", color=COLOR_P).scale(0.6).next_to(p_matrix, LEFT)
        p_group = VGroup(p_label, p_matrix)
        # Fix from Issue 36: Place at B4, scale 0.7
        self.place_at_grid(p_group, "B4", scale_factor=0.7)
        
        p_inv_matrix = Matrix(p_inv_val).scale(0.5).set_color(COLOR_P_INV)
        p_inv_label = MathTex("P^{-1} =", color=COLOR_P_INV).scale(0.6).next_to(p_inv_matrix, LEFT)
        p_inv_group = VGroup(p_inv_label, p_inv_matrix)
        # Fix from Issue 37: Place at B6, scale 0.7
        self.place_at_grid(p_inv_group, "B6", scale_factor=0.7)
        
        self.play(FadeIn(p_group), FadeIn(p_inv_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the symbolic equation: P⁻¹ * [v]_Standard = [v]_B.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_P_INV)
        
        eqn = MathTex(
            "P^{-1}", 
            "\\vec{v}_{std}", 
            "=", 
            "\\vec{v}_B",
            font_size=36
        )
        eqn[0].set_color(COLOR_P_INV)
        eqn[1].set_color(COLOR_STD)
        eqn[3].set_color(COLOR_BASIS)
        self.place_at_grid(eqn, "C5", scale_factor=1.1)
        
        self.play(Write(eqn))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display standard grid and drone icon
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_P_INV)
        
        std_grid = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_color": COLOR_STD, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_STD, "stroke_width": 1}
        ).scale(0.6)
        # Fix from Issue 38: Place in D4-F6 area, scale 0.8
        self.place_in_area(std_grid, "D4", "F6", scale_factor=0.8)
        
        basis_grid = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_color": COLOR_BASIS, "stroke_opacity": 0.5},
            axis_config={"stroke_color": COLOR_BASIS, "stroke_width": 2}
        ).scale(0.6)
        basis_grid.apply_matrix(p_mat_val) 
        # Fix from Issue 38: Place in D4-F6 area, scale 0.8
        self.place_in_area(basis_grid, "D4", "F6", scale_factor=0.8)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg]
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        drone.set_color(WHITE)
        # Fix from Issue 38: Place in D4-F6 area, scale 0.8
        self.place_in_area(drone, "D4", "F6", scale_factor=0.5) # Drone needs to be a bit smaller relative to the grid
        
        self.play(FadeIn(std_grid), Create(drone))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate drone shifting from standard grid alignment to Basis B alignment.
        # This is interpreted as showing the Basis B grid (the "backwards" perspective).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_P_INV)
        
        self.play(
            FadeOut(std_grid),
            FadeIn(basis_grid),
            drone.animate.set_color(COLOR_BASIS)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_P_INV)
        
        # Visual rotation to signify change of perspective/navigation
        self.play(
            drone.animate.rotate(PI/4),
            Indicate(eqn),
            run_time=2
        )
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
