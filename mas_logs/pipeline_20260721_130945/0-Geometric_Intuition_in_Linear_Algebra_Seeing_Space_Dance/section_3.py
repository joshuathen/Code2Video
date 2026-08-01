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

class Section3Scene(TeachingScene):
    def construct(self):
        # Section Title and Lecture Lines
        title_text = "Matrices as Space Transformers"
        lecture_lines = [
            "- Matrices don't just multiply; they transform the entire space.",
            "- The origin stays fixed while the world reshapes around.",
            "- Watch the grid tilt during this \"shear\" transformation.",
            "- Grid lines remain parallel and evenly spaced throughout.",
            "- Linear transformations keep the grid's underlying structure consistent."
        ]
        
        # Initialize Layout
        self.setup_layout(title_text, lecture_lines)
        
        # Asset path
        GRID_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"

        # Colors
        COLOR_YELLOW = "#FFFF00"
        COLOR_WHITE = "#FFFFFF"
        COLOR_CYAN = "#00FFFF"
        COLOR_MAGENTA = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # [Animation 1] Show standard grid [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]. 
        # Display matrix [[1, 1], [0, 1]] in #FFFFFF. Change color of line 1 to #FFFF00.
        self.lecture[0].set_color(COLOR_YELLOW)
        
        grid_svg = SVGMobject(GRID_ASSET)
        # Issue 31: Fix overlap by shifting grid area from B2:F5 to B3:F6 as requested.
        self.place_in_area(grid_svg, 'B3', 'F6', scale_factor=0.5)
        
        matrix_mobject = MathTex(r"\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}", color=COLOR_WHITE)
        # Issue 32: Better matrix alignment at A5.
        self.place_at_grid(matrix_mobject, 'A5', scale_factor=0.8)
        
        self.play(
            DrawBorderThenFill(grid_svg),
            FadeIn(matrix_mobject),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # [Animation 2] Highlight matrix in #FFFF00. Flash origin (0,0) in #FFFFFF. Change color of line 2 to #FFFF00.
        self.lecture[1].set_color(COLOR_YELLOW)
        
        origin_dot = Dot(grid_svg.get_center(), color=COLOR_WHITE, radius=0.08)
        
        self.play(
            matrix_mobject.animate.set_color(COLOR_YELLOW),
            Flash(origin_dot, color=COLOR_WHITE, flash_radius=0.3, line_stroke_width=3),
            FadeIn(origin_dot),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # [Animation 3] Transform grid: tilt vertical lines 45 degrees. Change color of line 3 to #FFFF00.
        self.lecture[2].set_color(COLOR_YELLOW)
        
        shear_matrix = [[1, 1], [0, 1]]
        
        self.play(
            grid_svg.animate.apply_matrix(shear_matrix),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # [Animation 4] Highlight two parallel grid lines in #00FFFF. Change color of line 4 to #FFFF00.
        self.lecture[3].set_color(COLOR_YELLOW)
        
        center = grid_svg.get_center()
        # Relative coordinates based on typical SVG scale.
        # Shear transformation (x, y) -> (x+y, y)
        # Vertical Line at x=-0.5 (y range -1 to 1): (-1.5, -1) to (0.5, 1)
        # Vertical Line at x=0.5 (y range -1 to 1): (-0.5, -1) to (1.5, 1)
        line1 = Line(
            center + np.array([-1.5, -1.0, 0]), 
            center + np.array([0.5, 1.0, 0]), 
            color=COLOR_CYAN, stroke_width=6
        )
        line2 = Line(
            center + np.array([-0.5, -1.0, 0]), 
            center + np.array([1.5, 1.0, 0]), 
            color=COLOR_CYAN, stroke_width=6
        )
        
        self.play(
            Create(line1),
            Create(line2)
        )
        self.play(Indicate(line1, color=COLOR_CYAN), Indicate(line2, color=COLOR_CYAN))
        self.play(FadeOut(line1), FadeOut(line2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # [Animation 5] Briefly flash several grid squares in #FF00FF [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        self.lecture[4].set_color(COLOR_YELLOW)
        
        # Helper to create small sheared grid markers from the asset
        def get_sheared_marker(pos):
            m = SVGMobject(GRID_ASSET).scale(0.15).set_color(COLOR_MAGENTA)
            m.apply_matrix(shear_matrix)
            m.move_to(center + pos)
            return m

        square1 = get_sheared_marker(np.array([0.5, 0.5, 0]))
        square2 = get_sheared_marker(np.array([-0.5, -0.5, 0]))
        square3 = get_sheared_marker(np.array([1.5, -0.5, 0]))
        
        self.play(
            Flash(square1, color=COLOR_MAGENTA),
            Flash(square2, color=COLOR_MAGENTA),
            Flash(square3, color=COLOR_MAGENTA),
            FadeIn(square1), FadeIn(square2), FadeIn(square3),
            run_time=2
        )
        self.play(FadeOut(square1), FadeOut(square2), FadeOut(square3))
        self.wait(2)
