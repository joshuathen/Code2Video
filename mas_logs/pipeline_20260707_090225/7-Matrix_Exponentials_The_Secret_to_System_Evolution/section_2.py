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
        # Setup the scene with specific title and lecture lines
        self.setup_layout(
            "Prerequisite: The Taylor Series Bridge", 
            [
                "Recall the Taylor Series for e to the x.", 
                "We can sum powers of matrices just like numbers.", 
                "Replace the scalar x with the matrix product At.", 
                "Each term involves powers of the matrix A.", 
                "This series defines the exponential of a matrix."
            ]
        )

        def create_matrix_mob(elements, color=WHITE):
            # Helper to create a 2x2 matrix without LaTeX dependency
            mob_elements = VGroup(*[
                VGroup(*[Text(str(e), font_size=24, color=color) for e in row]).arrange(RIGHT, buff=0.5)
                for row in elements
            ]).arrange(DOWN, buff=0.4)
            lb = Text("[", font_size=48, color=color)
            rb = Text("]", font_size=48, color=color)
            lb.next_to(mob_elements, LEFT, buff=0.1)
            rb.next_to(mob_elements, RIGHT, buff=0.1)
            return VGroup(mob_elements, lb, rb)

        # === Animation for Lecture Line 1 ===
        # Highlight x in #FFD700
        self.lecture[0].set_color(YELLOW)
        scalar_series = MarkupText(
            'e<sup><span color="#FFD700">x</span></sup> = 1 + <span color="#FFD700">x</span> + <span color="#FFD700">x</span><sup>2</sup>/2! + ...',
            color=WHITE,
            font_size=28
        )
        self.place_at_grid(scalar_series, "B3", scale_factor=1.0)
        self.play(Write(scalar_series))
        self.wait(1)

        # === Animation for Lecture Line 2 & 3 ===
        # Transitioning to matrix series using bridge asset
        self.lecture[1].set_color(YELLOW)
        self.lecture[2].set_color(YELLOW)
        
        bridge_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge_asset, "B5", scale_factor=0.6)
        
        # Matrix expansion formula at C3 (Issue 35 Fix)
        # Highlight I in #00FFFF (Issue 31 Integration)
        matrix_expansion = MarkupText(
            'e<sup>At</sup> = <span color="#00FFFF">I</span> + At + (At)<sup>2</sup>/2! + ...',
            color=WHITE,
            font_size=28
        )
        self.place_at_grid(matrix_expansion, "C3", scale_factor=1.0)
        
        self.play(FadeIn(bridge_asset, shift=RIGHT))
        self.play(Write(matrix_expansion))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show 3 distinct 2x2 Matrix objects vertically stacked
        self.lecture[3].set_color(YELLOW)
        
        m1 = create_matrix_mob([["1", "0"], ["0", "1"]])
        plus1 = Text("+", font_size=32)
        m2 = create_matrix_mob([["a", "b"], ["c", "d"]])
        plus2 = Text("+", font_size=32)
        m3 = create_matrix_mob([["...", "..."], ["...", "..."]])
        
        matrix_stack = VGroup(m1, plus1, m2, plus2, m3).arrange(DOWN, buff=0.2)
        self.place_at_grid(matrix_stack, "E2", scale_factor=0.5) # Issue 36 Fix
        
        self.play(Create(matrix_stack))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Merge them into a single matrix labeled e^{At} and show solution vector
        self.lecture[4].set_color(YELLOW)
        
        final_matrix = create_matrix_mob([["e^At_11", "e^At_12"], ["e^At_21", "e^At_22"]], color=YELLOW)
        self.place_at_grid(final_matrix, "E2", scale_factor=0.6)
        
        vector_sol = MarkupText("x(t) = e<sup>At</sup> x(0)", color=YELLOW, font_size=28)
        self.place_at_grid(vector_sol, "E5", scale_factor=1.0) # Issue 37 Fix

        self.play(
            ReplacementTransform(matrix_stack, final_matrix),
            Write(vector_sol)
        )
        self.wait(3)
