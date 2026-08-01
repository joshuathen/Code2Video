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
        # 1. Setup Layout
        self.setup_layout("The Infinite Limit: When Finite Becomes Infinite", [
            "What happens as we repeat this process infinitely?",
            "The total length of the line grows toward infinity.",
            "The gaps between the lines eventually shrink to zero.",
            "Mathematically, this limit curve passes through every point.",
            "A one-dimensional line has finally filled the two-dimensional space."
        ])

        # Recursive helper to get Hilbert points for curve generation
        def get_hilbert_points(order):
            if order == 0:
                return [np.array([0, 0, 0])]
            points = get_hilbert_points(order - 1)
            size = 2**(order - 1)
            # Quadrant 1: swap x and y
            p1 = [np.array([p[1], p[0], 0]) for p in points]
            # Quadrant 2: shift y
            p2 = [np.array([p[0], p[1] + size, 0]) for p in points]
            # Quadrant 3: shift x and y
            p3 = [np.array([p[0] + size, p[1] + size, 0]) for p in points]
            # Quadrant 4: shift x, flip and swap
            p4 = [np.array([2*size - 1 - p[1], size - 1 - p[0], 0]) for p in points]
            return p1 + p2 + p3 + p4

        def get_hilbert_curve(order, color="#00FFFF"):
            points = get_hilbert_points(order)
            size = 2**order - 1
            if size <= 0: size = 1
            # Scale to fit roughly a 3.5x3.5 area in the grid
            scale = 3.5 / size
            shifted_points = [(p - np.array([size/2, size/2, 0])) * scale for p in points]
            curve = VMobject()
            curve.set_points_as_corners(shifted_points)
            curve.set_color(color)
            # Adjust stroke width for density as order increases
            curve.set_stroke(width=max(0.5, 4.0 - order * 0.4))
            return curve

        # === Animation for Lecture Line 1 ===
        # Display the Level 4 Hilbert curve (#00FFFF) with label 'n = 4' (#FFFFFF)
        self.lecture[0].set_color("#00FFFF")
        current_n = 4
        curve = get_hilbert_curve(current_n)
        self.place_in_area(curve, "B2", "E5")
        
        n_label = Text(f"n = {current_n}", font_size=24, color=WHITE)
        self.place_in_area(n_label, "F3", "F4")
        
        self.play(Create(curve), Write(n_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rapidly cycle through Level 5, 6, and 7 curves (#00FFFF)
        self.lecture[1].set_color("#00FFFF")
        for n in [5, 6, 7]:
            new_curve = get_hilbert_curve(n)
            self.place_in_area(new_curve, "B2", "E5")
            new_label = Text(f"n = {n}", font_size=24, color=WHITE)
            self.place_in_area(new_label, "F3", "F4")
            
            self.play(
                Transform(curve, new_curve),
                Transform(n_label, new_label),
                run_time=0.7
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition the dense curve into a solid white square (#FFFFFF)
        self.lecture[2].set_color("#FFFFFF")
        square = Square(side_length=3.5, color=WHITE, fill_opacity=1.0, stroke_width=0)
        self.place_in_area(square, "B2", "E5")
        
        inf_label = Text("n → ∞", font_size=32, color=WHITE)
        self.place_in_area(inf_label, "F3", "F4")
        
        self.play(
            ReplacementTransform(curve, square),
            ReplacementTransform(n_label, inf_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display formulas 'L -> infinity' and 'Area -> 1' in yellow (#FFFF00)
        self.lecture[3].set_color("#FFFF00")
        len_formula = Text("L → ∞", font_size=32, color="#FFFF00")
        area_formula = Text("Area → 1", font_size=32, color="#FFFF00")
        
        # Improved positioning to ensure consistent centering and grouping (Issues 33, 34, 35)
        self.place_in_area(len_formula, 'A2', 'A3', scale_factor=0.8)
        self.place_in_area(area_formula, 'A4', 'A5', scale_factor=0.8)
        
        self.play(Write(len_formula), Write(area_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash the solid white square (#FFFFFF)
        self.lecture[4].set_color("#FFFFFF")
        self.play(Flash(square, color=WHITE, line_length=0.4, num_lines=12, flash_radius=1.8))
        self.wait(2)
