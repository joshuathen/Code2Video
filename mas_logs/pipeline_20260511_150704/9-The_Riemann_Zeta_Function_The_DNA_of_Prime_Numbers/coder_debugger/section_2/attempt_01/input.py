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
        # Initialize the teaching environment
        self.setup_layout(
            "Defining the Riemann Zeta Function", 
            [
                "Let's define the Riemann Zeta function for variable s.", 
                "It is the sum of n to the power negative s.", 
                "Initially, we consider s as a simple real number.", 
                "However, s can also be a complex coordinate.", 
                "This transforms the sum into a complex landscape."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Show Zeta function formula in yellow (#FFFF00)
        zeta_formula = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}", color="#FFFF00")
        # Issue 37 Fix: Position using place_in_area to avoid title/lecture overlap
        self.place_in_area(zeta_formula, 'A2', 'B5', scale_factor=1.1)
        
        self.play(
            self.lecture[0].animate.set_color("#FFFF00"),
            Write(zeta_formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight definition of sum
        self.play(
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to real number context
        self.play(
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Define complex coordinate s = sigma + it in blue (#50C8FF)
        complex_s_coord = MathTex(r"s = \sigma + it", color="#50C8FF")
        # Issue 38 Fix: Position using place_at_grid to clear visual clutter
        self.place_at_grid(complex_s_coord, 'C3', scale_factor=0.8)
        
        # Fade in 2D grid in gray (#808080)
        complex_plane_grid = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3, y_length=3,
            background_line_style={"stroke_color": "#808080", "stroke_opacity": 0.5},
            axis_config={"stroke_color": "#808080"}
        )
        self.place_in_area(complex_plane_grid, 'D1', 'F6', scale_factor=0.8)
        
        # Plot pulsing dot at coordinate s in magenta (#FF00FF)
        dot = Dot(complex_plane_grid.c2p(1, 0.5), color="#FF00FF")
        
        self.play(
            self.lecture[3].animate.set_color("#50C8FF"),
            Write(complex_s_coord),
            Create(complex_plane_grid)
        )
        self.play(FadeIn(dot))
        # Simple pulse animation
        self.play(dot.animate.scale(1.5), rate_func=there_and_back, run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Complex landscape visualization
        # Issue 39 Fix: Position landscape_png in 'D1' to 'F6'
        # Representing the "complex landscape" with a gradient rectangle
        landscape_png = Rectangle(width=4, height=3, fill_opacity=0.3).set_stroke(width=0)
        landscape_png.set_fill(color=[BLUE, PURPLE, PINK])
        self.place_in_area(landscape_png, 'D1', 'F6', scale_factor=0.9)
        
        self.play(
            self.lecture[4].animate.set_color("#FF00FF"),
            FadeIn(landscape_png)
        )
        self.wait(2)
