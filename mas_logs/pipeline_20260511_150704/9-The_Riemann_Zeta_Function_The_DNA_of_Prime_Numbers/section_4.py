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
        # Define content
        title_str = "Analytic Continuation: Mapping the Unknown"
        lines = [
            "The original sum only works when s is large.",
            "Analytic continuation extends the function across the entire plane.",
            "This reveals hidden points where the function equals zero."
        ]
        
        # Setup the stage
        self.setup_layout(title_str, lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw a vertical line at x=1 (#FFFFFF) with the right side shaded green (#00FF00).
        self.lecture[0].set_color("#00FF00")
        
        # Boundary line representing Re(s) = 1
        boundary_line = Line(
            self.grid["A3"] + UP*0.3,
            self.grid["F3"] + DOWN*0.3,
            color=WHITE,
            stroke_width=4
        )
        
        # Right shading (Initially valid region)
        # Use a Rectangle centered in the right half of the grid
        shading_right = Rectangle(
            width=2.8,
            height=4.5,
            fill_color="#00FF00",
            fill_opacity=0.3,
            stroke_width=0
        )
        # Position shading to the right of the line (Cols 4-6)
        self.place_in_area(shading_right, "A4", "F6")
        
        # Formula/Label for the valid region (Addressing Issue 43: move to B4)
        zeta_label = Text("ζ(s) = Σ 1/n^s", color="#00FF00", font_size=20)
        self.place_at_grid(zeta_label, "B4")
        
        self.play(Create(boundary_line))
        self.play(FadeIn(shading_right))
        self.play(Write(zeta_label))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Expand the green shading to cover the left side of the line in a smooth wave [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/wave.svg].
        self.lecture[1].set_color("#00FF00")
        
        # Loading wave asset (Addressing Issue 32)
        wave_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wave.svg")
        # Addressing Issue 44: Positioning at D4
        self.place_at_grid(wave_asset, "D4", scale_factor=0.6)
        wave_asset.set_color("#00FF00")
        
        # Expansion area (Cols 1-3)
        shading_left = Rectangle(
            width=2.8,
            height=4.5,
            fill_color="#00FF00",
            fill_opacity=0.3,
            stroke_width=0
        )
        self.place_in_area(shading_left, "A1", "F3")
        
        self.play(FadeIn(wave_asset))
        self.play(
            FadeIn(shading_left, shift=RIGHT),
            wave_asset.animate.scale(1.2).set_opacity(0.6),
            run_time=2
        )
        self.play(FadeOut(wave_asset))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Display the continuation formula in light blue (#ADD8E6) at the bottom.
        self.lecture[2].set_color("#ADD8E6")
        
        # Continuation formula (Addressing Issue 45: Positioning at E4)
        # Using a text-based representation to avoid complex LaTeX issues
        continuation_formula = Text(
            "ζ(s) = η(s) / (1 - 2^(1-s))",
            color="#ADD8E6",
            font_size=24
        )
        self.place_at_grid(continuation_formula, "E4", scale_factor=1.1)
        
        # Visualizing "hidden points" (zeros)
        zero_pts = VGroup(
            Dot(self.grid["C2"], color=RED),
            Dot(self.grid["D1"], color=RED),
            Dot(self.grid["B2"], color=RED)
        )
        zero_label = Text("Non-trivial Zeros", font_size=18, color=RED)
        self.place_at_grid(zero_label, "F4")
        
        self.play(Write(continuation_formula))
        self.play(LaggedStart(*[Flash(dot) for dot in zero_pts], lag_ratio=0.3))
        self.play(FadeIn(zero_pts), Write(zero_label))
        
        self.wait(3)
