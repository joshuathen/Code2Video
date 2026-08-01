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
        # Setup Data
        title_text = "Into the Complex Plane: The Rubber Sheet"
        lecture_lines = [
            "Now, we expand our view into the complex plane.",
            "The input s becomes a coordinate with real and imaginary parts.",
            "Pete’s grid warps like a colorful, stretching rubber sheet.",
            "Analytic continuation extends the function beyond its original limits.",
            "This reveals the function's behavior across the entire plane."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a 2D complex plane grid using #696969 lines.
        # We use a NumberPlane for the complex plane visual.
        complex_plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#696969", "stroke_width": 1, "stroke_opacity": 0.6},
            axis_config={"stroke_color": "#696969", "stroke_width": 2}
        )
        self.place_in_area(complex_plane, "B2", "F6", scale_factor=0.6)
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(complex_plane), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Plot a point representing 's = a + bi' in #ADFF2F.
        point_color = "#ADFF2F"
        s_point = Dot(color=point_color)
        self.place_at_grid(s_point, "C4", scale_factor=1.2) # Centered relative to plane area
        s_label = Text("s = a + bi", color=point_color, font_size=24)
        self.place_at_grid(s_label, "C5", scale_factor=0.9) # Near the point
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(FadeIn(s_point), Write(s_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Apply a warping transformation to the grid, stretching it like rubber.
        # Refinement: Move square to 'E2' (scale 0.7).
        square = Square(color=BLUE, stroke_width=2)
        self.place_at_grid(square, "E2", scale_factor=0.7)
        
        # Prepare for warping - subdivision helps smooth deformation
        complex_plane.prepare_for_nonlinear_transform()
        
        plane_center = complex_plane.get_center()
        def warp_func(p):
            rel_p = p - plane_center
            dist = np.linalg.norm(rel_p)
            if dist == 0: return p
            # Stretching effect resembling a rubber sheet
            return plane_center + rel_p * (1 + 0.3 * np.sin(dist))

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(FadeIn(square))
        self.play(
            complex_plane.animate.apply_function(warp_func),
            square.animate.apply_function(warp_func),
            s_point.animate.apply_function(warp_func),
            s_label.animate.shift(RIGHT*0.2), # Manual adjustment to stay near point
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Colors on the grid shift to #4B0082 as it expands.
        expansion_color = "#4B0082"
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        self.play(
            complex_plane.animate.set_color(expansion_color),
            square.animate.set_color(expansion_color),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The 'Zeta-Tron' machine label updates to show complex inputs.
        # Refinement: Add Triangle at 'D5' (scale 0.8).
        triangle = Triangle(color=GOLD, fill_opacity=0.3)
        self.place_at_grid(triangle, "D5", scale_factor=0.8)
        
        zeta_tron_label = Text("Zeta-Tron: s ∈ ℂ", font_size=24, color=WHITE)
        # Position label in Row B area to avoid overlap with grid visuals below
        self.place_in_area(zeta_tron_label, "B2", "B4", scale_factor=0.8)
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        self.play(
            FadeIn(triangle),
            Write(zeta_tron_label)
        )
        self.wait(2)
