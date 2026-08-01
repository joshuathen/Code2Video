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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout("The Cast of Characters", [
            "Mathematics features five fundamental, seemingly unrelated constants.",
            "Meet zero, one, e, the imaginary i, and pi.",
            "Today, we unite these different worlds in one equation."
        ])

        # Initial focus setup: all dimmed
        self.lecture.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )

        # Constants: {0: #FFD700, 1: #00FF00, e: #00BFFF, i: #FF00FF, pi: #FFA500}
        c_zero = Text("0", color="#FFD700")
        c_one = Text("1", color="#00FF00")
        c_e = Text("e", color="#00BFFF", slant=ITALIC)
        c_i = Text("i", color="#FF00FF", slant=ITALIC)
        c_pi = Text("π", color="#FFA500")

        # Labels for each constant with matching colors
        l_arithmetic = Text("Arithmetic", font_size=24, color="#FFD700")
        l_identity = Text("Identity", font_size=24, color="#00FF00")
        l_calculus = Text("Calculus", font_size=24, color="#00BFFF")
        l_imaginary = Text("Imaginary", font_size=24, color="#FF00FF")
        l_geometry = Text("Geometry", font_size=24, color="#FFA500")

        # Position characters using the grid system
        self.place_at_grid(c_zero, "B2", scale_factor=1.8)
        self.place_at_grid(l_arithmetic, "C2", scale_factor=0.6)
        
        self.place_at_grid(c_one, "B5", scale_factor=1.8)
        self.place_at_grid(l_identity, "C5", scale_factor=0.6)
        
        self.place_at_grid(c_e, "D2", scale_factor=1.8)
        self.place_at_grid(l_calculus, "E2", scale_factor=0.6)
        
        self.place_at_grid(c_i, "D5", scale_factor=1.8)
        self.place_at_grid(l_imaginary, "E5", scale_factor=0.6)
        
        self.place_at_grid(c_pi, "E4", scale_factor=1.8)
        self.place_at_grid(l_geometry, "F4", scale_factor=0.6)

        # Animation: Constants and their labels appear
        self.play(
            FadeIn(c_zero), FadeIn(l_arithmetic),
            FadeIn(c_one), FadeIn(l_identity),
            FadeIn(c_e), FadeIn(l_calculus),
            FadeIn(c_i), FadeIn(l_imaginary),
            FadeIn(c_pi), FadeIn(l_geometry),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )

        # Unification: Calculate target arrangement (horizontal group)
        # Sequence: e, i, pi, 1, 0 (consistent with final identity form exploration)
        target_group = VGroup(
            c_e.copy(), 
            c_i.copy(), 
            c_pi.copy(), 
            c_one.copy(), 
            c_zero.copy()
        ).arrange(RIGHT, buff=0.5)
        
        # Center the group in the middle area of the right grid
        self.place_in_area(target_group, "C2", "D5", scale_factor=1.2)

        # Animate constants merging toward center as labels fade out
        self.play(
            FadeOut(l_arithmetic),
            FadeOut(l_identity),
            FadeOut(l_calculus),
            FadeOut(l_imaginary),
            FadeOut(l_geometry),
            c_e.animate.move_to(target_group[0]),
            c_i.animate.move_to(target_group[1]),
            c_pi.animate.move_to(target_group[2]),
            c_one.animate.move_to(target_group[3]),
            c_zero.animate.move_to(target_group[4]),
            run_time=2
        )
        self.wait(2)
