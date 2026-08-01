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

class Section7Scene(TeachingScene):
    def construct(self):
        # Section data
        title_text = "Summary and Elegance"
        lecture_lines = [
            "The Triangle of Power turns symbols into geometric logic.",
            "Complex identities become simple movements across the shape.",
            "Math is about patterns, not just memorizing symbols."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        GOLDEN = "#FFD700"
        GRAY = "#808080"
        PIXEL_COLOR = "#FFFFFF"
        BEAK_COLOR = "#FFA500"
        BELLY_COLOR = "#E0E0E0"

        # === Animation for Lecture Line 1 ===
        # The cluttered gray machine from Section 1 fades out completely.
        # The golden (#FFD700) Triangle of Power fades in at the center.
        
        self.lecture[0].set_color(GOLDEN)
        
        # Build Gray Machine (representing the fragmented concepts)
        machine_part1 = Rectangle(width=1.5, height=1.0, color=GRAY, fill_opacity=0.5)
        machine_part2 = Circle(radius=0.4, color=GRAY, fill_opacity=0.5).shift(RIGHT*0.6)
        machine_part3 = Rectangle(width=0.8, height=1.2, color=GRAY, fill_opacity=0.5).shift(LEFT*0.5)
        machine_gear1 = Star(n=8, outer_radius=0.3, inner_radius=0.2, color=GRAY, fill_opacity=0.8).shift(UP*0.5 + LEFT*0.3)
        machine_gear2 = Star(n=8, outer_radius=0.2, inner_radius=0.1, color=GRAY, fill_opacity=0.8).shift(DOWN*0.4 + RIGHT*0.5)
        machine = VGroup(machine_part1, machine_part2, machine_part3, machine_gear1, machine_gear2)
        # Fix for Issue 37: change area from B2-E5 to A2-E5
        self.place_in_area(machine, "A2", "E5", scale_factor=1.0)
        
        self.add(machine)
        self.play(Rotate(machine_gear1, angle=2*PI), Rotate(machine_gear2, angle=-2*PI), run_time=2)
        
        # Triangle Vertices (Top B3.5, BL E2, BR E5)
        top_pos = (self.grid["B3"] + self.grid["B4"]) / 2
        bl_pos = self.grid["E2"]
        br_pos = self.grid["E5"]
        
        triangle_outline = Polygon(top_pos, bl_pos, br_pos, color=GOLDEN, stroke_width=6)
        triangle_fill = triangle_outline.copy().set_fill(GOLDEN, opacity=0.2)
        
        self.play(
            FadeOut(machine),
            Create(triangle_outline),
            FadeIn(triangle_fill),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Complex identities become simple movements across the shape.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GOLDEN)
        
        # Movement along the triangle edges
        path_dot = Dot(color=WHITE, radius=0.1)
        path_dot.move_to(bl_pos)
        
        # Define edge lines for MoveAlongPath
        edge1 = Line(bl_pos, top_pos)
        edge2 = Line(top_pos, br_pos)
        edge3 = Line(br_pos, bl_pos)
        
        self.play(FadeIn(path_dot))
        # Cycle through the vertices
        self.play(MoveAlongPath(path_dot, edge1), run_time=1)
        self.play(MoveAlongPath(path_dot, edge2), run_time=1)
        self.play(MoveAlongPath(path_dot, edge3), run_time=1)
        self.play(FadeOut(path_dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Math is about patterns, not just memorizing symbols.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(PIXEL_COLOR)
        
        # Create Pixel the Penguin (using Manim shapes)
        p_body = Ellipse(width=0.8, height=1.0, color=PIXEL_COLOR, fill_opacity=1)
        p_belly = Ellipse(width=0.5, height=0.7, color=BELLY_COLOR, fill_opacity=1).shift(DOWN*0.1)
        p_eye_l = Dot(radius=0.05, color=BLACK).shift(LEFT*0.15 + UP*0.2)
        p_eye_r = Dot(radius=0.05, color=BLACK).shift(RIGHT*0.15 + UP*0.2)
        p_beak = Triangle(color=BEAK_COLOR, fill_opacity=1).scale(0.1).rotate(PI).shift(UP*0.05)
        pixel = VGroup(p_body, p_belly, p_eye_l, p_eye_r, p_beak)
        
        # Position Pixel to the right of the triangle
        # Fix for Issue 36: change grid pos from E6 to F6
        self.place_at_grid(pixel, "F6", scale_factor=0.8)
        
        # Happy entrance and jump animation
        self.play(FadeIn(pixel, shift=UP))
        for _ in range(2):
            self.play(pixel.animate.shift(UP*0.3), run_time=0.4, rate_func=rate_functions.there_and_back)
        
        self.wait(2)
