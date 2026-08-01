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
        # Initialize Scene
        title = "The Topological Twist: Creating the Möbius Strip"
        lines = [
            "Unordered pairs create a mathematical twist.",
            "This configuration space forms a Möbius strip.",
            "The curve forms the strip's single boundary.",
            "This topological property is key to our proof.",
            "We seek intersections within this twisted space."
        ]
        self.setup_layout(title, lines)

        # Colors
        LAVENDER = "#E6E6FA"
        RED = "#FF0000"
        YELLOW = "#FFFF00"
        HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show a flat rectangular strip in lavender (#E6E6FA) in a 3D-like perspective.
        # Mark the left and right edges with red (#FF0000) arrows pointing in opposite directions.
        self.lecture[0].set_color(HIGHLIGHT)
        
        strip_rect = Rectangle(width=3.5, height=1.0, fill_color=LAVENDER, fill_opacity=0.8, stroke_color=WHITE)
        # Resolved Issue 40: Changed area from B2-E5 to B1-E6
        self.place_in_area(strip_rect, "B1", "E6", scale_factor=1.0)
        
        # Adding arrows to indicate orientation flip
        arrow_left = Arrow(start=strip_rect.get_left() + DOWN*0.4, end=strip_rect.get_left() + UP*0.4, color=RED, buff=0)
        arrow_right = Arrow(start=strip_rect.get_right() + UP*0.4, end=strip_rect.get_right() + DOWN*0.4, color=RED, buff=0)
        
        self.play(FadeIn(strip_rect), GrowArrow(arrow_left), GrowArrow(arrow_right))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the strip twisting 180 degrees in the center.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT)

        # We represent the twist by transforming the rectangle into a path that crosses itself
        # simulating a 180-degree flip.
        twisted_shape = VGroup()
        # Top segment (twisted)
        top_twisted = ParametricFunction(
            lambda t: np.array([t, 0.3 * np.sin(PI * t / 1.5), 0]),
            t_range=[-1.5, 1.5], color=LAVENDER
        )
        # Bottom segment (twisted)
        bottom_twisted = ParametricFunction(
            lambda t: np.array([t, -0.3 * np.sin(PI * t / 1.5), 0]),
            t_range=[-1.5, 1.5], color=LAVENDER
        )
        twisted_group = VGroup(top_twisted, bottom_twisted).scale(1.2)
        # Resolved Issue 41: Changed area from B2-E5 to B1-E6
        self.place_in_area(twisted_group, "B1", "E6", scale_factor=1.0)

        self.play(
            strip_rect.animate.become(twisted_group),
            arrow_left.animate.shift(RIGHT * 0.5),
            arrow_right.animate.shift(LEFT * 0.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Merge the edges to form a 3D Möbius strip.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT)

        # Parametric Mobius Strip representation (2D projection)
        def mobius_func(u, v):
            # u: 0 to 2PI, v: -0.3 to 0.3
            x = (1 + v * np.cos(u/2)) * np.cos(u)
            y = (1 + v * np.cos(u/2)) * np.sin(u)
            return np.array([x, y, 0])

        mobius_strip = VGroup()
        for v in np.linspace(-0.3, 0.3, 5):
            path = ParametricFunction(lambda u: mobius_func(u, v), t_range=[0, 2*PI], color=LAVENDER)
            mobius_strip.add(path)
        
        # Resolved Issue 42: Adjusted scale_factor to 1.1
        self.place_in_area(mobius_strip, "B2", "E5", scale_factor=1.1)

        self.play(
            ReplacementTransform(strip_rect, mobius_strip),
            FadeOut(arrow_left), FadeOut(arrow_right),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the single continuous boundary of the Möbius strip in yellow (#FFFF00).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT)

        # The boundary of a Mobius strip is a single closed loop
        # For our parametric v in [-0.3, 0.3], the boundary is u in [0, 4PI] with v fixed (effectively)
        # However, a simpler visual is to trace the outer edge.
        boundary = ParametricFunction(
            lambda t: mobius_func(t, 0.3),
            t_range=[0, 4*PI], # It takes 4PI to close the loop on the edge
            color=YELLOW,
            stroke_width=4
        )
        # Resolved Issue 42: Adjusted scale_factor to 1.1
        self.place_in_area(boundary, "B2", "E5", scale_factor=1.1)

        self.play(Create(boundary, run_time=3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We seek intersections within this twisted space.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT)

        # Pulse the strip to emphasize the "space"
        self.play(
            mobius_strip.animate.scale(1.1),
            boundary.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
