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
        # Setup the scene
        title = "The Span: Reaching Every Corner"
        lines = [
            "- The span is the set of all reachable points.",
            "- Two non-parallel vectors can span an entire plane.",
            "- One vector alone only spans a single line."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_A = "#FFD700"  # Gold
        COLOR_B = "#00BFFF"  # Sky Blue
        COLOR_SPAN = "#404040" # Dark Grey
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        # Update lecture line color
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Vector A from D3 to D5 (horizontal)
        vec_a = Arrow(
            start=self.grid["D3"], 
            end=self.grid["D5"], 
            buff=0, 
            color=COLOR_A,
            stroke_width=6
        )
        label_a = MathTex("A", color=COLOR_A)
        # Fix for Issue 22: place label_a at D5
        self.place_at_grid(label_a, "D5", scale_factor=0.8)
        
        # Vector B from D3 to B3 (vertical)
        vec_b = Arrow(
            start=self.grid["D3"], 
            end=self.grid["B3"], 
            buff=0, 
            color=COLOR_B,
            stroke_width=6
        )
        label_b = MathTex("B", color=COLOR_B)
        # Fix for Issue 21: place label_b at B3
        self.place_at_grid(label_b, "B3", scale_factor=0.8)
        
        self.play(GrowArrow(vec_a), Write(label_a))
        self.play(GrowArrow(vec_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture line colors
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Issue 16: Use Asset for plane.svg
        span_plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        span_plane.set_color(COLOR_SPAN)
        span_plane.set_fill(opacity=0.3)
        
        # Fix for Issue 23: place span_plane in area B2 to F6
        self.place_in_area(span_plane, "B2", "F6", scale_factor=3.5)
        
        # Create many points representing linear combinations
        np.random.seed(42)
        origin = self.grid["D3"]
        dots = VGroup(*[
            Dot(
                point=[
                    origin[0] + np.random.uniform(-1, 3), 
                    origin[1] + np.random.uniform(-2, 2), 
                    0
                ],
                radius=0.03,
                color=COLOR_SPAN
            ) for _ in range(100)
        ])
        
        self.play(FadeIn(span_plane), FadeIn(dots))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Update lecture line colors
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Remove vector B and the plane visuals to focus on A
        self.play(
            FadeOut(vec_b),
            FadeOut(label_b),
            FadeOut(span_plane),
            FadeOut(dots)
        )
        
        # Issue 16: Use Asset for line.svg
        # The line should extend along the X-axis (span of vector A)
        span_line = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg")
        span_line.set_color(COLOR_A)
        # Position it to align with vector A (which is horizontal at row D)
        self.place_in_area(span_line, "D1", "D6", scale_factor=1.0)
        
        self.play(FadeIn(span_line))
        self.wait(2)
        
        # Final cleanup: reset lecture line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
