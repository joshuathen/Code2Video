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

class Section5Scene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            "We extend the function into the infinite complex plane.",
            "Shading marks the initial region where the sum converges.",
            "Analytic continuation expands the function's reach across the map.",
            "Visualize magnitudes as mountains and valleys in this landscape.",
            "A single point at s equals one remains undefined."
        ]
        self.setup_layout("Entering the Complex Realm (Analytic Continuation)", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # A 2D complex plane appears with the horizontal axis σ and vertical axis 'it'.
        self.lecture[0].set_color(YELLOW)
        
        # Define Axes based on grid: horizontal along Row D, vertical along Col 4
        # D4 is the origin (0,0) in the right-side grid.
        sigma_axis = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        it_axis = Line(self.grid["F4"], self.grid["A4"], color=WHITE)
        
        # Labels for axes
        sigma_label = Text("σ", font_size=24, color=WHITE)
        self.place_at_grid(sigma_label, "D6", scale_factor=0.8)
        sigma_label.shift(RIGHT * 0.3)
        
        it_label = Text("it", font_size=24, slant=ITALIC, color=WHITE)
        self.place_at_grid(it_label, "A4", scale_factor=0.8)
        it_label.shift(UP * 0.3)
        
        plane_group = VGroup(sigma_axis, it_axis, sigma_label, it_label)
        self.play(Create(plane_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The region where σ > 1 is shaded in light blue (#ADD8E6) to show the initial domain.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Convergence region: σ > 1 corresponds to area from column 5 onwards.
        # The area spans from A5 to F6.
        convergence_shade = Rectangle(
            width=1.5, height=5.5, 
            fill_color="#ADD8E6", fill_opacity=0.4, stroke_width=0
        )
        self.place_in_area(convergence_shade, "A5", "F6")
        
        # Fix Issue 42: Move conv_label to A5-B6 area
        conv_label = Text("Convergence: σ > 1", font_size=18, color="#ADD8E6")
        self.place_in_area(conv_label, 'A5', 'B6', scale_factor=0.8)
        
        self.play(FadeIn(convergence_shade), Write(conv_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Analytic continuation expands the function's reach across the map.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Expanded reach covering the whole grid area A1 to F6
        full_shade = Rectangle(
            width=5.5, height=5.5, 
            fill_color="#ADD8E6", fill_opacity=0.2, stroke_width=0
        )
        self.place_in_area(full_shade, "A1", "F6")
        
        # Fix Issue 41: Move continuation_label to A1-B2 area
        continuation_label = Text("Analytic Continuation", font_size=20, color=WHITE)
        self.place_in_area(continuation_label, 'A1', 'B2', scale_factor=0.8)
        
        self.play(
            ReplacementTransform(convergence_shade, full_shade),
            FadeOut(conv_label),
            Write(continuation_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Visualize magnitudes as mountains and valleys in this landscape.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Represent landscape features as colorful blobs (simulating magnitude |ζ(s)|)
        # Fix Issue 43: Move mountain to C2
        mountain = VGroup(
            Circle(radius=0.7, color=ORANGE, fill_opacity=0.3, stroke_width=0),
            Circle(radius=0.3, color=YELLOW, fill_opacity=0.6, stroke_width=0)
        )
        self.place_at_grid(mountain, 'C2', scale_factor=0.7)
        
        # Valley at E2
        valley = VGroup(
            Circle(radius=0.6, color=BLUE_C, fill_opacity=0.3, stroke_width=0),
            Circle(radius=0.2, color=BLUE_E, fill_opacity=0.6, stroke_width=0)
        )
        self.place_at_grid(valley, "E2", scale_factor=0.8)
        
        # Additional features for "heat map" look
        ridge = Circle(radius=0.5, color=RED, fill_opacity=0.2, stroke_width=0)
        self.place_at_grid(ridge, "B4", scale_factor=0.9)
        
        lake = Circle(radius=0.4, color=PURPLE, fill_opacity=0.2, stroke_width=0)
        self.place_at_grid(lake, "F5", scale_factor=0.6)
        
        landscape = VGroup(mountain, valley, ridge, lake)
        self.play(FadeIn(landscape, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A single point at s equals one remains undefined.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Pole at s = 1. Row D, Column 5.
        pole_dot = Dot(color="#FFFFFF", radius=0.12)
        self.place_at_grid(pole_dot, "D5")
        
        # Glow effect
        pole_glow = Circle(radius=0.25, color=WHITE, fill_opacity=0.2, stroke_width=0)
        self.place_at_grid(pole_glow, "D5")
        
        # Label at C5
        pole_label = Text("s = 1", font_size=24, color=WHITE)
        self.place_at_grid(pole_label, "C5", scale_factor=1.0)
        
        self.play(FadeIn(pole_dot), FadeIn(pole_glow))
        self.play(Write(pole_label))
        self.play(Indicate(pole_dot, color=YELLOW), run_time=1.5)
        self.wait(2)

        # Final state
        self.lecture[4].set_color(WHITE)
        self.wait(2)
