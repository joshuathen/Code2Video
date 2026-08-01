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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize the scene layout
        lecture_lines = [
            "A sphere’s surface area follows a surprising rule.",
            "The surface area is four times the shadow’s area.",
            "We write this formula as four π r squared.",
            "Peeling a sphere fills four circles of radius r.",
            "Archimedes first discovered this elegant 4-to-1 ratio."
        ]
        self.setup_layout("The Core Revelation: The 4-to-1 Ratio", lecture_lines)
        
        # Dim all lecture lines initially
        self.lecture.set_color(GREY_D)

        # Pre-create objects
        # Sphere's total surface area representation
        sphere_surface_rep = Circle(radius=0.4, color="#FFFFFF", fill_opacity=0.8)
        self.place_in_area(sphere_surface_rep, 'B2', 'C3', scale_factor=0.8)
        
        # Circular shadow
        grey_shadow = Circle(radius=0.4, color="#808080", fill_opacity=0.4)
        self.place_in_area(grey_shadow, 'B5', 'C5', scale_factor=0.8)
        shadow_label = Text("Shadow", font_size=16, color="#808080")
        self.place_at_grid(shadow_label, 'D5')

        # Four orange circles representing the "peeled" surface
        orange_circles = VGroup(*[
            Circle(radius=0.4, color="#FFA500", fill_opacity=0.8) for _ in range(4)
        ])
        # Grid positions for the 4 circles
        grid_pos_list = ['B2', 'B3', 'C2', 'C3']
        for i, pos in enumerate(grid_pos_list):
            self.place_at_grid(orange_circles[i], pos)

        # Formula - Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        formula = Text("Surface Area = 4πr²", font_size=24, color=WHITE)
        self.place_in_area(formula, 'E2', 'E5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        # A sphere’s surface area follows a surprising rule.
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(sphere_surface_rep), FadeIn(grey_shadow), FadeIn(shadow_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The surface area is four times the shadow’s area.
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        # Animate the sphere's surface splitting into four identical orange circles.
        self.play(
            ReplacementTransform(sphere_surface_rep, orange_circles),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We write this formula as four π r squared.
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Peeling a sphere fills four circles of radius r.
        self.play(self.lecture[3].animate.set_color("#FFA500"))
        # Overlap one orange circle onto the shadow to show they match.
        # Calculate shadow position dynamically to ensure correct overlap regardless of grid tweaks
        shadow_pos = grey_shadow.get_center()
        self.play(
            orange_circles[1].animate.move_to(shadow_pos),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Archimedes first discovered this elegant 4-to-1 ratio.
        self.play(self.lecture[4].animate.set_color(WHITE))
        # Final highlight: flashing the formula and circles
        self.play(
            orange_circles.animate.set_stroke(YELLOW, width=3),
            formula.animate.scale(1.1),
            rate_func=there_and_back
        )
        self.wait(2)
