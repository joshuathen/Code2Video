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
        # Setup layout with the requested lecture lines
        lecture_lines = [
            "Light travels faster through air than through dense glass.",
            "We see light slow down as it enters glass.",
            "The index of refraction calculates this speed change."
        ]
        self.setup_layout("Prerequisite Knowledge: Speed and Optical Density", lecture_lines)

        # Define colors for lecture steps
        STEP_COLORS = [YELLOW, TEAL, GREEN]

        # === Animation for Lecture Line 1 ===
        # Background regions for Air and Glass
        # Air region (A1-C6)
        air_box = Rectangle(width=5.8, height=2.8, fill_opacity=0.1, color=WHITE, stroke_width=0)
        self.place_in_area(air_box, "A1", "C6")
        
        # Glass region (D1-F6)
        glass_box = Rectangle(width=5.8, height=2.8, fill_opacity=0.2, color="#E0FFFF", stroke_width=0)
        self.place_in_area(glass_box, "D1", "F6")
        
        # Asset integration: Glass icon representing the medium
        glass_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/glass.svg")
        glass_asset.set_color("#E0FFFF")
        self.place_in_area(glass_asset, "D2", "E2", scale_factor=0.6)

        # Labels for media with updated positioning/scaling
        air_label = Text("Air", font_size=20, color=WHITE)
        self.place_at_grid(air_label, "A1", scale_factor=0.8)
        
        glass_label = Text("Glass", font_size=20, color="#E0FFFF")
        self.place_at_grid(glass_label, "D1", scale_factor=0.8)

        # Starting light pulses
        pulse_air = Dot(color=WHITE, radius=0.12)
        self.place_at_grid(pulse_air, "B1")
        
        pulse_glass = Dot(color=WHITE, radius=0.12)
        self.place_at_grid(pulse_glass, "E1")

        self.play(
            self.lecture[0].animate.set_color(STEP_COLORS[0]),
            FadeIn(air_box),
            FadeIn(glass_box),
            FadeIn(glass_asset),
            FadeIn(air_label),
            FadeIn(glass_label),
            FadeIn(pulse_air),
            FadeIn(pulse_glass)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pulses move: Air covers full distance, Glass covers partial distance (slower)
        self.play(
            self.lecture[1].animate.set_color(STEP_COLORS[1]),
            pulse_air.animate.move_to(self.grid["B6"]),
            pulse_glass.animate.move_to(self.grid["E4"]),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Index of Refraction formula: n = c / v
        n_text = Text("n", color=WHITE, font_size=32)
        eq_text = Text("=", color=WHITE, font_size=32)
        c_text = Text("c", color=WHITE, font_size=32)
        v_text = Text("v", color=WHITE, font_size=32)
        frac_line = Line(LEFT, RIGHT, color=WHITE).set_length(0.35)
        
        # Build formula fraction n = c/v
        fraction = VGroup(c_text, frac_line, v_text).arrange(DOWN, buff=0.1)
        formula = VGroup(n_text, eq_text, fraction).arrange(RIGHT, buff=0.15)
        
        # Place formula in F4-F6 to avoid line overlap as requested
        self.place_in_area(formula, 'F4', 'F6', scale_factor=0.7)

        # Display formula and highlight 'v' in green
        self.play(
            self.lecture[2].animate.set_color(STEP_COLORS[2]),
            FadeIn(formula)
        )
        self.play(
            v_text.animate.set_color("#00FF00"),
            run_time=1
        )
        self.wait(2)
