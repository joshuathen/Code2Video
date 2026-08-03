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
        # 1. Setup Title and Lecture Lines
        title_text = "Prerequisite: The Reynolds Number (Re)"
        lecture_lines = [
            "Reynolds number compares inertial forces to viscous forces.",
            "High inertia creates the violent, chaotic wake.",
            "Viscosity tries to dampen the fluid's motion."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        INERTIA_COLOR = "#808080"  # Heavy Gray
        VISCOSITY_COLOR = "#6699FF" # Light Blue for visibility
        FORMULA_COLOR = "#FFFFFF"  # White
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line (stays white as per formula color)
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.1)
        
        # Display the formula Re = uL/nu in white
        formula = MathTex(r"Re = \frac{uL}{\nu}", color=FORMULA_COLOR)
        # Fix for Issue 29: Place formula in the top-right area (A4-B6)
        self.place_in_area(formula, "A4", "B6", scale_factor=1.1)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update Lecture Line 2 Color to match Inertia
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(INERTIA_COLOR),
            run_time=0.5
        )

        # Scale Setup
        # Pivot point (E4)
        pivot_point = self.grid["E4"].copy()
        pivot_tri = Triangle(color=WHITE, fill_opacity=1.0).scale(0.15)
        pivot_tri.move_to(pivot_point + np.array([0.0, -0.15, 0.0]))
        
        # Beam (E3 to E5)
        beam = Line(self.grid["E3"].copy(), self.grid["E5"].copy(), color=WHITE, stroke_width=4)
        
        # Inertia Weight (gray weight)
        inertia_box = Square(side_length=0.6, color=INERTIA_COLOR, fill_opacity=1.0)
        inertia_text = Text("Inertia", font_size=16, color=WHITE)
        inertia_weight = VGroup(inertia_box, inertia_text)
        
        # Fix for Issue 30: Initial position of weight at D3
        self.place_at_grid(inertia_weight, "D3", scale_factor=1.0)
        
        # Viscosity drop (on the right end of the beam)
        viscosity_drop = Circle(radius=0.15, color=VISCOSITY_COLOR, fill_opacity=1.0)
        viscosity_label = Text("Viscosity", font_size=16, color=VISCOSITY_COLOR)
        viscosity_item = VGroup(viscosity_drop, viscosity_label).arrange(DOWN, buff=0.1)
        # Position viscosity item at E5 with offset to sit on beam
        viscosity_item.move_to(self.grid["E5"].copy() + np.array([0.0, 0.3, 0.0]))

        # Add pivot, beam and viscosity drop
        self.play(Create(pivot_tri), Create(beam), FadeIn(viscosity_item))
        
        # Drop onto the left side of the beam (E3)
        inertia_target = self.grid["E3"].copy() + np.array([0.0, 0.3, 0.0])
        self.play(inertia_weight.animate.move_to(inertia_target), run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Update Lecture Line 3 Color to match Viscosity
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(VISCOSITY_COLOR),
            run_time=0.5
        )

        # Scale assembly for tipping
        scale_assembly = VGroup(beam, inertia_weight, viscosity_item)
        
        # Rotate around the pivot point (tilt left side down)
        tip_angle = 15.0 * DEGREES
        
        self.play(
            Rotate(scale_assembly, angle=tip_angle, about_point=pivot_point),
            run_time=1.5
        )
        self.wait(2.0)

        # Final cleanup: Reset all lecture lines to white
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)
        self.wait(1.0)
