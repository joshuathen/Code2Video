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
        # Setup the scene layout
        lines = [
            'The Reynolds number determines when smooth flow turns turbulent.', 
            "It weighs inertial forces against the fluid's internal friction.", 
            'Increasing this ratio transforms steady streams into chaotic rivers.'
        ]
        self.setup_layout("Prerequisite: The Gateway to Turbulence (Reynolds Number)", lines)

        # Colors for matching lecture lines
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00BFFF" # Deep Sky Blue
        COLOR_3 = "#FF0000" # Red

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Build Reynolds Equation: Re = (rho * v * L) / mu
        re_text = Text("Re =", font_size=40, color=WHITE)
        num_text = Text("\u03C1 v L", font_size=36, color=WHITE) # \u03C1 is rho
        den_text = Text("\u03BC", font_size=36, color=WHITE)     # \u03BC is mu
        frac_line = Line(LEFT, RIGHT, stroke_width=2, color=WHITE).set_width(1.2)
        
        num_v = VGroup(num_text).shift(UP * 0.4)
        den_v = VGroup(den_text).shift(DOWN * 0.4)
        fraction = VGroup(num_v, den_v, frac_line)
        
        formula = VGroup(re_text, fraction).arrange(RIGHT, buff=0.3)
        # Fix Issue 31: Adjusted formula width from A2-B5 to A2-B4
        self.place_in_area(formula, "A2", "B4", scale_factor=1.0)
        
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Labels for Inertial and Viscous forces
        inertia_label = Text("Inertial Forces", font_size=24, color=COLOR_2)
        viscous_label = Text("Viscous Forces", font_size=24, color=COLOR_2)
        
        # Fix Issue 32: Use place_in_area for inertia_label
        self.place_in_area(inertia_label, "A5", "A6", scale_factor=0.7)
        # Fix Issue 33: Use place_in_area for viscous_label
        self.place_in_area(viscous_label, "B5", "B6", scale_factor=0.7)
        
        arrow_up = Arrow(inertia_label.get_left(), num_text.get_right(), color=COLOR_2, buff=0.1)
        arrow_down = Arrow(viscous_label.get_left(), den_text.get_right(), color=COLOR_2, buff=0.1)
        
        self.play(
            FadeIn(inertia_label),
            FadeIn(viscous_label),
            Create(arrow_up),
            Create(arrow_down)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_3))

        # Slider Setup
        slider_line = Line(self.grid["F2"], self.grid["F5"], color=WHITE)
        slider_knob = Dot(self.grid["F2"], color=COLOR_3, radius=0.15)
        label_laminar = Text("Laminar", font_size=20).next_to(slider_line, LEFT, buff=0.2)
        label_turbulent = Text("Turbulent", font_size=20).next_to(slider_line, RIGHT, buff=0.2)
        
        # Asset Integration (Issue 26)
        river_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/river.svg")
        self.place_in_area(river_icon, "D3", "E4", scale_factor=0.6)
        river_icon.set_color(COLOR_3)

        # Flow Visualization
        # Laminar lines (parallel)
        laminar_lines = VGroup(*[
            Line(self.grid["C2"] + UP*i*0.3, self.grid["C5"] + UP*i*0.3, color=WHITE)
            for i in range(-2, 3)
        ])
        
        # Turbulent lines (tangled, red)
        turbulent_lines = VGroup()
        for i in range(-2, 3):
            start = self.grid["C2"] + UP*i*0.3
            end = self.grid["C5"] + UP*i*0.3
            # Add some "chaotic" control points
            mid1 = start + RIGHT*1.0 + UP*np.random.uniform(-0.5, 0.5)
            mid2 = start + RIGHT*2.0 + DOWN*np.random.uniform(-0.5, 0.5)
            curve = CubicBezier(start, mid1, mid2, end, color=COLOR_3, stroke_width=2)
            turbulent_lines.add(curve)

        self.play(
            Create(slider_line),
            FadeIn(slider_knob),
            FadeIn(label_laminar),
            FadeIn(label_turbulent),
            Create(laminar_lines)
        )
        
        # Animate the transition with asset integration
        self.play(
            slider_knob.animate.move_to(self.grid["F5"]),
            ReplacementTransform(laminar_lines, turbulent_lines),
            FadeIn(river_icon),
            run_time=3,
            rate_func=linear
        )
        
        self.wait(2)
