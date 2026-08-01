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
        # Setup the layout with section title and lecture lines
        self.setup_layout("Prerequisite: The Gateway Number (Reynolds Number)", [
            "The Reynolds number compares inertial forces to viscous forces.",
            "High Reynolds numbers signify inertia dominating the fluid's stickiness.",
            "Beyond a critical threshold, flow inevitably becomes turbulent."
        ])
        
        # Initialize lecture lines as dimmed
        self.lecture.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Formula display: Re = rho*v*L / mu
        re_formula = MathTex(
            "Re", "=", "{ \\rho v L", "\\over", "\\mu }",
            font_size=42, color=WHITE
        )
        self.place_in_area(re_formula, "B3", "C5") 
        
        inertia_label = Text("Inertia", font_size=18, color="#90EE90")
        viscosity_label = Text("Viscosity", font_size=18, color="#FFB6C1")
        
        # Position labels relative to the grid
        self.place_at_grid(inertia_label, "A4")
        self.place_at_grid(viscosity_label, "D4")

        self.play(Write(re_formula))
        self.play(
            re_formula[2].animate.set_color("#90EE90"), # rho v L
            FadeIn(inertia_label),
            run_time=0.8
        )
        self.play(
            re_formula[4].animate.set_color("#FFB6C1"), # mu
            FadeIn(viscosity_label),
            run_time=0.8
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#ADD8E6")
        )
        
        # Clear previous elements to make room for simulation
        self.play(FadeOut(re_formula), FadeOut(inertia_label), FadeOut(viscosity_label))
        
        # Fast moving shape with wake
        # Fix for Issue 24: Position at B3 instead of B2
        high_re_obj = Circle(radius=0.15, color="#ADD8E6", fill_opacity=1)
        self.place_at_grid(high_re_obj, "B3")
        
        # Fix for Issue 23: Use area for multi-word label
        high_re_label = Text("High Re: Inertia Wins", font_size=20, color=WHITE)
        self.place_in_area(high_re_label, "A2", "A5", scale_factor=0.8)
        
        # Simple wake particles with updater
        wakes = VGroup(*[Dot(radius=0.03, color="#ADD8E6", fill_opacity=0.4) for _ in range(8)])
        for w in wakes:
            w.move_to(high_re_obj.get_center())
            
        def update_wakes(mob, dt):
            for dot in mob:
                # Move left relative to object and add chaotic jitter
                dot.shift(LEFT * 2.0 * dt + UP * np.random.uniform(-1.0, 1.0) * dt)
                # Reset if too far from object
                if np.linalg.norm(dot.get_center() - high_re_obj.get_center()) > 1.2:
                    dot.move_to(high_re_obj.get_center())

        wakes.add_updater(update_wakes)
        
        self.play(FadeIn(high_re_obj), FadeIn(high_re_label))
        self.add(wakes)
        # Move object from B3 to B6
        self.play(high_re_obj.animate.move_to(self.grid["B6"]), run_time=3.0, rate_func=linear)
        wakes.remove_updater(update_wakes)
        self.play(FadeOut(high_re_obj), FadeOut(wakes), FadeOut(high_re_label))

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#ADD8E6")
        )
        
        # Slow moving shape in a straight line
        # Fix for Issue 24: Position at E3 instead of E2
        low_re_obj = Circle(radius=0.12, color="#ADD8E6", fill_opacity=1)
        self.place_at_grid(low_re_obj, "E3")
        
        # Fix for Issue 23: Use area for multi-word label
        low_re_label = Text("Low Re: Viscosity Wins", font_size=20, color=WHITE)
        self.place_in_area(low_re_label, "D2", "D5", scale_factor=0.8)
        
        # Background straight path
        path_start = self.grid["E3"]
        path_end = self.grid["E6"]
        path = Line(path_start, path_end, color=WHITE, stroke_opacity=0.2)
        
        self.play(Create(path), FadeIn(low_re_obj), FadeIn(low_re_label))
        # Move object slowly to simulate laminar/viscous flow
        self.play(low_re_obj.animate.move_to(path_end), run_time=4.0, rate_func=linear)
        self.wait(2)
