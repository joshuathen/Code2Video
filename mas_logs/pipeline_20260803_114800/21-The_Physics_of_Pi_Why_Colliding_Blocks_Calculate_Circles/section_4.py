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
        self.setup_layout("Mapping Physics to Geometry", [
            "Let's map block velocities onto a coordinate graph.",
            "Rescaled velocities must follow the energy conservation law.",
            "This energy constraint forms a perfect circle.",
            "Each collision represents a step along this arc.",
            "Physics is transforming into a geometric puzzle."
        ])
        
        # Colors
        GREEN_COLOR = "#00FF00"
        RED_COLOR = "#FF0000"
        WHITE_COLOR = "#FFFFFF"
        GRAY_COLOR = "#888888"

        # Initialize all lecture lines to gray
        self.lecture.set_color(GRAY_COLOR)

        # === Animation for Lecture Line 1 ===
        # Let's map block velocities onto a coordinate graph.
        self.lecture[0].set_color(WHITE_COLOR)
        
        axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=3.5,
            y_length=3.5,
            axis_config={"color": WHITE_COLOR, "include_tip": True},
            tips=False
        )
        # Using grids for labels to be strictly compliant
        x_label = MathTex("v", color=WHITE_COLOR).scale(0.7)
        y_label = MathTex("V", color=WHITE_COLOR).scale(0.7)
        
        # Place graph in area B2 to E5
        self.place_in_area(axes, "B2", "E5")
        
        # Position labels at specific grid points near the axes tips
        self.place_at_grid(x_label, "D6", scale_factor=0.8)
        self.place_at_grid(y_label, "A3", scale_factor=0.8)
        
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rescaled velocities must follow the energy conservation law.
        self.lecture[0].set_color(GRAY_COLOR)
        self.lecture[1].set_color(WHITE_COLOR)
        
        energy_formula = MathTex("v^2 + V^2 = E", color=WHITE_COLOR)
        # FIX ISSUE 25: Place at A5
        self.place_at_grid(energy_formula, "A5", scale_factor=0.8)
        
        self.play(Write(energy_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This energy constraint forms a perfect circle.
        self.lecture[1].set_color(GRAY_COLOR)
        self.lecture[2].set_color(GREEN_COLOR)
        
        # Calculate radius based on the axes conversion
        p_origin = axes.c2p(0, 0)
        p_unit = axes.c2p(1, 0)
        radius = np.linalg.norm(p_unit - p_origin)
        
        circle = Circle(radius=radius, color=GREEN_COLOR)
        circle.move_to(p_origin)
        
        self.play(
            Create(circle),
            energy_formula.animate.set_color(GREEN_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Each collision represents a step along this arc.
        self.lecture[2].set_color(GRAY_COLOR)
        self.lecture[3].set_color(RED_COLOR)
        
        # Starting point on the circle
        start_angle = PI/6
        p1 = axes.c2p(np.cos(start_angle), np.sin(start_angle))
        dot = Dot(p1, color=RED_COLOR, radius=0.1)
        
        # Next point (simulated collision)
        step_angle = 0.6
        p2 = axes.c2p(np.cos(start_angle + step_angle), np.sin(start_angle + step_angle))
        connecting_line = Line(p1, p2, color=WHITE_COLOR)
        
        self.play(FadeIn(dot))
        self.wait(0.5)
        self.play(
            dot.animate.move_to(p2),
            Create(connecting_line)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Physics is transforming into a geometric puzzle.
        self.lecture[3].set_color(GRAY_COLOR)
        self.lecture[4].set_color(RED_COLOR)
        
        # Rapidly add segments around the circle
        current_angle = start_angle + step_angle
        num_steps = 8
        
        for _ in range(num_steps):
            next_angle = current_angle + step_angle
            next_p = axes.c2p(np.cos(next_angle), np.sin(next_angle))
            current_p = axes.c2p(np.cos(current_angle), np.sin(current_angle))
            
            step_line = Line(current_p, next_p, color=WHITE_COLOR)
            
            self.play(
                dot.animate.move_to(next_p),
                Create(step_line),
                run_time=0.3
            )
            current_angle = next_angle
            
        self.wait(2)
