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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        lecture_lines = [
            'Derivatives help us analyze motion in the real world.',
            "The tangent slope gives the rocket's exact velocity.",
            'Calculus allows us to calculate the mathematics of change.'
        ]
        self.setup_layout("Application: The Speedometer of Life", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color("#3498DB"))

        # Create Rocket from Asset
        rocket = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/rocket.svg")
        rocket.set_color(WHITE)
        
        # Place rocket (Issue 43: use place_at_grid for single object)
        self.place_at_grid(rocket, 'E2', scale_factor=0.8)
        
        # Create Axes and Graph
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 15, 5],
            x_length=2.5,
            y_length=3.5,
            axis_config={"color": WHITE, "include_tip": True},
            tips=True
        )
        x_label = Text("Time", font_size=14).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_label = Text("Pos", font_size=14).next_to(axes.y_axis, UP, buff=0.1)
        axes_group = VGroup(axes, x_label, y_label)
        
        # Place graph area (Issue 42: avoid title and bottom text overlap)
        self.place_in_area(axes_group, 'B4', 'E6', scale_factor=0.8)
        
        # Define the position function y = e^x - 1
        graph = axes.plot(lambda x: np.exp(x) - 1, x_range=[0, 2.7], color="#3498DB")

        self.play(
            FadeIn(rocket),
            Create(axes_group),
            Create(graph),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#2ECC71")
        )

        # Draw tangent at t=2
        t_val = 2.0
        p_val = np.exp(t_val) - 1
        slope = np.exp(t_val)
        
        dot = Dot(axes.c2p(t_val, p_val), color="#2ECC71")
        
        # Tangent line equation: y = slope * (x - t_val) + p_val
        def tangent_func(x):
            return slope * (x - t_val) + p_val
        
        tangent_line = axes.plot(tangent_func, x_range=[t_val-0.4, t_val+0.4], color="#2ECC71")
        velocity_label = Text("Velocity", font_size=18, color="#2ECC71")
        # Issue 44: Use scale factor 0.8
        self.place_at_grid(velocity_label, 'B5', scale_factor=0.8)

        self.play(
            FadeIn(dot),
            Create(tangent_line),
            Write(velocity_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#F1C40F")
        )

        # Animate rocket moving up and tangent line getting steeper
        target_rocket_pos = self.grid["B2"]
        
        # Steeper tangent line at t=2.4
        t_val_2 = 2.4
        p_val_2 = np.exp(t_val_2) - 1
        slope_2 = np.exp(t_val_2)
        dot_2 = Dot(axes.c2p(t_val_2, p_val_2), color="#F1C40F")
        
        def tangent_func_2(x):
            return slope_2 * (x - t_val_2) + p_val_2
            
        tangent_line_2 = axes.plot(tangent_func_2, x_range=[t_val_2-0.3, t_val_2+0.3], color="#F1C40F")

        final_text = Text("Calculus is the Mathematics of Change", font_size=24, color="#F1C40F")
        # Final text spans bottom row
        self.place_in_area(final_text, "F1", "F6", scale_factor=1.0)

        self.play(
            rocket.animate.move_to(target_rocket_pos),
            ReplacementTransform(tangent_line, tangent_line_2),
            ReplacementTransform(dot, dot_2),
            velocity_label.animate.set_color("#F1C40F").next_to(dot_2, LEFT, buff=0.2),
            run_time=2
        )
        
        self.play(FadeIn(final_text))
        self.wait(3)
