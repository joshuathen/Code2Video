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
        # 1. Setup layout
        title = "Defining Rigor: The Epsilon-Delta Challenge"
        lines = [
            "Let's define the limit with mathematical precision.",
            "Epsilon sets a vertical target window around the limit.",
            "Can we find a horizontal window for our inputs?",
            "This delta window ensures all outputs stay inside epsilon.",
            "If it works for any epsilon, the limit exists."
        ]
        self.setup_layout(title, lines)

        # Colors
        eps_color = "#FFD700"
        delta_color = "#00BFFF"
        white = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(white))
        
        # Create Axes and Graph
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 3, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": white}
        )
        # Define a smooth curve: f(x) = 0.5x^2 + 0.5
        curve = axes.plot(lambda x: 0.5 * x**2 + 0.5, x_range=[0.2, 2.5], color=white)
        
        # Target Point (c, L) where c = 1.5, L = 0.5(1.5)^2 + 0.5 = 1.625
        c_val = 1.5
        L_val = 1.625
        target_point = Dot(axes.c2p(c_val, L_val), color=white)
        
        # Visual lines for c and L
        c_line = axes.get_vertical_line(axes.c2p(c_val, L_val), color=white)
        L_line = axes.get_horizontal_line(axes.c2p(c_val, L_val), color=white)
        
        c_label = Text("c", font_size=24, color=white).next_to(axes.c2p(c_val, 0), DOWN, buff=0.1)
        L_label = Text("L", font_size=24, color=white).next_to(axes.c2p(0, L_val), LEFT, buff=0.1)
        
        # Group and place in area B1 to F6 (Issue 27 Fix)
        graph_elements = VGroup(axes, curve, target_point, c_line, L_line, c_label, L_label)
        self.place_in_area(graph_elements, "B1", "F6", scale_factor=0.7)
        
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(target_point), Create(c_line), Create(L_line), Write(c_label), Write(L_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(eps_color))
        
        # Epsilon tolerance around L
        epsilon = 0.5
        y_top = L_val + epsilon
        y_bottom = L_val - epsilon
        
        eps_band_height = abs(axes.c2p(0, y_top)[1] - axes.c2p(0, y_bottom)[1])
        
        eps_band = Rectangle(
            width=axes.x_length * 0.9, 
            height=eps_band_height,
            fill_color=eps_color,
            fill_opacity=0.2,
            stroke_width=0
        ).move_to(axes.c2p(c_val, L_val))
        
        # Epsilon labels (Issue 26 Fix)
        L_plus_eps = Text("L + ε", font_size=20, color=eps_color)
        L_minus_eps = Text("L - ε", font_size=20, color=eps_color)
        epsilon_labels = VGroup(L_plus_eps, L_minus_eps).arrange(DOWN, buff=0.6)
        self.place_at_grid(epsilon_labels, 'C1', scale_factor=0.5)
        
        self.play(FadeIn(eps_band), Write(epsilon_labels))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(delta_color))
        
        # Delta window around c
        delta = 0.3
        x_right = c_val + delta
        x_left = c_val - delta
        
        delta_band_width = abs(axes.c2p(x_right, 0)[0] - axes.c2p(x_left, 0)[0])
        
        delta_band = Rectangle(
            width=delta_band_width,
            height=axes.y_length * 0.9,
            fill_color=delta_color,
            fill_opacity=0.2,
            stroke_width=0
        ).move_to(axes.c2p(c_val, L_val))
        
        # Delta labels (Issue 25 Fix)
        c_plus_delta = Text("c + δ", font_size=20, color=delta_color)
        c_minus_delta = Text("c - δ", font_size=20, color=delta_color)
        delta_labels = VGroup(c_minus_delta, c_plus_delta).arrange(RIGHT, buff=0.6)
        self.place_at_grid(delta_labels, 'F4', scale_factor=0.4)

        self.play(FadeIn(delta_band), Write(delta_labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(white))
        
        # Highlight the rectangular intersection
        intersection_rect = Rectangle(
            width=delta_band.width,
            height=eps_band.height,
            color=white,
            stroke_width=2
        ).move_to(axes.c2p(c_val, L_val))
        
        self.play(Create(intersection_rect))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(white))
        
        # Shrink the Epsilon band and the Delta band
        new_epsilon = 0.25
        new_delta = 0.15 
        
        new_eps_height = abs(axes.c2p(0, L_val + new_epsilon)[1] - axes.c2p(0, L_val - new_epsilon)[1])
        new_delta_width = abs(axes.c2p(c_val + new_delta, 0)[0] - axes.c2p(c_val - new_delta, 0)[0])
        
        self.play(
            eps_band.animate.stretch_to_fit_height(new_eps_height).move_to(axes.c2p(c_val, L_val)),
            delta_band.animate.stretch_to_fit_width(new_delta_width).move_to(axes.c2p(c_val, L_val)),
            intersection_rect.animate.stretch_to_fit_width(new_delta_width).stretch_to_fit_height(new_eps_height).move_to(axes.c2p(c_val, L_val)),
            run_time=2
        )
        self.wait(2)
