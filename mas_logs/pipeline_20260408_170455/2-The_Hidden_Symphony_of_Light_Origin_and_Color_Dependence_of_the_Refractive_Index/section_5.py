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
        self.setup_layout(
            "Mathematical Landscape: Cauchy’s Equation", 
            [
                "Cauchy’s equation relates refractive index to wavelength.", 
                "Shorter wavelengths experience a higher index of refraction.", 
                "This mathematical curve predicts how colors will bend."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Equation n(λ) = A + B / λ²
        # λ highlighted in cyan (#00FFFF)
        self.lecture[0].set_color("#00FFFF")
        
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        eq = Text(
            "n(λ) = A + B / λ²",
            color=WHITE, font_size=36,
            t2c={"λ": "#00FFFF"}
        )
        self.place_in_area(eq, "A2", "A5")
        
        self.play(Write(eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A plot of n vs λ appears. 
        # Blue point (#0000FF) high on the left; a red point (#FF0000) low on the right.
        self.lecture[1].set_color("#0000FF") # Shorter wavelength color
        
        axes = Axes(
            x_range=[400, 750, 100],
            y_range=[1.4, 1.8, 0.1],
            axis_config={"include_tip": True},
            x_length=4.5,
            y_length=3
        ).set_color(GRAY)
        
        # Replace Tex labels with Text labels
        x_label = axes.get_x_axis_label(Text("λ", font_size=24), edge=DOWN, direction=DOWN).scale(0.7)
        y_label = axes.get_y_axis_label(Text("n", font_size=24), edge=LEFT, direction=LEFT).scale(0.7)
        
        # Cauchy Curve: n = 1.45 + 40000 / lambda^2
        curve = axes.plot(
            lambda l: 1.45 + 40000 / (l**2),
            x_range=[400, 700],
            color=WHITE
        )
        
        graph_group = VGroup(axes, x_label, y_label, curve)
        self.place_in_area(graph_group, "B2", "D6", scale_factor=0.9)
        
        # Points
        # Blue: wavelength 400
        blue_dot = Dot(axes.c2p(400, 1.45 + 40000/(400**2)), color="#0000FF")
        blue_label = Text("Blue", font_size=16, color="#0000FF").next_to(blue_dot, UP, buff=0.1)
        
        # Red: wavelength 700
        red_dot = Dot(axes.c2p(700, 1.45 + 40000/(700**2)), color="#FF0000")
        red_label = Text("Red", font_size=16, color="#FF0000").next_to(red_dot, RIGHT, buff=0.1)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(curve))
        self.play(FadeIn(blue_dot), Write(blue_label))
        self.play(FadeIn(red_dot), Write(red_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A white beam hits a prism. 
        # Red and blue components emerge at different angles.
        self.lecture[2].set_color(WHITE)
        
        prism = Triangle().set_color(WHITE).set_stroke(width=2)
        self.place_in_area(prism, "E2", "F5", scale_factor=0.8)
        
        # Beam points
        p_center = prism.get_center()
        p_left_side = p_center + LEFT * 0.4 + DOWN * 0.2
        entry_point = p_left_side
        
        # Beams
        white_in = Line(entry_point + LEFT * 1.5 + UP * 0.5, entry_point, color=WHITE)
        
        # Internal paths (refraction)
        # Blue bends more (downwards)
        blue_internal = Line(entry_point, p_center + RIGHT * 0.3 + DOWN * 0.1, color="#0000FF")
        red_internal = Line(entry_point, p_center + RIGHT * 0.3 + UP * 0.05, color="#FF0000")
        
        # Outgoing paths
        blue_out = Line(blue_internal.get_end(), blue_internal.get_end() + RIGHT * 1.0 + DOWN * 0.4, color="#0000FF")
        red_out = Line(red_internal.get_end(), red_internal.get_end() + RIGHT * 1.0 + DOWN * 0.1, color="#FF0000")
        
        beams = VGroup(white_in, blue_internal, red_internal, blue_out, red_out)
        
        self.play(Create(prism))
        self.play(Create(white_in))
        self.play(
            Create(blue_internal), 
            Create(red_internal),
            run_time=1
        )
        self.play(
            Create(blue_out),
            Create(red_out),
            run_time=1
        )
        
        self.wait(3)
