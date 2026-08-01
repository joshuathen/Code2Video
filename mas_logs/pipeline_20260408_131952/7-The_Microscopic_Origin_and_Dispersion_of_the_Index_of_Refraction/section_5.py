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
        # Setup the layout with specific lecture lines
        self.setup_layout(
            "Visualizing the Math: Cauchy’s Equation",
            [
                "Cauchy’s equation relates refractive index to wavelength.",
                "As wavelength increases, the refractive index decreases.",
                "Violet light bends more than red light."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Cauchy's Equation 'n(lambda) = A + B / lambda^2'
        self.lecture[0].set_color(WHITE)
        
        # Using Text with Unicode characters to avoid the FileNotFoundError: 'latex'
        equation = Text("n(λ) = A + B / λ²", color=WHITE)
        self.place_in_area(equation, "A2", "B5", scale_factor=1.0)
        
        label_n = Text("Refractive Index", font_size=18, color=BLUE_A)
        label_lambda = Text("Wavelength", font_size=18, color=YELLOW_A)
        
        # Position labels near relevant parts of the equation
        # Index 0 is 'n', Index -2 is 'λ' (in n(λ) = A + B / λ²)
        label_n.next_to(equation[0], UP, buff=0.2)
        label_lambda.next_to(equation[-2], DOWN, buff=0.2)
        
        eq_group = VGroup(equation, label_n, label_lambda)
        
        self.play(Write(eq_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # A graph appears with axes and a downward sloping curve
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Fade equation group to top-left to make room for graph
        self.play(eq_group.animate.scale(0.6).move_to(self.grid["A5"]))

        # Create Axes
        # Explicitly passing Text mobjects to avoid internal MathTex calls
        axes = Axes(
            x_range=[300, 800, 100],
            y_range=[1.4, 1.7, 0.1],
            x_length=4.5,
            y_length=3.0,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        x_label = axes.get_x_axis_label(Text("λ"), edge=DOWN, direction=DOWN, buff=0.1).scale(0.8)
        y_label = axes.get_y_axis_label(Text("n"), edge=LEFT, direction=LEFT, buff=0.1).scale(0.8)
        
        # Cauchy Curve: n = 1.5 + 40000 / lambda^2 (simplified coefficients for visual)
        graph = axes.plot(
            lambda x: 1.5 + 40000 / (x**2),
            x_range=[350, 750],
            color="#00FF00"
        )
        
        graph_group = VGroup(axes, x_label, y_label, graph)
        self.place_in_area(graph_group, "B1", "E6", scale_factor=0.8)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Red/Violet dots and Prism
        self.play(self.lecture[2].animate.set_color("#8B00FF"))

        # Red dot (longer wavelength, lower n) - around 700nm
        p_red = axes.c2p(700, 1.5 + 40000/(700**2))
        dot_red = Dot(p_red, color="#FF0000")
        
        # Violet dot (shorter wavelength, higher n) - around 400nm
        p_violet = axes.c2p(400, 1.5 + 40000/(400**2))
        dot_violet = Dot(p_violet, color="#8B00FF")

        self.play(FadeIn(dot_red), FadeIn(dot_violet))

        # Prism Asset
        try:
            prism = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/prism.svg")
        except:
            # Fallback if asset is missing for some reason during local dev
            prism = Triangle().set_fill(BLUE, opacity=0.3).set_stroke(WHITE)
            
        self.place_at_grid(prism, "F3", scale_factor=0.6)
        
        # Light Rays
        # Beam entering from left
        in_point = prism.get_left() + LEFT * 1.0
        entry_point = prism.get_left()
        beam_in = Line(in_point, entry_point, color=WHITE, stroke_width=2)
        
        # Exit points - violet bends more (lower)
        # Assuming prism orientation: tip up
        exit_base = prism.get_right()
        out_red = exit_base + RIGHT * 1.5 + UP * 0.2
        out_violet = exit_base + RIGHT * 1.5 + DOWN * 0.5
        
        beam_red = Line(entry_point, out_red, color="#FF0000", stroke_width=2)
        beam_violet = Line(entry_point, out_violet, color="#8B00FF", stroke_width=2)

        self.play(FadeIn(prism), Create(beam_in))
        self.play(Create(beam_red), Create(beam_violet))
        self.wait(3)
