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
        # Define title and lecture lines from storyboard
        title = "The Core Concept: Height vs. Area"
        lecture_lines = [
            "- Curve height represents density, not exact probability.",
            "- Density shows where data points cluster most thickly.",
            "- Probability is the area under the curve between points.",
            "- Sand depth shows density; volume shows the probability.",
            "- We measure ranges, never just a single exact point."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # "Curve height represents density, not exact probability."
        self.play(self.lecture[0].animate.set_color("#90EE90"))
        
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 1, 0.5],
            x_length=5,
            y_length=3.5,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        self.place_in_area(axes, 'A1', 'F6')
        
        def pdf_func(x):
            # A simple Gaussian-like curve
            return 0.8 * np.exp(-0.5 * (x - 3)**2 / 1.0**2)
            
        curve = axes.plot(pdf_func, color="#90EE90")
        curve_label = MathTex("f(x)", color="#90EE90").scale(0.8)
        # Position f(x) near the peak (x=3)
        peak_pos = axes.c2p(3, pdf_func(3))
        curve_label.next_to(peak_pos, UP, buff=0.1)

        self.play(Create(axes), Create(curve), Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Density shows where data points cluster most thickly."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000")
        )
        
        # Focus on a single point to show height = density
        x_val = 3.5
        point_on_curve = Dot(axes.c2p(x_val, pdf_func(x_val)), color="#FF0000")
        vertical_line = axes.get_vertical_line(axes.c2p(x_val, pdf_func(x_val)), color="#FF0000")
        
        # Height represents density, so P(X=x) is 0
        # Resolving Issue 30: place in area B5-B6
        height_label = MathTex(r"P(X = 3.5) = 0", color="#FF0000")
        self.place_in_area(height_label, 'B5', 'B6', scale_factor=0.8)

        self.play(Create(point_on_curve), Create(vertical_line))
        self.play(Flash(point_on_curve, color="#FF0000", line_length=0.3), Write(height_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Probability is the area under the curve between points."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Define markers a and b on the x-axis
        a, b = 2.0, 4.0
        line_a = axes.get_vertical_line(axes.c2p(a, pdf_func(a)), color=WHITE)
        line_b = axes.get_vertical_line(axes.c2p(b, pdf_func(b)), color=WHITE)
        label_a = MathTex("a", color=WHITE).scale(0.8).next_to(axes.c2p(a, 0), DOWN)
        label_b = MathTex("b", color=WHITE).scale(0.8).next_to(axes.c2p(b, 0), DOWN)

        self.play(
            FadeOut(point_on_curve), FadeOut(vertical_line), FadeOut(height_label),
            Create(line_a), Create(line_b), Write(label_a), Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Sand depth shows density; volume shows the probability."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFFE0")
        )
        
        # Shade the area between a and b
        area = axes.get_area(curve, x_range=[a, b], color="#FFFFE0", opacity=0.5)
        
        # Analogy label: volume represents probability
        # Resolving Issue 28: place in area A4-A6
        sand_label = Text("Volume = Probability", font_size=20, color="#FFFFE0")
        self.place_in_area(sand_label, 'A4', 'A6', scale_factor=0.7)

        self.play(FadeIn(area), Write(sand_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We measure ranges, never just a single exact point."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )
        
        # Final label: P(a <= X <= b) = Area
        # Resolving Issue 29: place in area F4-F6
        prob_label = MathTex(r"P(a \le X \le b) = \text{Area}", color=WHITE)
        self.place_in_area(prob_label, 'F4', 'F6', scale_factor=0.7)
        
        # Highlight the interval on the x-axis
        range_line = Line(axes.c2p(a, 0), axes.c2p(b, 0), color=WHITE, stroke_width=6)

        self.play(Write(prob_label), Create(range_line))
        self.wait(3)

        # Final state: reset lecture line colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
