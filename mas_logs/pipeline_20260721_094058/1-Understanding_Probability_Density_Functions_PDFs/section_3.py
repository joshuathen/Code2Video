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
        # Metadata
        title_text = "Defining the PDF Curve"
        lecture_lines = [
            "The curve's height represents probability density.",
            "Height is not the probability itself.",
            "Shaded area represents the probability of an interval.",
            "Total area under the curve equals one.",
            "Calculus tools like integrals calculate these areas."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Helpers
        def bell_curve(x, mu=0, sigma=1):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

        # Colors
        COLOR_CURVE = WHITE
        COLOR_ARROW = "#58C4DD"
        COLOR_SHADE = BLUE_D
        COLOR_HIGHLIGHT = YELLOW

        # Axes setup
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.4],
            axis_config={"include_tip": False, "color": GREY},
            x_length=5,
            y_length=4
        )
        # FIX ISSUE 26: Poor grid utilization
        self.place_in_area(axes, "B2", "F5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # Display f(x) in white #FFFFFF above a bell curve.
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        curve = axes.plot(lambda x: bell_curve(x, sigma=1.0), color=COLOR_CURVE)
        label_fx = MathTex("f(x)", color=WHITE)
        # FIX ISSUE 25: f(x) label position
        self.place_at_grid(label_fx, "A2", scale_factor=0.8)
        
        self.play(Create(axes), Create(curve), FadeIn(label_fx))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a blue #58C4DD arrow pointing to the peak of the curve.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        peak_point = axes.c2p(0, bell_curve(0, sigma=1.0))
        arrow = Arrow(
            start=self.grid["A3"], # Adjusted to point from A3 to keep it closer to A2 label
            end=peak_point,
            color=COLOR_ARROW,
            buff=0.1
        )
        
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in a shaded region between two points on the x-axis.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        a, b = -1, 1
        shaded_area = axes.get_area(curve, x_range=[a, b], color=COLOR_SHADE, opacity=0.5)
        line_a = axes.get_vertical_line(axes.c2p(a, bell_curve(a, sigma=1.0)), color=WHITE)
        line_b = axes.get_vertical_line(axes.c2p(b, bell_curve(b, sigma=1.0)), color=WHITE)
        label_a = MathTex("a", font_size=24).next_to(axes.c2p(a, 0), DOWN, buff=0.1)
        label_b = MathTex("b", font_size=24).next_to(axes.c2p(b, 0), DOWN, buff=0.1)

        self.play(
            FadeIn(shaded_area),
            Create(line_a), Create(line_b),
            FadeIn(label_a), FadeIn(label_b),
            FadeOut(arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Stretch the curve peak higher and show the shaded area expanding.
        # "Total area under the curve equals one."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        new_sigma = 0.6
        higher_curve = axes.plot(lambda x: bell_curve(x, sigma=new_sigma), color=COLOR_CURVE)
        # Expanding the shaded area to cover almost everything to imply "total area"
        expanded_shaded_area = axes.get_area(higher_curve, x_range=[-3, 3], color=COLOR_SHADE, opacity=0.3)
        
        self.play(
            Transform(curve, higher_curve),
            Transform(shaded_area, expanded_shaded_area),
            FadeOut(line_a), FadeOut(line_b),
            FadeOut(label_a), FadeOut(label_b),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show the integral symbol from 'a' to 'b' of f(x) dx.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        integral_formula = MathTex(r"\int_a^b f(x) dx", color=WHITE)
        # FIX ISSUE 24: integral_formula cut off
        self.place_in_area(integral_formula, "A4", "A5", scale_factor=0.7)
        
        # Bring back a smaller interval to show the integral concept
        a_f, b_f = -0.5, 1.2
        final_shaded = axes.get_area(higher_curve, x_range=[a_f, b_f], color=COLOR_SHADE, opacity=0.6)
        line_a_f = axes.get_vertical_line(axes.c2p(a_f, bell_curve(a_f, sigma=new_sigma)), color=WHITE)
        line_b_f = axes.get_vertical_line(axes.c2p(b_f, bell_curve(b_f, sigma=new_sigma)), color=WHITE)
        label_a_f = MathTex("a", font_size=24).next_to(axes.c2p(a_f, 0), DOWN, buff=0.1)
        label_b_f = MathTex("b", font_size=24).next_to(axes.c2p(b_f, 0), DOWN, buff=0.1)

        self.play(
            Write(integral_formula),
            Transform(shaded_area, final_shaded),
            Create(line_a_f), Create(line_b_f),
            FadeIn(label_a_f), FadeIn(label_b_f)
        )
        self.wait(2)
