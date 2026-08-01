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
        self.setup_layout("Geometric Intuition: The Magnifying Glass", [
            "Finding the slope of a curve is harder.",
            "Let's zoom into this point on the curve.",
            "As we zoom in, the curve looks straighter.",
            "Eventually, it looks like a simple straight line.",
            "This local slope is the derivative at that point."
        ])

        # === Animation for Lecture Line 1 ===
        # Parabola y=x^2
        axes = Axes(
            x_range=[-0.5, 2.5, 1],
            y_range=[-0.5, 4.5, 1],
            axis_config={"include_tip": True, "color": GREY_D},
            x_length=5,
            y_length=5
        )
        self.place_in_area(axes, "A1", "F6")
        
        parabola = axes.plot(lambda x: x**2, x_range=[-0.5, 2.1], color="#FC6255")
        parabola_label = MathTex("y = x^2", color="#FC6255")
        # Resolution for Issue 27: Position at A5
        self.place_at_grid(parabola_label, "A5", scale_factor=0.8)

        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Create(parabola), Write(parabola_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Points and secant line
        p1 = axes.c2p(1, 1)
        p2 = axes.c2p(2, 4)
        dot1 = Dot(p1, color=WHITE)
        dot2 = Dot(p2, color=WHITE)
        secant = Line(axes.c2p(0.5, 0.25), axes.c2p(2.2, 4.6), color=WHITE)
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(dot1), FadeIn(dot2), Create(secant))
        self.wait(1)

        # Resolution for Issue 19: Use Asset for magnifying glass
        mag_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        mag_glass.set_color(WHITE)
        mag_glass.scale(0.3)
        mag_glass.move_to(p1)
        
        self.play(FadeIn(mag_glass))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Zooming in (Step 1)
        zoomed_axes = Axes(
            x_range=[0.8, 1.2, 0.1],
            y_range=[0.6, 1.4, 0.2],
            axis_config={"include_tip": True, "color": GREY_D},
            x_length=5,
            y_length=5
        )
        self.place_in_area(zoomed_axes, "A1", "F6")
        zoomed_parabola = zoomed_axes.plot(lambda x: x**2, x_range=[0.8, 1.2], color="#FC6255")
        
        # Scaling up magnifying glass to act as a frame
        # We target a size that occupies the right-side demonstration area
        mag_glass_large = mag_glass.copy().scale(5).move_to(zoomed_axes.get_center())

        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Resolution for Issue 29: Ensure parabola_label is faded out
        self.play(
            ReplacementTransform(axes, zoomed_axes),
            ReplacementTransform(parabola, zoomed_parabola),
            ReplacementTransform(mag_glass, mag_glass_large),
            FadeOut(secant),
            FadeOut(dot2),
            FadeOut(parabola_label),
            dot1.animate.move_to(zoomed_axes.c2p(1, 1)),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Even more zoom (Step 2)
        final_axes = Axes(
            x_range=[0.95, 1.05, 0.025],
            y_range=[0.9, 1.1, 0.05],
            axis_config={"include_tip": True, "color": GREY_D},
            x_length=5,
            y_length=5
        )
        self.place_in_area(final_axes, "A1", "F6")
        final_parabola = final_axes.plot(lambda x: x**2, x_range=[0.95, 1.05], color="#FC6255")

        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(
            ReplacementTransform(zoomed_axes, final_axes),
            ReplacementTransform(zoomed_parabola, final_parabola),
            dot1.animate.move_to(final_axes.c2p(1, 1)),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Tangent line and Slope
        tangent = final_axes.plot(lambda x: 2*x - 1, x_range=[0.95, 1.05], color="#FFFF00")
        slope_label = MathTex("slope", "=", "2", color="#FFFF00")
        # Resolution for Issue 28: Position at D6
        self.place_at_grid(slope_label, "D6", scale_factor=0.8)

        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(Create(tangent), Write(slope_label))
        self.wait(2)
