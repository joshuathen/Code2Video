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
        # Data from storyboard
        title_text = "The Birth of the Derivative"
        lecture_lines = [
            "This single-point slope is the tangent line's slope.",
            "We call this instantaneous rate the derivative.",
            "Mathematically, it is the limit as h reaches zero.",
            "The formula captures change at a specific moment.",
            "This is the foundation of all calculus."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard and requirements
        TANGENT_COLOR = "#9B59B6"
        HIGHLIGHT_COLOR = "#F1C40F"
        FORMULA_COLOR = "#FFFFFF"

        # Setup Geometry (Right Side)
        axes = Axes(
            x_range=[-0.5, 4, 1],
            y_range=[-0.5, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY}
        )
        # Resolved Issue 30: Adjust axes placement
        self.place_in_area(axes, 'C1', 'F6', scale_factor=0.8)
        
        func = lambda x: 0.25 * (x**2)
        curve = axes.plot(func, x_range=[0, 4], color=BLUE)
        
        x_a = 1.2
        h_tracker = ValueTracker(1.5)
        
        dot_a = Dot(axes.c2p(x_a, func(x_a)), color=RED)
        label_a = MathTex("A", font_size=20).next_to(dot_a, DL, buff=0.1)
        
        # Dot B depends on h
        dot_b = Dot(color=WHITE)
        dot_b.add_updater(lambda m: m.move_to(axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value()))))
        
        label_b = MathTex("B", font_size=20)
        label_b.add_updater(lambda m: m.next_to(dot_b, UR, buff=0.1))
        
        # Secant line extended
        secant_line = Line(color=WHITE, stroke_width=2)
        def update_secant(m):
            p1 = dot_a.get_center()
            p2 = dot_b.get_center()
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            if dist < 0.005:
                m.set_stroke(opacity=0)
                return
            m.set_stroke(opacity=1)
            unit_vec = vec / dist
            # Extend the line visually
            m.put_start_and_end_on(p1 - 1.5 * unit_vec, p1 + 3.5 * unit_vec)
        
        secant_line.add_updater(update_secant)

        # === Animation for Lecture Line 1 ===
        # Show Point B overlapping Point A, making the interval 'h' vanish.
        self.lecture[0].set_color(YELLOW)
        self.add(axes, curve, dot_a, label_a, dot_b, label_b, secant_line)
        self.play(h_tracker.animate.set_value(0.001), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the Secant Line into a purple 'Tangent Line' (#9B59B6) touching only Point A.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Calculate slope for final tangent line: f'(x) = 0.5x at x=1.2 is 0.6
        slope_val = 0.5 * x_a
        tangent_line = Line(
            axes.c2p(x_a - 1.5, func(x_a) - 1.5 * slope_val),
            axes.c2p(x_a + 3.0, func(x_a) + 3.0 * slope_val),
            color=TANGENT_COLOR,
            stroke_width=4
        )
        
        # Stop updaters before transform to avoid glitches or errors
        secant_line.clear_updaters()
        dot_b.clear_updaters()
        label_b.clear_updaters()
        
        self.play(
            FadeOut(dot_b),
            FadeOut(label_b),
            ReplacementTransform(secant_line, tangent_line),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in the formal limit definition: f'(x) = lim(h->0) [f(x+h)-f(x)]/h in #FFFFFF.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Use raw strings and explicit prime notation
        formula_parts = MathTex(
            r"f^{\prime}(x) =", 
            r"\lim_{h \to 0}", 
            r"\frac{f(x+h) - f(x)}{h}",
            color=FORMULA_COLOR,
            font_size=32
        )
        # Resolved Issue 29: Adjust formula placement
        self.place_in_area(formula_parts, 'A1', 'B6', scale_factor=0.8)
        
        self.play(FadeIn(formula_parts))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the 'lim(h->0)' part of the formula in #F1C40F.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(formula_parts[1].animate.set_color(HIGHLIGHT_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pulse the Tangent Line to emphasize it represents the 'Instantaneous Rate'.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(
            tangent_line.animate.scale(1.2).set_stroke(width=6),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
