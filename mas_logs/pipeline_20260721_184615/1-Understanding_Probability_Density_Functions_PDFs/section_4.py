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
        title = "The Golden Rule: Probability = Area"
        lines = [
            "How do we find actual probability from density?",
            "We calculate the area under the curve.",
            "Choose a range between two specific time points.",
            "The shaded region represents the total likelihood.",
            "More area means a higher chance of occurrence."
        ]
        self.setup_layout(title, lines)

        # Helper for Bell Curve
        def bell_curve(x):
            return 2 * np.exp(-0.5 * (x)**2)

        # Define Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 2.5, 1],
            x_length=4.5,
            y_length=3,
            axis_config={"color": "#FFFFFF", "include_tip": False},
        )
        # L002: Start visuals at Column 2 or further right
        self.place_in_area(axes, "C2", "F6")

        # Define Curve - L008: Use Hex colors
        curve = axes.plot(bell_curve, color="#58C4DD")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(Create(axes), Create(curve))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Initial area
        a_val = -1.0
        # L027: Use updaters and ValueTracker for efficiency
        b_tracker = ValueTracker(1.0)
        
        # Area mobject with updater
        # L011: Avoid expensive objects in always_redraw. Polygon (from get_area) is okay.
        area = always_redraw(lambda: axes.get_area(
            curve, 
            x_range=[a_val, b_tracker.get_value()], 
            color="#90EE90", 
            opacity=0.4
        ))
        
        self.play(FadeIn(area))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Markers for a and b - L002: Scale labels
        label_a = MathTex("a", color="#FFFFFF", font_size=24)
        label_b = MathTex("b", color="#FFFFFF", font_size=24)
        
        def update_label_a(m):
            m.move_to(axes.c2p(a_val, -0.3))
        
        def update_label_b(m):
            m.move_to(axes.c2p(b_tracker.get_value(), -0.3))
            
        label_a.add_updater(update_label_a)
        label_b.add_updater(update_label_b)
        
        dot_a = Dot(axes.c2p(a_val, 0), color="#FFFFFF")
        dot_b = Dot(color="#FFFFFF")
        dot_b.add_updater(lambda m: m.move_to(axes.c2p(b_tracker.get_value(), 0)))

        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_a), Write(label_b))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        
        # Issue 30: Place sigma at A1
        sigma = MathTex(r"\sum", color="#FFFFFF")
        self.place_at_grid(sigma, "A1", scale_factor=0.8)
        
        # Issue 29: Place integral_formula in area A2-B5
        integral_formula = MathTex(
            r"P(a \le X \le b) = \int_{a}^{b} f(x) dx", 
            color="#FFFFFF"
        )
        self.place_in_area(integral_formula, "A2", "B5", scale_factor=0.7)
        
        self.play(Write(sigma))
        self.wait(1)
        self.play(ReplacementTransform(sigma, integral_formula))
        
        # Label with arrow
        # Issue 31: Place prob_label at B6
        prob_label = Text("Probability = Area", font_size=20, color="#90EE90")
        self.place_at_grid(prob_label, "B6", scale_factor=0.7)
        
        arrow = Arrow(
            start=prob_label.get_bottom(), 
            end=axes.c2p(0, 0.5), 
            color="#90EE90", 
            buff=0.1
        )
        
        self.play(FadeIn(prob_label), GrowArrow(arrow))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # Animate b moving to the right
        # L024: prefix rate_functions
        self.play(
            b_tracker.animate.set_value(2.2),
            run_time=3,
            rate_func=rate_functions.linear
        )
        
        # L004: Final highlight using Indicate
        self.play(Indicate(area, color="#90EE90"))
        self.wait(3.0)
        
        # Final cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1.0)
