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
        # Setup Layout
        title = "Defining the Central Limit Theorem (CLT)"
        lines = [
            'This is the formula for the Central Limit Theorem.', 
            'Larger sample sizes make the bell curve narrower.', 
            "The symbol Mu identifies the population's true average.", 
            'Usually, we need thirty or more samples for accuracy.', 
            'This golden rule ensures a stable, normal distribution.'
        ]
        self.setup_layout(title, lines)
        
        # Colors
        clt_formula_color = "#FFD700"  # Golden
        curve_color = "#00BFFF"        # Deep Sky Blue
        mu_color = "#FF4500"           # Orange-Red
        condition_color = "#00FF00"    # Green
        final_color = "#FFFF00"        # Yellow
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(clt_formula_color))
        
        # Formula: x_bar ~ N(mu, sigma^2/n)
        # Using Text to avoid LaTeX dependency
        formula = VGroup(
            Text("x̄"),     # 0
            Text("~"),     # 1
            Text("N("),    # 2
            Text("μ"),     # 3
            Text(","),     # 4
            Text("σ²/"),   # 5
            Text("n"),     # 6
            Text(")")      # 7
        ).arrange(RIGHT, buff=0.1).set_color(clt_formula_color)
        
        # Resolve Issue 51: scale_factor=0.9, area A2-B5
        self.place_in_area(formula, "A2", "B5", scale_factor=0.9)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(curve_color))
        
        # Bell curve logic using Axes
        n_tracker = ValueTracker(1) 
        axes = Axes(
            x_range=[-3, 3], y_range=[0, 1.2], 
            axis_config={"include_tip": False, "stroke_width": 1},
            tips=False
        )
        
        def get_curve():
            n = n_tracker.get_value()
            # Probability density function approximation for narrowing
            return axes.plot(lambda x: np.exp(-(n * (x**2))), color=curve_color)

        curve = get_curve()
        curve_group = VGroup(axes, curve)
        
        # Resolve Issue 53: scale_factor=0.7, area C2-E5
        self.place_in_area(curve_group, "C2", "E5", scale_factor=0.7)
        
        self.play(Create(axes), Create(curve))
        
        # Updater for narrowing curve
        curve.add_updater(lambda m: m.become(get_curve()))
        
        # Scale 'n' in formula and narrow the curve
        n_part = formula[6]
        self.play(
            n_part.animate.scale(1.5).set_color(WHITE),
            n_tracker.animate.set_value(5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(mu_color))
        
        # Highlight Mu in formula
        mu_part = formula[3]
        mu_highlight_box = SurroundingRectangle(mu_part, color=mu_color, buff=0.05)
        
        # Arrow from mu symbol to the center of the distribution
        arrow = Arrow(
            start=mu_part.get_bottom() + DOWN * 0.1, 
            end=axes.c2p(0, 0.2), 
            color=mu_color, 
            buff=0.1,
            stroke_width=3
        )
        
        self.play(Create(mu_highlight_box))
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(condition_color))
        
        # n >= 30 condition using Text
        n_cond = Text("n ≥ 30", color=condition_color)
        
        # Resolve Issue 52: scale_factor=0.8, grid pos F6
        self.place_at_grid(n_cond, "F6", scale_factor=0.8)
        
        # Glowing border around the curve area
        border = SurroundingRectangle(curve_group, color=WHITE, buff=0.1, stroke_width=1)
        border_glow = border.copy().set_stroke(WHITE, width=4).set_opacity(0.3)
        
        self.play(FadeIn(n_cond))
        # Pulsing effect while border surrounds curve
        self.play(
            n_cond.animate.scale(1.2), 
            Create(border), 
            FadeIn(border_glow), 
            run_time=1, 
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(final_color))
        self.wait(2)
