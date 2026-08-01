from manim import *

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
        # Initial Setup: Title and Lecture Lines
        title = "The Math: Mean and Standard Error"
        lines = [
            "The curve stays centered on the population mean.",
            "We calculate variation using the Standard Error formula.",
            "Small samples produce a wider, flatter bell curve.",
            "Larger samples provide much higher statistical certainty.",
            "Precision increases with the square root of n."
        ]
        self.setup_layout(title, lines)

        # Colors
        HIGHLIGHT_YELLOW = "#FFFF00"
        BLUE_C = "#0000FF"

        # === Animation for Lecture Line 1 ===
        # The curve stays centered on the population mean.
        # Fixed: Using Text to avoid LaTeX dependency
        self.lecture[0].set_color(HIGHLIGHT_YELLOW)
        formula_mean = Text("μ_x = μ", font_size=32, color=WHITE)
        self.place_in_area(formula_mean, 'A2', 'B5', scale_factor=1.1)
        
        self.play(Write(formula_mean))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We calculate variation using the Standard Error formula.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_YELLOW)
        
        se_math = VGroup(
            Text("σ_x = σ /", font_size=24),
            Text("√n", font_size=24)
        ).arrange(RIGHT, buff=0.1).set_color(WHITE)
        
        se_text = Text("Standard Error:", font_size=24, color=WHITE)
        formula_se_group = VGroup(se_text, se_math).arrange(DOWN, buff=0.2)
        
        self.place_in_area(formula_se_group, 'D2', 'E5', scale_factor=1.1)
        
        self.play(Write(formula_se_group))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Small samples produce a wider, flatter bell curve.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE_C)

        self.play(
            FadeOut(formula_mean),
            formula_se_group.animate.scale(0.6).move_to(self.grid["A3"] + RIGHT * 1.0)
        )

        # Create Axes for the curve
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False, "color": GREY},
            x_length=5,
            y_length=3
        )
        self.place_in_area(axes, 'C1', 'F6', scale_factor=0.8)
        
        n_tracker = ValueTracker(30)
        
        def get_bell_curve_func(n_val):
            # Safeguard n_val
            n_val = max(0.1, n_val)
            curr_sigma = 6.0 / (n_val**0.5)
            return axes.plot(
                lambda x: (1 / (curr_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / curr_sigma)**2),
                color=BLUE_C
            )

        curve = always_redraw(lambda: get_bell_curve_func(n_tracker.get_value()))
        
        n_label_prefix = Text("n = ", font_size=22).move_to(axes.get_top() + UP*0.3 + LEFT*0.3)
        # FIXED: Pass mob_class=Text to Integer to avoid default MathTex which requires 'latex'
        n_val_display = Integer(30, font_size=22, mob_class=Text).next_to(n_label_prefix, RIGHT, buff=0.1)
        n_val_display.add_updater(lambda m: m.set_value(int(n_tracker.get_value())))

        self.play(Create(axes), Create(curve), Write(n_label_prefix), Write(n_val_display))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Larger samples provide much higher statistical certainty.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE_C)
        
        self.play(n_tracker.animate.set_value(1000), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Precision increases with the square root of n.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_YELLOW)
        
        # se_math is formula_se_group[1], and Text("√n") is se_math[1]
        target_sqrt_n = formula_se_group[1][1]
        
        self.play(
            target_sqrt_n.animate.set_color(HIGHLIGHT_YELLOW),
            Flash(target_sqrt_n, color=HIGHLIGHT_YELLOW)
        )
        self.wait(3)
