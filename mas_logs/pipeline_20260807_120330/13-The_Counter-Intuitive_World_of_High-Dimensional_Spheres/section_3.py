from manim import *
import numpy as np
import math

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
        # 1. Setup Layout
        title_text = "The Volume Paradox"
        lecture_lines = [
            "n-ball volume follows a very surprising pattern.",
            "Initially, volume increases as we add dimensions.",
            "Volume peaks around the fifth dimension.",
            "Then, the volume rapidly drops toward zero.",
            "In infinite dimensions, the ball's volume effectively vanishes."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Helper for volume of n-ball with R=1
        def v_n_func(n):
            if n <= 0: return 0
            # math.gamma(x) = (x-1)!
            return (math.pi**(n/2)) / math.gamma(n/2 + 1)

        # === Animation for Lecture Line 1 ===
        # Display volume formula V(n) at area B3. Formula color #00FFFF. R highlighted #FFFF00.
        self.lecture[0].set_color("#00FFFF")
        # Formula parts: V_n( (0), R (1), ) = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2} + 1)} (2), R (3), ^n (4)
        formula = MathTex(
            "V_n(", "R", ") = \\frac{\\pi^{n/2}}{\\Gamma(\\frac{n}{2} + 1)} ", "R", "^n",
            font_size=32, color="#00FFFF"
        )
        formula[1].set_color("#FFFF00") # First R
        formula[3].set_color("#FFFF00") # Second R
        
        # Issue 30: The 'formula' at C3 has a scale factor of 1.2... Fix: scale_factor 1.0
        # Storyboard says B3, so we use B3.
        self.place_at_grid(formula, "B3", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create graph axes in area C3 to E6. Plot points n=1 to 3 showing upward trend.
        self.lecture[1].set_color(WHITE)
        
        axes = Axes(
            x_range=[0, 22, 5],
            y_range=[0, 6, 1],
            x_length=4.5,
            y_length=3.2,
            axis_config={"include_tip": True, "font_size": 20},
        )
        self.place_in_area(axes, "C3", "E6")
        
        x_label = axes.get_x_axis_label("n", edge=RIGHT, direction=DOWN, buff=0.1)
        y_label = axes.get_y_axis_label("V_n", edge=UP, direction=LEFT, buff=0.1)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Plot points n=1 to 3
        points_1_3 = VGroup(*[
            Dot(axes.c2p(n, v_n_func(n)), color=WHITE, radius=0.06) for n in range(1, 4)
        ])
        curve_1_3 = axes.plot(lambda x: v_n_func(x), x_range=[1, 3], color=WHITE)
        
        self.play(Create(curve_1_3), FadeIn(points_1_3, lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Extend plot to n=5. Highlight peak point in #00FF00. Area C3 to E6.
        self.lecture[2].set_color("#00FF00")
        
        curve_3_5 = axes.plot(lambda x: v_n_func(x), x_range=[3, 5], color="#00FF00")
        peak_dot = Dot(axes.c2p(5, v_n_func(5)), color="#00FF00", radius=0.08)
        peak_label = Text("Peak (n≈5)", font_size=16, color="#00FF00")
        peak_label.next_to(peak_dot, UP, buff=0.1)
        
        self.play(Create(curve_3_5), FadeIn(peak_dot))
        self.play(Write(peak_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Plot n=10, 20. Curve drops sharply. Highlight tail in #FF0000. Area C3 to E6.
        self.lecture[3].set_color("#FF0000")
        
        curve_5_20 = axes.plot(lambda x: v_n_func(x), x_range=[5, 20], color="#FF0000")
        tail_dots = VGroup(*[
            Dot(axes.c2p(n, v_n_func(n)), color="#FF0000", radius=0.05) for n in [10, 15, 20]
        ])
        
        self.play(Create(curve_5_20), FadeIn(tail_dots))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Visual of 'Magic Balloon' [Asset: balloon.svg] at n=5 shrinking to a tiny dot at center. Area D4.
        self.lecture[4].set_color(WHITE)
        
        # Cleanup graph for balloon focus
        self.play(
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label), 
            FadeOut(curve_1_3), FadeOut(points_1_3),
            FadeOut(curve_3_5), FadeOut(peak_dot), FadeOut(peak_label),
            FadeOut(curve_5_20), FadeOut(tail_dots)
        )

        # Conclusion Text (Issue 29: place_in_area 'E2', 'E5')
        conclusion_text = Text(
            "In infinite dimensions, the ball's volume effectively vanishes.",
            font_size=20, color=WHITE
        )
        self.place_in_area(conclusion_text, "E2", "E5", scale_factor=0.8)

        # Balloon asset (Issue 21)
        # Load balloon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg]
        balloon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg")
        self.place_at_grid(balloon, "D4", scale_factor=1.2) # Issue 28: Adjusting scale for visual clarity
        
        self.play(FadeIn(balloon), Write(conclusion_text))
        self.wait(1)
        
        # Shrink balloon (n -> infinity)
        self.play(balloon.animate.scale(0.001), run_time=3)
        self.wait(2)
