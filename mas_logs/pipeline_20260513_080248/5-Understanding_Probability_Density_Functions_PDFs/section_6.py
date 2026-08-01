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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Summary and Key Takeaway", 
            [
                "PDFs help us measure an infinite world of possibilities.", 
                "Always remember: in the PDF world, area is probability.", 
                "Now you can visualize continuous data with confidence."
            ]
        )
        
        # Define colors for highlighting
        highlight_color1 = "#FFFF00"  # Yellow
        highlight_color2 = "#00FFFF"  # Cyan
        highlight_color3 = "#00FF00"  # Green
        formula_color = "#FFD700"    # Gold

        # === Animation for Lecture Line 1 ===
        # Script: "PDFs help us measure an infinite world of possibilities."
        # Animation: Pip the Robot [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg] looks at a white Bell Curve (#FFFFFF).
        
        self.play(self.lecture[0].animate.set_color(highlight_color1), run_time=0.5)

        # Create Pip the Robot from Asset
        pip = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(pip, "B1", scale_factor=0.7)

        # Create Axis and Bell Curve
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False, "color": GREY},
            x_length=4,
            y_length=2.5
        )
        # Shift axes to center in the area C2 to F6 per review feedback
        self.place_in_area(axes, "C2", "F6", scale_factor=0.8)
        
        bell_curve = axes.plot(lambda x: np.exp(-x**2), color=WHITE)
        
        self.play(
            FadeIn(pip),
            Create(axes),
            Create(bell_curve),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Script: "Always remember: in the PDF world, area is probability."
        # Animation: Shade a random slice; show the formula 'Integral from a to b f(x) dx' (#FFD700).

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color2),
            run_time=0.5
        )

        # Create shaded slice
        a, b = -0.5, 0.8
        shaded_area = axes.get_area(bell_curve, x_range=[a, b], color=highlight_color2, opacity=0.5)
        
        # Integral Formula Construction (No MathTex)
        integral_sign = Text("∫", font_size=32, color=formula_color)
        upper_limit = Text("b", font_size=18, color=formula_color)
        lower_limit = Text("a", font_size=18, color=formula_color)
        func_text = Text("f(x) dx", font_size=28, color=formula_color)
        
        upper_limit.next_to(integral_sign, UR, buff=0.02)
        lower_limit.next_to(integral_sign, DR, buff=0.02)
        func_text.next_to(integral_sign, RIGHT, buff=0.1)
        formula = VGroup(integral_sign, upper_limit, lower_limit, func_text)
        
        # Scale integral formula per review feedback
        self.place_at_grid(formula, "B5", scale_factor=0.8)

        self.play(
            FadeIn(shaded_area),
            FadeIn(formula),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Script: "Now you can visualize continuous data with confidence."
        # Animation: Scale up the final text: 'Area = Probability' (#FFFFFF) on a black background.

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color3),
            run_time=0.5
        )

        final_text = Text("Area = Probability", font_size=40, color=WHITE)
        # Reposition final text per review feedback to avoid obstruction
        self.place_in_area(final_text, "C2", "E5", scale_factor=0.9)

        self.play(
            FadeOut(pip),
            FadeOut(axes),
            FadeOut(bell_curve),
            FadeOut(shaded_area),
            FadeOut(formula),
            FadeIn(final_text),
            run_time=2
        )
        self.wait(3)
