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
        # Data initialization
        title_text = "The Power Rule: Pattern Recognition"
        lecture_lines = [
            "Finding every slope using limits takes too long.",
            "The Power Rule offers a magical shortcut.",
            "For x squared, the derivative is simply 2x.",
            "At x equals one, the slope is two.",
            "The formula predicts the curve's steepness perfectly."
        ]
        
        # 1. Setup Layout
        self.setup_layout(title_text, lecture_lines)

        # Pre-defined Colors
        CYAN = "#00FFFF"
        YELLOW = "#FFFF00"
        WHITE_CLR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in Cyan
        self.play(self.lecture[0].animate.set_color(CYAN))
        
        # Display the power rule formula in bright cyan.
        power_rule_formula = MathTex(r"\frac{d}{dx}[x^n] = nx^{n-1}", color=CYAN)
        self.place_in_area(power_rule_formula, 'E1', 'F3', scale_factor=0.7)
        self.play(Write(power_rule_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in White
        self.play(self.lecture[1].animate.set_color(WHITE_CLR))
        
        # Animate the 'Derivative Machine' asset in the center of the grid area.
        machine_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg"
        machine = SVGMobject(machine_asset_path).set_color(WHITE_CLR)
        self.place_in_area(machine, 'B3', 'D4', scale_factor=1.5)
        self.play(FadeIn(machine))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 in Cyan
        self.play(self.lecture[2].animate.set_color(CYAN))
        
        # Move 'x^3' into the machine and show '3x^2' emerging as output.
        input_val = MathTex("x^3", color=CYAN)
        output_val = MathTex("3x^2", color=CYAN)
        
        self.place_at_grid(input_val, 'B2', scale_factor=0.8)
        self.play(FadeIn(input_val))
        
        # Entrance into machine
        self.play(
            input_val.animate.move_to(machine.get_center()).scale(0.1),
            run_time=1,
            rate_func=rate_functions.ease_in_quad
        )
        self.remove(input_val)
        
        # Emergence from machine
        output_val.move_to(machine.get_center()).scale(0.1)
        target_output_pos = self.grid['D5']
        self.play(
            output_val.animate.move_to(target_output_pos).scale(10.0),
            run_time=1,
            rate_func=rate_functions.ease_out_quad
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4 in Yellow
        self.play(
            FadeOut(machine),
            FadeOut(output_val),
            FadeOut(power_rule_formula),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Show a parabola where the slope at x=1 is labeled '2'.
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 6, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_numbers": True, "color": BLUE}
        )
        parabola = axes.plot(lambda x: x**2, x_range=[0, 2.2], color=YELLOW)
        
        # Label f(x)=x^2 at A5
        func_label = MathTex("f(x) = x^2", color=YELLOW)
        
        graph_group = VGroup(axes, parabola)
        self.place_in_area(graph_group, 'A2', 'F6', scale_factor=0.85)
        self.place_at_grid(func_label, 'A5', scale_factor=0.6)
        
        # Geometric demonstration at x=1
        x1 = 1
        y1 = x1**2
        dot_1 = Dot(axes.c2p(x1, y1), color=RED)
        tangent_1 = Line(axes.c2p(0.5, 0), axes.c2p(1.5, 2), color=WHITE)
        slope_label_1 = MathTex("m = 2", color=WHITE).scale(0.7)
        slope_label_1.next_to(dot_1, UR, buff=0.1)
        
        self.play(Create(axes), Create(parabola), Write(func_label))
        self.play(FadeIn(dot_1), Create(tangent_1), Write(slope_label_1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5 in Yellow
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # Geometric demonstration at x=2
        x2 = 2
        y2 = x2**2
        dot_2 = Dot(axes.c2p(x2, y2), color=RED)
        tangent_2 = Line(axes.c2p(1.6, 2.4), axes.c2p(2.4, 5.6), color=WHITE)
        slope_label_2 = MathTex("m = 4", color=WHITE).scale(0.7)
        slope_label_2.next_to(dot_2, UL, buff=0.1)
        
        # Animation sequence
        self.play(FadeIn(dot_2), Create(tangent_2), Write(slope_label_2))
        
        # Flash labels to emphasize the pattern
        self.play(
            Indicate(slope_label_1, color=YELLOW),
            Indicate(slope_label_2, color=YELLOW),
            run_time=2
        )
        self.wait(2)
