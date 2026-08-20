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
        lecture_lines = [
            "Integration and differentiation are inverse processes.",
            "Like addition and subtraction for functions.",
            "An accumulator recovers the original function.",
            "Differentiation undoes what integration creates.",
            "They are perfectly connected foundations."
        ]
        self.setup_layout("The Fundamental Theorem: Connecting the Dots", lecture_lines)
        
        # Define visual elements
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve_f = axes.plot(lambda x: x**2/3, color="#00FF00") # Integral = Green
        curve_df = axes.plot(lambda x: 2*x/3, color="#FF0000") # Derivative = Red
        
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        puzzle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.place_at_grid(calc_icon, "A3", scale_factor=0.3)
        self.play(FadeIn(calc_icon))
        self.place_in_area(axes, "B3", "E6", scale_factor=0.5)
        self.play(Create(axes), Create(curve_df), Create(curve_f))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        dot = Dot(color=YELLOW).move_to(curve_f.get_start())
        self.play(MoveAlongPath(dot, curve_f))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        accumulator_label = Text("Accumulator", font_size=20)
        self.place_at_grid(accumulator_label, "A5", scale_factor=0.7)
        self.play(Write(accumulator_label))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(Indicate(curve_df), Indicate(curve_f))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        final_eq = MathTex(r"\\frac{d}{dx} \\int_a^x f(t) dt = f(x)", font_size=32)
        self.place_at_grid(final_eq, "D5", scale_factor=0.9)
        self.place_at_grid(puzzle_icon, "D6", scale_factor=0.3)
        self.play(Write(final_eq), FadeIn(puzzle_icon))
        
        self.wait(2)
