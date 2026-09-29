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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Grand Finale: Deriving e^(πi) = -1", [
            "Set θ to π.",
            "cos(π) is -1 and sin(π) is 0.",
            "Plugging into Euler’s formula.",
            "The point hits (-1, 0).",
            "Resulting in e^(πi) = -1."
        ])
        
        # Initial Formula
        euler_formula = MathTex("e^{i\\theta} = \\cos\\theta + i\\sin\\theta", font_size=36)
        self.place_in_area(euler_formula, 'A3', 'B5', scale_factor=0.8)
        self.play(Write(euler_formula))

        # === Animation for Lecture Line 1 ===
        # Set θ to π.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        theta_pi = MathTex("e^{i\\pi} = \\cos\\pi + i\\sin\\pi", font_size=36)
        self.place_in_area(theta_pi, 'A3', 'B5', scale_factor=0.8)
        self.play(ReplacementTransform(euler_formula, theta_pi))

        # === Animation for Lecture Line 2 ===
        # cos(π) is -1 and sin(π) is 0.
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # === Animation for Lecture Line 3 ===
        # Plugging into Euler’s formula.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        plugged_formula = MathTex("e^{i\\pi} = -1 + i(0)", font_size=36)
        self.place_in_area(plugged_formula, 'C3', 'C5', scale_factor=0.75)
        self.play(ReplacementTransform(theta_pi, plugged_formula))
        
        # === Animation for Lecture Line 4 ===
        # The point hits (-1, 0).
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        axes = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": True}).scale(0.5)
        self.place_in_area(axes, 'D2', 'F4', scale_factor=0.9)
        point = Dot(axes.c2p(-1, 0), color=RED)
        
        # Asset integration placeholder (None.svg used as base for glow)
        glow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        glow.scale(0.2).move_to(point.get_center())
        
        label = MathTex("(-1, 0)", font_size=24).next_to(point, UP)
        self.play(Create(axes), FadeIn(point), FadeIn(glow), Write(label))
        
        # === Animation for Lecture Line 5 ===
        # Resulting in e^(πi) = -1.
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        final_result = MathTex("e^{i\\pi} = -1", font_size=48, color=YELLOW)
        self.place_at_grid(final_result, 'E5', scale_factor=0.85)
        self.play(Write(final_result), Indicate(final_result))
        self.wait(2)
