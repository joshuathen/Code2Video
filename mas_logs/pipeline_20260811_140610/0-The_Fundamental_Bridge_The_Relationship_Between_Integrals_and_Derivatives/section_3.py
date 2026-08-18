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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The Fundamental Theorem connects these two concepts.",
            "Integration and differentiation effectively cancel out.",
            "Like multiplication and division, they are opposites."
        ]
        self.setup_layout("The Fundamental Theorem: Connecting the Dots", lecture_lines)
        
        # Define math objects
        f_x = MathTex("f(x)", color=WHITE)
        integral_fx = MathTex("F(x) = \\int_{a}^{x} f(t) dt", color="#FFFF00")
        derivative_fx = MathTex("\\frac{d}{dx} F(x) = f(x)", color="#FF00FF")
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        abacus_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_in_area(calc_icon, "A3", "B5", scale_factor=0.5)
        self.place_in_area(f_x, "A3", "B5", scale_factor=1.2)
        self.play(FadeIn(calc_icon), Write(f_x))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.place_in_area(integral_fx, "C3", "D5", scale_factor=0.9)
        self.play(ReplacementTransform(f_x.copy(), integral_fx))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.place_in_area(derivative_fx, "E3", "F5", scale_factor=0.9)
        self.place_in_area(abacus_icon, "F4", "F6", scale_factor=0.5)
        self.play(Write(derivative_fx), FadeIn(abacus_icon))
        self.wait(2)
