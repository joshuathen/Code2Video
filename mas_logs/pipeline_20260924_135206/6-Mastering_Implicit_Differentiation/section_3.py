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
        self.setup_layout("The 3-Step Procedure", ["Differentiate both sides with respect to x.", "Collect all dy/dx terms together.", "Factor out dy/dx to solve."])
        
        equation = MathTex(r"x^3 + y^3 = 6xy").set_color(BLUE)
        self.place_at_grid(equation, 'A2', scale_factor=1.0)
        self.play(Write(equation))
        
        # Fixed: SVG files should be loaded with SVGMobject, not ImageMobject
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        paper = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paper.svg")
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(pencil, 'B5', scale_factor=0.3)
        self.play(FadeIn(pencil))
        step1 = MathTex(r"\frac{d}{dx}(x^3 + y^3) = \frac{d}{dx}(6xy)").set_color(YELLOW)
        self.place_in_area(step1, 'C2', 'C5', scale_factor=0.7)
        self.play(ReplacementTransform(equation.copy(), step1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(paper, 'B6', scale_factor=0.3)
        self.play(FadeIn(paper))
        step2 = MathTex(r"3x^2 + 3y^2 \frac{dy}{dx} = 6y + 6x \frac{dy}{dx}").set_color(GREEN)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.place_at_grid(calculator, 'B4', scale_factor=0.3)
        self.play(FadeIn(calculator))
        step3 = MathTex(r"\frac{dy}{dx} = \frac{6y - 3x^2}{3y^2 - 6x}").set_color(RED)
        
        group_steps2_3 = VGroup(step2, step3).arrange(DOWN, buff=0.5)
        self.place_in_area(group_steps2_3, 'D2', 'F5', scale_factor=0.75)
        self.play(ReplacementTransform(step1.copy(), step2), ReplacementTransform(step2.copy(), step3))
        self.wait(2)
