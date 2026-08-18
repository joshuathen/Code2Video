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
        lecture_lines = ["Compute area using anti-derivatives.", "Find F(b) minus F(a).", "Infinite summation is now unnecessary.", "Evaluation simplifies complex calculations.", "The result is total area."]
        self.setup_layout("The Power of Evaluation", lecture_lines)
        
        # Assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Formula
        formula = MathTex(r"\int_{a}^{b} f(x) \, dx = F(b) - F(a)", font_size=40)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        
        # Define Curve
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2*x**3 - 0.8*x**2 + 1.2*x + 0.5, x_range=[0.5, 3.5])
        area = axes.get_area(curve, x_range=[1, 3], color=BLUE, opacity=0.5)
        
        plot_group = VGroup(axes, curve, area)
        self.place_in_area(plot_group, 'D4', 'F6', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(calculator.move_to(self.grid['A4'])), Write(formula.set_color(WHITE)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(plot_group))
        
        # Highlight a and b
        a_label = MathTex("a", font_size=30, color=RED).next_to(axes.c2p(1,0), DOWN)
        b_label = MathTex("b", font_size=30, color=RED).next_to(axes.c2p(3,0), DOWN)
        self.play(Write(a_label), Write(b_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        cross = Cross(formula, color=RED)
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        self.play(FadeOut(cross), formula.animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(FadeIn(protractor.move_to(self.grid['C2'])), Indicate(area.set_fill(color="#00CED1")))
        self.wait(2)
