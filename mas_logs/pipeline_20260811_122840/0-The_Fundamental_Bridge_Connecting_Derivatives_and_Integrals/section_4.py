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
        lecture_lines = [
            "Differentiation and Integration are inverse operations.",
            "They work like adding and subtracting numbers.",
            "One builds up, the other breaks down.",
            "F of x differentiated becomes f of x.",
            "F of x integrated returns to original form."
        ]
        self.setup_layout("The Fundamental Bridge", lecture_lines)
        
        # Define visual objects
        curve = FunctionGraph(lambda x: 0.1 * x**2 + 1, x_range=[-2, 2], color=BLUE)
        f_x = MathTex("f(x)", color=BLUE)
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color=WHITE)
        integral_sign = MathTex("\\int", color=YELLOW)
        f_capital = MathTex("F(x)", color=PURPLE)
        rect = Rectangle(width=0.5, height=1, color=RED, fill_opacity=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(curve))
        self.place_at_grid(speedometer, "B4", scale_factor=0.3)
        self.play(FadeIn(speedometer))
        self.place_at_grid(f_x, "B3", scale_factor=0.85)
        self.play(Write(f_x))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(Indicate(f_x))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        self.place_at_grid(rect, "C3", scale_factor=0.9)
        self.play(FadeIn(rect))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        self.place_at_grid(integral_sign, "C4", scale_factor=0.8)
        self.play(Write(integral_sign))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.place_at_grid(f_capital, "D4", scale_factor=0.8)
        self.play(ReplacementTransform(integral_sign.copy(), f_capital))
        self.wait(2)
