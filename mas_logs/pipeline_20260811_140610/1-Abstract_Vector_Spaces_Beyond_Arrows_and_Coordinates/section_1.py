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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors are not just arrows in space.",
            "They are objects obeying specific algebraic rules.",
            "Think of functions as vectors, not arrows.",
            "We define spaces by rules, not shapes.",
            "Abstraction lets us solve wider classes of problems."
        ]
        self.setup_layout("The Bridge: From Geometry to Abstraction", lecture_lines)
        
        # Objects
        arrow = Arrow(start=ORIGIN, end=RIGHT*1.0 + UP*0.7, color=BLUE)
        func_curve = FunctionGraph(lambda x: x**2, x_range=[-1.2, 1.2], color=YELLOW)
        axioms = MathTex(r"u+v = v+u", r"c(u+v) = cu+cv", font_size=30)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(arrow, "B2", scale_factor=0.6)
        self.play(Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(axioms, "B5", scale_factor=0.7)
        self.play(Write(axioms))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeOut(arrow))
        self.place_in_area(func_curve, "C4", "E6", scale_factor=0.6)
        self.play(Create(func_curve))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        rule_box = SurroundingRectangle(axioms, color=RED, buff=0.2)
        self.play(Create(rule_box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)
