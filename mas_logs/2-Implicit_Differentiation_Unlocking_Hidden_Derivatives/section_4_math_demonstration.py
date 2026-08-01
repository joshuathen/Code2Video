from manim import *; MAGENTA, CYAN = PINK, TEAL

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

class Section4MathDemonstrationScene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("Worked Example: The Circle's Slope", [
            "Let's find the slope of x^2 + y^2 = 25.",
            "Differentiate: 2x + 2y times dy/dx = 0.",
            "Move 2x to the other side.",
            "Solve for dy/dx to get -x/y.",
            "At (3, 4), the tangent slope is -3/4."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        eq1 = Text("x^2 + y^2 = 25", color=WHITE)
        self.place_in_area(eq1, "A2", "A5", scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(MAGENTA)
        
        # d/dx operator appearing - changed MathTex to Text to bypass missing LaTeX dependency
        eq2_diff = Text("d/dx(x^2) + d/dx(y^2) = d/dx(25)")
        self.place_in_area(eq2_diff, "B1", "B6", scale_factor=0.9)
        self.play(FadeIn(eq2_diff, shift=DOWN))
        self.wait(1)

        # Result of differentiation: 2x + 2y * (dy/dx) = 0
        # Highlighting dy/dx in magenta
        eq2_res = MathTex("2x + 2y", r"\frac{dy}{dx}", "= 0")
        eq2_res[1].set_color(MAGENTA)
        self.place_in_area(eq2_res, "C2", "C5", scale_factor=1.0)
        self.play(TransformMatchingShapes(eq2_diff, eq2_res))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # Move 2x to the other side: 2y * (dy/dx) = -2x
        eq3 = MathTex("2y", r"\frac{dy}{dx}", "= -2x")
        eq3[1].set_color(MAGENTA)
        self.place_in_area(eq3, "D2", "D5", scale_factor=1.0)
        self.play(ReplacementTransform(eq2_res.copy(), eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        
        # Solve for dy/dx: dy/dx = -x/y
        eq4 = MathTex(r"\frac{dy}{dx}", "=", r"-\frac{x}{y}")
        eq4.set_color(GREEN)
        self.place_in_area(eq4, "E2", "E5", scale_factor=1.0)
        self.play(ReplacementTransform(eq3.copy(), eq4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(CYAN)
        
        # At (3, 4), the tangent slope is -3/4
        eval_point = MathTex(r"\frac{dy}{dx}\Big|_{(3,4)} = -\frac{3}{4}")
        eval_point.set_color(CYAN)
        self.place_in_area(eval_point, "F2", "F5", scale_factor=1.0)
        self.play(FadeIn(eval_point, shift=UP))
        self.wait(2)
