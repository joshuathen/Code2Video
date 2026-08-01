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
        title_text = "The Step-by-Step 'Untangling' Algorithm"
        lecture_lines = [
            "Step one: Differentiate both sides with respect to x.",
            "For x^2 + y^2 = 25, we get 2x + 2y(dy/dx) = 0.",
            "Step two: Collect all dy/dx terms on one side.",
            "Step three: Factor out dy/dx and solve algebraically.",
            "The result: dy/dx equals negative x divided by y."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        ORANGE_DYDX = "#FF8800"
        STEP_2_COLOR = "#00FF00"
        STEP_3_COLOR = "#FF00FF"
        RESULT_COLOR = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        eq1 = Text("x^2 + y^2 = 25", color=WHITE)
        self.place_in_area(eq1, "A1", "C6", scale_factor=1.5)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE_DYDX)
        # Replacing MathTex with Text because the environment lacks a LaTeX installation
        eq2 = Text("2x + 2y(dy/dx) = 0", color=WHITE)
        eq2[3].set_color(ORANGE_DYDX)
        self.place_in_area(eq2, "A1", "C6", scale_factor=1.5)
        
        self.play(TransformMatchingShapes(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(STEP_2_COLOR)
        eq3 = MathTex("2y", "\\frac{dy}{dx}", " = -2x", color=STEP_2_COLOR)
        eq3[1].set_color(ORANGE_DYDX)
        self.place_in_area(eq3, "C1", "D6", scale_factor=1.5)
        
        self.play(TransformMatchingShapes(eq2.copy(), eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(STEP_3_COLOR)
        eq4 = MathTex("\\frac{dy}{dx}", " = ", "\\frac{-2x}{2y}", color=STEP_3_COLOR)
        eq4[0].set_color(ORANGE_DYDX)
        self.place_in_area(eq4, "D1", "F6", scale_factor=1.5)
        
        self.play(TransformMatchingShapes(eq3.copy(), eq4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RESULT_COLOR)
        eq5 = MathTex("\\frac{dy}{dx}", " = ", "-\\frac{x}{y}", color=RESULT_COLOR)
        self.place_in_area(eq5, "D1", "F6", scale_factor=1.5)
        
        self.play(TransformMatchingShapes(eq4, eq5))
        self.wait(2)