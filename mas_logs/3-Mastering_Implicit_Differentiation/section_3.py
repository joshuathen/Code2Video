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
        # Section Title and Lecture Lines
        lecture_lines = [
            "Step one: Differentiate both sides with respect to x.",
            "Step two: Group all dy/dx terms on one side.",
            "Move remaining terms to the opposite side.",
            "Step three: Factor out dy/dx from the group.",
            "Finally, divide to solve for the derivative."
        ]
        self.setup_layout("The Three-Step Recipe", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Stage 1: Display 'x² + y² = 25' and apply the operator 'd/dx' to both sides.
        self.lecture[0].set_color(YELLOW)
        
        # FIXED: Replaced MathTex with Text to avoid FileNotFoundError when LaTeX is not installed.
        eq_initial = Text("x² + y² = 25")
        self.place_in_area(eq_initial, "B2", "B5", scale_factor=1.2)
        
        self.play(Write(eq_initial))
        self.wait(1)
        
        eq_op = Text("d/dx(x² + y²) = d/dx(25)")
        self.place_in_area(eq_op, "B2", "B5", scale_factor=1.2)
        
        self.play(ReplacementTransform(eq_initial, eq_op))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Stage 2: Transform the terms to '2x + 2y * dy/dx = 0' with 'dy/dx' highlighted in Green (#009E73).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        eq_diff = MathTex("2x + 2y", "\\frac{dy}{dx}", "= 0")
        eq_diff[1].set_color("#009E73")
        self.place_in_area(eq_diff, "C2", "C5", scale_factor=1.2)
        
        self.play(ReplacementTransform(eq_op, eq_diff))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Stage 3: Move '2x' to the right side of the equals sign, changing to '-2x'.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        eq_move = MathTex("2y", "\\frac{dy}{dx}", "= -2x")
        eq_move[1].set_color("#009E73")
        self.place_in_area(eq_move, "D2", "D5", scale_factor=1.2)
        
        self.play(ReplacementTransform(eq_diff.copy(), eq_move))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Stage 4: Divide both sides by '2y' to isolate the 'dy/dx' term.
        # (Aligned with "Factor out dy/dx" as the logical precursor to division in general recipes)
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        eq_isolate = MathTex("\\frac{dy}{dx}", "=", "\\frac{-2x}{2y}")
        eq_isolate[0].set_color("#009E73")
        self.place_in_area(eq_isolate, "E2", "E5", scale_factor=1.2)
        
        self.play(ReplacementTransform(eq_move.copy(), eq_isolate))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Stage 5: Simplify to 'dy/dx = -x/y' and draw a Cyan (#00FFFF) box around the final answer.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        eq_final = MathTex("\\frac{dy}{dx}", "=", "-\\frac{x}{y}")
        eq_final[0].set_color("#009E73")
        self.place_in_area(eq_final, "F2", "F5", scale_factor=1.2)
        
        box = SurroundingRectangle(eq_final, color="#00FFFF", buff=0.1)
        
        self.play(ReplacementTransform(eq_isolate.copy(), eq_final))
        self.play(Create(box))
        self.wait(2)
        
        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1)
