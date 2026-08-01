from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(f"• {line}", font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset to the right side of the screen
                x = 1.5 + j * 0.9
                y = 2.2 - i * 0.9
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Define layout and lecture points
        title = "Implicit Differentiation: Untangling Variables"
        bullets = [
            "Identify dependent variables",
            "Apply d/dx to both sides",
            "Use Chain Rule for y-terms",
            "Isolate dy/dx algebraically",
            "Factor out the derivative"
        ]
        self.setup_layout(title, bullets)

        # Step 1: Initial Equation 
        # Using Text instead of MathTex to avoid 'latex' FileNotFoundError
        eqn = Text("x² + y² = 25", color=BLUE, font_size=32)
        self.place_at_grid(eqn, "A2", scale_factor=0.9)
        self.play(Write(eqn))
        self.wait(1)

        # Step 2: Differentiating
        diff_op = Text("d/dx(x² + y²) = d/dx(25)", color=YELLOW, font_size=28)
        self.place_at_grid(diff_op, "B2", scale_factor=0.8)
        self.play(FadeIn(diff_op, shift=DOWN))
        self.wait(1)

        # Step 3: Result of differentiation (Implicit Step)
        result_step = Text("2x + 2y · dy/dx = 0", color=WHITE, font_size=32)
        self.place_at_grid(result_step, "C2", scale_factor=0.9)
        self.play(TransformMatchingShapes(diff_op.copy(), result_step))
        self.wait(1)

        # Step 4: Isolate dy/dx
        isolate_step = Text("2y · dy/dx = -2x", color=WHITE, font_size=32)
        self.place_at_grid(isolate_step, "D2", scale_factor=0.9)
        self.play(Write(isolate_step))
        self.wait(1)

        # Step 5: Final Solution
        final_sol = Text("dy/dx = -x/y", color=GREEN, font_size=36)
        self.place_at_grid(final_sol, "E2", scale_factor=1.0)
        self.play(
            Indicate(final_sol),
            FadeIn(final_sol, shift=UP)
        )
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(eqn, diff_op, result_step, isolate_step, final_sol)))
        self.wait(1)