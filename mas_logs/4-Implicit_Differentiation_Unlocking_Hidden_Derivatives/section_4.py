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
        # 1. Setup layout
        lecture_lines = [
            "Step one: Differentiate every term on both sides.",
            "Step two: Move all dy/dx terms to the left.",
            "Step three: Factor out dy/dx from the group.",
            "Finally, divide to isolate dy/dx on its own.",
            "This mechanical process works for any implicit equation."
        ]
        self.setup_layout("The 3-Step Execution Recipe", lecture_lines)

        # Colors
        active_color = "#FFFF00"  # Yellow
        highlight_color = "#55FF55" # Green

        # === Animation for Lecture Line 1 ===
        # Step one: Differentiate every term on both sides.
        self.play(self.lecture[0].animate.set_color(active_color))
        
        # Equation: x^2 + y^2 = 25
        eq1 = Text("x^2 + y^2 = 25", font_size=36)
        self.place_in_area(eq1, "A2", "A5")
        self.play(Write(eq1))
        self.wait(0.5)
        
        # Derivative: 2x + 2y(dy/dx) = 0
        # Passing components as separate strings for easier highlighting or transformation
        eq2 = Text("2x + 2y(dy/dx) = 0", font_size=36)
        self.place_in_area(eq2, "B2", "B5")
        
        # Animate derivation
        self.play(FadeIn(eq2, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step two: Move all dy/dx terms to the left.
        # (In practice, this means isolating terms with dy/dx on the left)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(active_color)
        )
        
        eq3 = VGroup(Text("2y", font_size=36), Text("dy/dx", font_size=36), Text("=", font_size=36), Text("-2x", font_size=36)).arrange(buff=0.1)
        self.place_in_area(eq3, "C2", "C5")
        
        # Transform eq2 to show the movement of 2x
        self.play(ReplacementTransform(eq2.copy(), eq3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step three: Factor out dy/dx from the group.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(active_color)
        )
        
        # Highlight the dy/dx term in eq3 to signify "factoring/focusing"
        # eq3[1] corresponds to "\\frac{dy}{dx}"
        self.play(eq3[1].animate.set_color(active_color))
        self.wait(0.5)
        self.play(eq3[1].animate.set_color(WHITE))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Finally, divide to isolate dy/dx on its own.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(active_color)
        )
        
        eq4 = MathTex("\\frac{dy}{dx}", "=", "\\frac{-2x}{2y}", font_size=36)
        self.place_in_area(eq4, "D2", "D5")
        
        self.play(ReplacementTransform(eq3.copy(), eq4))
        self.wait(0.5)
        
        # Final simplified version: dy/dx = -x/y
        eq5 = MathTex("\\frac{dy}{dx}", "=", "-\\frac{x}{y}", font_size=40)
        self.place_in_area(eq5, "E2", "E5")
        
        self.play(ReplacementTransform(eq4, eq5))
        
        # Box the final result in green #55FF55 as requested
        box = SurroundingRectangle(eq5, color=highlight_color, buff=0.15)
        self.play(Create(box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This mechanical process works for any implicit equation.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(active_color)
        )
        self.wait(2)
        
        # Final cleanup: reset lecture color
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
