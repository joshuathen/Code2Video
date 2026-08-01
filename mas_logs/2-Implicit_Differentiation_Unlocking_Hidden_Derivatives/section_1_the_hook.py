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

class Section1TheHookScene(TeachingScene):
    def construct(self):
        # 1. Setup layout with Section Title and Lecture Lines
        self.setup_layout(
            "Introduction: Explicit vs. Implicit Functions",
            [
                "Meet explicit functions like y = x^2.",
                "But some curves, like circles, tangle x and y.",
                "How do we find the slope of these hidden functions?"
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Lecture line 1: Highlighting in white as it matches the first element
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Explicit equation: y = x^2
        # Positioned at the top of the animation area
        # FIXED: Replaced MathTex with Text because LaTeX was not found in the environment
        explicit_eq = Text("y = x^2", color="#FFFFFF")
        self.place_in_area(explicit_eq, "A2", "A5", scale_factor=0.8)
        self.play(Write(explicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture line 2 to green
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Transform the explicit text into the implicit equation x^2 + y^2 = 25
        # FIXED: Replaced MathTex with Text because LaTeX was not found in the environment
        implicit_eq = Text("x^2 + y^2 = 25", color="#00FF00")
        self.place_in_area(implicit_eq, "A2", "A5", scale_factor=0.8)
        self.play(Transform(explicit_eq, implicit_eq))
        
        # Draw green circle (radius 2.5, as specified in animation plan)
        circle = Circle(radius=2.5, color="#00FF00")
        # Center in the 6x6 grid area (A1-F6)
        # Note: Center of A1-F6 is at x=3.0, y=-0.3. Radius 2.5 fits vertically (-2.8 to 2.2).
        self.place_in_area(circle, "A1", "F6") 
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture line 3 to yellow to match the slope indicator
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Plot yellow dot at point (1.5, 2) on the circle
        # Coordinates (1.5, 2) satisfy x^2 + y^2 = 2.5^2 = 6.25
        center_pos = circle.get_center()
        dot_pos = center_pos + np.array([1.5, 2.0, 0])
        dot = Dot(point=dot_pos, color="#FFFF00")
        
        # Tangent line at (1.5, 2). Slope is -x/y = -1.5/2.0 = -0.75
        # Perpendicular vector to radius (1.5, 2) is (2, -1.5)
        tangent_vec = np.array([2.0, -1.5, 0])
        tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
        tangent_line = Line(
            dot_pos - tangent_vec * 1.0, 
            dot_pos + tangent_vec * 1.0, 
            color="#FFFF00"
        )
        
        self.play(
            FadeIn(dot),
            Create(tangent_line)
        )
        self.wait(1)
        
        # Flash the equation in orange to emphasize the word 'implicit'
        self.play(
            explicit_eq.animate.set_color("#FF8C00"),
            Flash(explicit_eq, color="#FF8C00", line_length=0.3)
        )
        self.wait(2)
