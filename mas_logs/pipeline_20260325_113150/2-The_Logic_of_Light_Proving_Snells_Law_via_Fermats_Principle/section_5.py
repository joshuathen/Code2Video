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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        title_text = "Calculus and the Optimization Step"
        lecture_lines = [
            "To find the minimum time, we set the derivative to zero.",
            "Differentiating with respect to x yields this ratio equality.",
            "These terms represent the geometry of the light's path.",
            "We substitute the ratios with sines of the angles.",
            "This simplifies to a relationship between sines and velocities."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for alignment
        c1 = "#FFFFFF"  # White
        c2 = "#FFFF00"  # Yellow
        c3 = "#FFA500"  # Orange
        c4 = "#FFC0CB"  # Pink
        c5 = "#90EE90"  # Light Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(c1)
        # Using Text instead of MathTex to avoid latex dependency error
        l1_eqn = Text("dT / dx = 0", color=c1)
        self.place_in_area(l1_eqn, "A2", "A5", scale_factor=1.2)
        self.play(Write(l1_eqn))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(c2)
        # Reconstructing equation with Text VGroup to allow targeted indexing without LaTeX
        l2_p1 = Text("x / (v₁ √(h₁² + x²))", color=c2, font_size=32)
        l2_minus = Text(" - ", color=c2, font_size=32)
        l2_p2 = Text("(L-x) / (v₂ √(h₂² + (L-x)²)) = 0", color=c2, font_size=32)
        l2_eqn = VGroup(l2_p1, l2_minus, l2_p2).arrange(RIGHT, buff=0.1)
        
        # ISSUE 41: Adjusted area from B1-B6 to B2-B6 and scale from 0.9 to 0.8
        self.place_in_area(l2_eqn, "B2", "B6", scale_factor=0.8)
        self.play(Write(l2_eqn))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(c3)
        # Diagram representing the geometry
        interface = Line(self.grid["E1"], self.grid["E6"], color=BLUE)
        normal = DashedLine(self.grid["C3"], self.grid["F3"], color=GRAY)
        
        # Ray 1: from top-left (C2) to P (E3)
        p_point = self.grid["E3"]
        ray1 = Line(self.grid["C2"], p_point, color=YELLOW)
        # Ray 2: from P (E3) to bottom-right (F5)
        ray2 = Line(p_point, self.grid["F5"], color=YELLOW)
        
        # Labels for the diagram using Text for theta symbols
        theta1 = Text("θ₁", color=c3, font_size=24)
        self.place_at_grid(theta1, "D3", scale_factor=0.8)
        theta1.shift(LEFT*0.3 + UP*0.2)
        
        theta2 = Text("θ₂", color=c3, font_size=24)
        self.place_at_grid(theta2, "F3", scale_factor=0.8)
        theta2.shift(RIGHT*0.3 + DOWN*0.2)

        diagram = VGroup(interface, normal, ray1, ray2, theta1, theta2)
        self.play(Create(diagram))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(c4)
        # Highlight first term (index 0 of our VGroup)
        box1 = SurroundingRectangle(l2_eqn[0], color=c4, buff=0.1)
        sin1_term = Text("sin(θ₁) / v₁", color=c4, font_size=36)
        # ISSUE 42: Adjusted position to B2 and scale to 0.9
        self.place_at_grid(sin1_term, "B2", scale_factor=0.9)
        
        self.play(Create(box1))
        self.play(Write(sin1_term))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(c5)
        # Highlight second term (index 2 of our VGroup)
        box2 = SurroundingRectangle(l2_eqn[2], color=c5, buff=0.1)
        sin2_term = Text("sin(θ₂) / v₂", color=c5, font_size=36)
        # ISSUE 42: Adjusted position to B5 and scale to 0.9
        self.place_at_grid(sin2_term, "B5", scale_factor=0.9)
        
        self.play(Create(box2))
        self.play(Write(sin2_term))
        self.wait(1)

        # Final conclusion
        final_eqn = Text("sin(θ₁) / v₁ = sin(θ₂) / v₂", color=c5)
        # ISSUE 43: Adjusted area to B2-B5 and scale to 1.0
        self.place_in_area(final_eqn, "B2", "B5", scale_factor=1.0)
        
        self.play(
            FadeOut(l1_eqn),
            FadeOut(l2_eqn),
            FadeOut(box1),
            FadeOut(box2),
            Transform(VGroup(sin1_term, sin2_term), final_eqn)
        )
        self.wait(2)
