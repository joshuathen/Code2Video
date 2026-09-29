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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: Normalization", [
            "Result: I² = π, so I = sqrt(π).", 
            "Total probability area must equal one.", 
            "The factor 1/sqrt(2π) scales the curve."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Result: I² = π, so I = sqrt(π).
        eq1 = MathTex("I^2 = \\pi", color=WHITE)
        self.place_at_grid(eq1, "B2")
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)
        
        eq2 = MathTex("I = \\sqrt{\\pi}", color="#E74C3C")
        self.place_at_grid(eq2, "C2")
        self.play(ReplacementTransform(eq1.copy(), eq2), self.lecture[0].animate.set_color("#E74C3C"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Total probability area must equal one.
        prob_label = Text("Total Area = 1.0", font_size=24)
        self.place_at_grid(prob_label, "D3", scale_factor=0.9)
        self.play(Write(prob_label), self.lecture[1].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The factor 1/sqrt(2π) scales the curve.
        constant = MathTex("\\frac{1}{\\sqrt{2\\pi}}", color="#33FF57")
        self.place_at_grid(constant, "B6", scale_factor=0.7)
        
        # [Asset: NormalizationScalingGraphic] represented as a simple visual proxy
        rect = Rectangle(width=2, height=1.5, color="#33FF57")
        self.place_in_area(rect, "C3", "E5", scale_factor=0.8)
        
        self.play(FadeIn(constant), FadeIn(rect), self.lecture[2].animate.set_color("#33FF57"))
        self.wait(2)
