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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Nature of Pi: Irrationality", [
            "Pi is an irrational number.",
            "Its digits never end or repeat.",
            "It cannot be a simple fraction."
        ])
        
        # === Animation for Lecture Line 1 ===
        # B002: Avoid cols 1-3, rows A/F. Use B4-B6, C4-C6. 
        # B035: Offset center.
        # B018: Explicitly label.
        pi_symbol = MathTex(r"\\pi", font_size=40)
        pi_digits = MathTex("3.14159...", color="#FF4500")
        pi_group = VGroup(pi_symbol, pi_digits).arrange(DOWN)
        
        self.place_at_grid(pi_group, 'B4', scale_factor=0.9)
        self.play(Write(pi_group))
        self.lecture[0].set_color("#FF4500")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        
        # B011: Vertical stacking.
        digits_flow = VGroup(*[Text(str(i % 10), font_size=20, color=WHITE) for i in range(15)])
        digits_flow.arrange(DOWN, buff=0.15)
        
        # B035/B037: Refine placement (A4-B6)
        self.place_in_area(digits_flow, 'A4', 'B6', scale_factor=0.75)
        
        # B038: Label variables if needed, here just digits flow
        irrational_label = Text("Irrational", color="#FFFFFF", font_size=24)
        self.place_at_grid(irrational_label, 'D4', scale_factor=0.8)

        self.play(FadeIn(digits_flow))
        self.play(digits_flow.animate.shift(DOWN * 1.5), run_time=2, rate_func=linear)
        self.play(Write(irrational_label))
        
        # Flash digits red
        self.play(pi_digits.animate.set_color("#FF4500"), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.play(FadeOut(pi_group), FadeOut(digits_flow), FadeOut(irrational_label))
