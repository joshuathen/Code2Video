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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        # Section Snapshot: Replace velocities (v) with (c/n). The 'c' terms cancel out.
        self.setup_layout("The Final Reveal: Snell's Law", [
            "Replace speeds v with c over refractive index n.",
            "The constant speed of light c cancels out.",
            "Result: n1 sin(theta 1) equals n2 sin(theta 2)."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(BLUE_A))
        
        # Start with the equation derived in previous sections
        # Switched from MathTex to Text to avoid 'latex' executable requirement
        eq_base = Text("sin θ₁ / v₁ = sin θ₂ / v₂", color=WHITE, font_size=32)
        self.place_in_area(eq_base, "A2", "B5", scale_factor=1.1)
        self.play(Write(eq_base))
        self.wait(0.5)

        # Show substitutions: v = c/n
        sub1 = Text("v₁ = c / n₁", font_size=24, color=BLUE_A)
        sub2 = Text("v₂ = c / n₂", font_size=24, color=BLUE_A)
        self.place_at_grid
