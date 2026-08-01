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

class Section6Scene(TeachingScene):
    def construct(self):
        title = "Conclusion: The Definition of Consistency"
        lines = [
            "Zero factorial equaling one ensures mathematical consistency.",
            "It's the key piece that fits the puzzle perfectly.",
            "Now the mystery of zero factorial is finally solved."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        # Convert LaTeX strings to plain text/Unicode equivalents
        eqs = VGroup(
            Text("n! = n × (n-1)!", font_size=24),
            Text("C(n,k) = n! / (k!(n-k)!)", font_size=24),
            Text("P(n,k) = n! / (n-k)!", font_size=24),
            Text("e^x = Σ x^n / n!", font_size=24),
            Text("(a+b)^n = Σ C(n,k) a^k b^(n-k)", font_size=24),
            Text("Γ(n) = (n-1)!", font_size=24),
            Text("∫ x^m (1-x)^n dx", font_size=24),
            Text("S_n = Σ i", font_size=24)
        )
        
        # Position them around a central gap at C3-C4
        positions = ["A2", "A5", "B1", "B6", "D1", "D6", "E2", "E5"]
        for eq, pos in zip(eqs, positions):
            self.place_at_grid(eq, pos)
            
        # Draw a faint border or gap indicator
        gap_rect = DashedVMobject(RoundedRectangle(width=2.0, height=1.0, stroke_opacity=0.3))
        self.place_in_area(gap_rect, "C3", "C4")
        
        self.play(FadeIn(eqs), Create(gap_rect))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Glowing puzzle piece labeled '0! = 1' (Color: #00FF00)
        puzzle_piece = VGroup(
            RoundedRectangle(width=2.0, height=1.0, fill_opacity=0.2, fill_color="#00FF00", stroke_color="#00FF00"),
            Text("0! = 1", color="#00FF00", font_size=36)
        )
        
        # Start at a corner
        self.place_at_grid(puzzle_piece, "F6", scale_factor=0.8)
        
        # Animation: move into the gap
        target_center = (self.grid["C3"] + self.grid["C4"]) / 2
        
        self.play(
            puzzle_piece.animate.move_to(target_center).scale(1.25),
            gap_rect.animate.set_stroke(opacity=0),
            run_time=2
        )
        
        # Add glow
