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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Pi is an irrational, transcendental number.",
            "Its decimals continue forever without repeating.",
            "It never settles into a simple pattern.",
            "This distinguishes pi from simple fractions.",
            "Infinite complexity is hidden within pi."
        ]
        self.setup_layout("Formalizing π: Irrationality and Beyond", lecture_lines)

        # Define objects
        pi_symbol = MathTex(r"\\pi", font_size=96, color="#FFFFFF")
        self.place_at_grid(pi_symbol, 'B4', scale_factor=0.7)
        
        decimal_stream = Text("3.14159265358979323846...", font_size=24, color="#FF00FF")
        self.place_in_area(decimal_stream, 'D4', 'D6', scale_factor=0.6)
        
        # Asset integration
        fraction_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        fraction_x = VGroup(
            MathTex(r"\\frac{22}{7}", font_size=64, color="#FFFFFF"),
            fraction_icon.set_color("#FF0000").scale(0.5)
        )
        self.place_at_grid(fraction_x, 'E3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(FadeIn(pi_symbol))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#FFFF00")
        self.play(Write(decimal_stream))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#FFFF00")
        self.play(Indicate(decimal_stream))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color("#FFFFFF")
        self.lecture[3].set_color("#FFFF00")
        self.play(FadeIn(fraction_x))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color("#FFFF00")
        glow = Rectangle(color="#FFFF00", fill_opacity=0.2).surround(pi_symbol)
        self.play(
            pi_symbol.animate.scale(1.5).move_to(self.grid['C3']),
            FadeIn(glow)
        )
        self.wait(2)
