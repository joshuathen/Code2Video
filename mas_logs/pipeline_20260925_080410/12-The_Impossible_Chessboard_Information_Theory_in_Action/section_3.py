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
            "We assign a unique binary index to each square.",
            "Alice treats the board as a 64-bit string.",
            "She calculates the parity of all heads-up coins.",
            "Flipping one coin changes the total parity.",
            "This forces the board to reveal the secret."
        ]
        self.setup_layout("Mapping the Board to the Information Space", lecture_lines)
        
        # Elements
        board = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chessboard.svg")
        
        # === Animation for Lecture Line 1 ===
        # Use place_in_area per VideoCritic guidance to resolve balance and scale
        self.place_in_area(board, "B2", "E5", scale_factor=0.9)
        self.play(FadeIn(board))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        bits = VGroup(*[Text(str(i % 2), font_size=12, color="#00FFFF") for i in range(16)])
        # Arrange bits over the board area
        self.place_in_area(bits, "B2", "E5", scale_factor=1.0)
        self.play(FadeIn(bits))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        key_indicator = Rectangle(width=0.5, height=0.5, color="#FF00FF")
        self.place_at_grid(key_indicator, "C3", scale_factor=1.0)
        self.play(Create(key_indicator))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Demonstrate interaction: shift bit color
        self.play(bits[0].animate.set_color(YELLOW))
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.wait(1)
