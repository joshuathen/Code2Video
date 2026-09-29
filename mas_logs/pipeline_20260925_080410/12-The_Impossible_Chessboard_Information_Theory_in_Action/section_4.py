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
        lecture_lines = [
            "Alice uses the board as a communication channel.",
            "Changing parity points exactly to the hidden key.",
            "Bob reads the board to recover the coordinate.",
            "They survive with absolute, mathematical certainty.",
            "The strategy transforms a guess into pure logic."
        ]
        self.setup_layout("The Solution: Communication through State Change", lecture_lines)
        
        # Load Assets
        prisoner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prisoner.svg")
        board = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/board.svg")
        
        # Positioning
        self.place_in_area(board, "C3", "F6", scale_factor=0.75)
        self.place_at_grid(prisoner, "B3", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(prisoner), FadeIn(board))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Visualizing the state change as signal transmission in #FFFF00.
        signal = Circle(radius=0.3, color="#FFFF00").move_to(board.get_center())
        self.play(Create(signal), board.animate.set_color("#FFFF00"))
        self.play(FadeOut(signal))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.play(board.animate.set_color("#00FF00"))
        self.wait(1)
