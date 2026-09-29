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
            "The board state represents sixty-four bits of data.",
            "Flipping one coin encodes the target square's index.",
            "The XOR sum now points to the guesser's square.",
            "Information theory allows us to solve the puzzle.",
            "The prisoner effectively communicates via the board's parity."
        ]
        self.setup_layout("The Solution: Information Theory in Action", lecture_lines)
        
        # Elements
        data_stream = VGroup(*[Text(str(i % 2), font_size=18, color="#EE82EE") for i in range(32)])
        data_stream.arrange_in_grid(rows=4, cols=8, buff=0.2)
        
        # Asset Loading
        board_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/board.svg")
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        compression_shape = Circle(radius=0.5, color="#FFA07A", fill_opacity=0.3)
        reduced_stream = VGroup(*[Text("X", font_size=20, color="#BFFF00") for _ in range(6)])
        reduced_stream.arrange(RIGHT, buff=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#EE82EE"))
        self.place_in_area(data_stream, 'A3', 'C6', scale_factor=0.8)
        self.place_at_grid(board_icon, 'A2', scale_factor=0.5)
        self.play(FadeIn(data_stream), FadeIn(board_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFA07A"))
        self.place_at_grid(compression_shape, 'D3', scale_factor=0.9)
        self.play(Create(compression_shape))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#BFFF00"))
        self.place_in_area(reduced_stream, 'E4', 'F6', scale_factor=0.7)
        self.place_at_grid(coin_icon, 'F3', scale_factor=0.5)
        self.play(FadeIn(reduced_stream), FadeIn(coin_icon))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
