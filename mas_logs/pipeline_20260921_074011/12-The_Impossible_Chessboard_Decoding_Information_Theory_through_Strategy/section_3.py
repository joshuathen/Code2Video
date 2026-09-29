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
            "Each of the sixty-four squares has a unique coordinate.",
            "Calculate the XOR sum of all Heads indices.",
            "This sum pinpoints the square needing a flip.",
            "Mapping coordinates encodes information onto the board.",
            "This strategy aligns parity with a target state."
        ]
        self.setup_layout("The Strategy: Mapping Information to Coordinates", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Load Chessboard Asset
        chessboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chessboard.svg")
        self.place_in_area(chessboard, 'A2', 'E5', scale_factor=0.75)
        self.play(FadeIn(chessboard), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Represent data points as light pink (#FFB6C1) circles on grid.
        dots = VGroup(*[Dot(color="#FFB6C1", radius=0.05) for _ in range(5)])
        self.place_at_grid(dots, 'B2', scale_factor=0.5)
        self.play(FadeIn(dots), self.lecture[1].animate.set_color("#FFB6C1"))

        # === Animation for Lecture Line 3 ===
        # Highlight the mapping transformation with bright cyan (#00FFFF) arrows.
        arrow = Arrow(start=self.grid['A1'], end=self.grid['E5'], color="#00FFFF", buff=0.1)
        self.play(GrowArrow(arrow), self.lecture[2].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 4 ===
        # Show coordinate movement with smooth light gray (#D3D3D3) paths.
        path = CurvedArrow(self.grid['A1'], self.grid['F6'], angle=-TAU/6, color="#D3D3D3")
        self.place_in_area(path, 'C4', 'F6', scale_factor=0.6)
        self.play(Create(path), self.lecture[3].animate.set_color("#D3D3D3"))

        # === Animation for Lecture Line 5 ===
        # Overlay final state in light gold (#FFD700) at coordinate using coin.svg.
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        self.place_in_area(coin, 'E5', 'E5', scale_factor=0.5)
        self.play(FadeIn(coin), self.lecture[4].animate.set_color("#FFD700"))
        
        self.wait(2)
