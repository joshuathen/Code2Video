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
            "ODE solutions are simple curves.",
            "PDE solutions are evolving surfaces or volumes.",
            "Wave equation defines guitar string shapes.",
            "Boundary constraints define the allowed vibrations.",
            "Membranes create complex shifting geometric patterns."
        ]
        self.setup_layout("Visualizing the Solution Space", lecture_lines)
        
        # Elements
        curve = FunctionGraph(lambda x: np.sin(x), x_range=[-2, 2], color=BLUE)
        
        # Assets
        guitar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg", color=WHITE)
        string_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color=WHITE)
        
        # We need to not show lecture text until animated
        for line in self.lecture:
            line.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(curve, 'C5', scale_factor=0.8)
        self.play(Create(curve))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.play(FadeIn(self.lecture[1]))
        # Use guitar asset here
        self.place_at_grid(guitar, 'B4', scale_factor=0.5)
        self.play(FadeIn(guitar))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.play(FadeIn(self.lecture[2]))
        self.place_at_grid(string_asset, 'D3', scale_factor=0.9)
        # Using #FF00FF for string contours as requested
        string_asset.set_color("#FF00FF")
        self.play(Create(string_asset))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        self.play(FadeIn(self.lecture[3]))
        dot1 = Dot(color=RED).move_to(string_asset.get_left())
        dot2 = Dot(color=RED).move_to(string_asset.get_right())
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.lecture[3].set_color(RED)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        self.play(FadeIn(self.lecture[4]))
        circle = Circle(radius=1.0, color=PURPLE)
        self.place_at_grid(circle, 'F4', scale_factor=0.7)
        self.play(Create(circle))
        self.lecture[4].set_color(PURPLE)
        
        self.wait(2)
