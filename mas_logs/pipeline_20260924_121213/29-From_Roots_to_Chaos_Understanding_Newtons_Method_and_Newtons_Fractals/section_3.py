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
        self.setup_layout("The Mystery of the Starting Point", 
                          ["Does starting point choice matter?", 
                           "Points gravitate toward specific roots.", 
                           "Three magnets pull a ball bearing."])
        
        # Colors per constraint
        C1 = "#FFFFFF" # White
        C2 = "#00FFFF" # Cyan
        C3 = "#FF0000" # Red
        
        # === Animation for Lecture Line 1 ===
        # Show multiple potential starting points
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg]
        balls = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color=C1) for _ in range(5)])
        self.place_in_area(balls, 'B3', 'B4', scale_factor=0.8)
        self.play(FadeIn(balls))
        self.lecture[0].set_color(C1)

        # === Animation for Lecture Line 2 ===
        # Display how different start points lead to different roots.
        arrows = VGroup(*[Arrow(start=ball.get_center(), end=ball.get_center() + DOWN * 1.5, color=C2) for ball in balls])
        self.play(Create(arrows))
        self.lecture[1].set_color(C2)

        # === Animation for Lecture Line 3 ===
        # Show a point being pulled towards a local minimum
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg]
        magnet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg", color=C3)
        self.place_at_grid(magnet, 'F3', scale_factor=0.9)
        self.play(FadeIn(magnet))
        self.lecture[2].set_color(C3)
        self.play(FadeOut(arrows), FadeOut(balls))
