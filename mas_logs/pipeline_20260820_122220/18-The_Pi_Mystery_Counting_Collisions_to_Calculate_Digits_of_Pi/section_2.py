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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Elastic Collisions and Momentum", 
                          ["Collisions conserve both momentum and energy.", 
                           "Elastic bounces keep total energy constant.", 
                           "Velocity vectors change during every impact."])
        
        # Mobjects
        eq = MathTex(r"p = mv", color=WHITE)
        ball1 = Circle(radius=0.5, color="#FF5733", fill_opacity=0.8)
        ball2 = Circle(radius=0.5, color="#33FF57", fill_opacity=0.8)
        v1 = Arrow(start=ORIGIN, end=RIGHT, color="#FF0000")
        v2 = Arrow(start=ORIGIN, end=LEFT, color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(eq, "A3", "B4", scale_factor=0.9)
        self.play(Write(eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.place_at_grid(ball1, "D3", scale_factor=0.8)
        self.place_at_grid(ball2, "D4", scale_factor=0.8)
        self.play(FadeIn(ball1), FadeIn(ball2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        v1.next_to(ball1, UP)
        v2.next_to(ball2, UP)
        self.play(Create(v1), Create(v2))
        
        # Animate velocity change
        self.play(
            v1.animate.rotate(PI),
            v2.animate.rotate(PI),
            ball1.animate.shift(RIGHT * 0.5),
            ball2.animate.shift(LEFT * 0.5)
        )
        self.wait(2)
