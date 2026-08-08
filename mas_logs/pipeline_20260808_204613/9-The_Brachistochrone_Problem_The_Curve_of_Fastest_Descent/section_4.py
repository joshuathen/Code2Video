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
        self.setup_layout("The Mathematical Solution: The Cycloid", [
            "The solution path is a cycloid.",
            "A cycloid traces a point on a rolling circle.",
            "It is defined by parametric equations.",
            "X equals r times theta minus sine theta.",
            "Y equals r times one minus cosine theta."
        ])
        
        # Prepare objects
        eq = MathTex(r"x = r(\theta - \sin \theta)", r"\\", r"y = r(1 - \cos \theta)", font_size=32)
        
        # Asset Assets
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        ball_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Using Asset as requested
        self.place_at_grid(circle_icon, "E3", scale_factor=0.7)
        point_label = Text("Point", font_size=20, color=RED)
        self.place_at_grid(point_label, "E4", scale_factor=0.5)
        self.play(Create(circle_icon), Write(point_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        # Fix: Positioning of equation
        self.place_in_area(eq, "B2", "D4", scale_factor=0.9)
        self.play(Write(eq))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.play(Indicate(eq[0]))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.play(Indicate(eq[2]))
        
        # Additional required assets (placeholder for logic)
        self.place_at_grid(ball_icon, "A6", scale_factor=0.5)
        self.play(FadeIn(ball_icon))
        
        self.wait(2)
