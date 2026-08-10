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
        self.setup_layout("The Problem: Does the Parent Distribution Matter?", [
            "Does the parent distribution shape matter?", 
            "Let's sample from non-normal populations.", 
            "Do sample means follow original patterns?"
        ])
        
        # Load Assets
        hist_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        squirrel_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        
        # --- Animation for Lecture Line 1 ---
        # Display 'Parent Population' histogram [Asset: ...]
        self.place_at_grid(hist_svg, 'B2', scale_factor=0.7)
        self.play(FadeIn(hist_svg))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # --- Animation for Lecture Line 2 ---
        self.lecture[1].set_color(RED)
        # 'Professor Squirrel' icon [Asset: ...] appears and picks 5 random samples.
        squirrel_svg.scale(0.5)
        self.place_at_grid(squirrel_svg, 'C3')
        self.play(FadeIn(squirrel_svg))
        
        dot_cloud1 = VGroup(*[Dot(radius=0.03, color=BLUE) for _ in range(50)]).arrange_in_grid(rows=5, cols=10)
        dot_cloud_group = VGroup(dot_cloud1)
        self.place_in_area(dot_cloud_group, 'C2', 'D5', scale_factor=0.6)
        self.play(FadeIn(dot_cloud_group))
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        self.lecture[2].set_color(GREEN)
        # Histograms (using rectangles as visual representation)
        hist1 = VGroup(*[Rectangle(height=0.1*i, width=0.2, color=BLUE, fill_opacity=0.8) for i in [1, 3, 5, 3, 1]]).arrange(RIGHT, buff=0.05)
        hist_group = VGroup(hist1)
        self.place_in_area(hist_group, 'E2', 'F5', scale_factor=0.7)
        
        self.play(Create(hist_group))
        self.wait(2)
