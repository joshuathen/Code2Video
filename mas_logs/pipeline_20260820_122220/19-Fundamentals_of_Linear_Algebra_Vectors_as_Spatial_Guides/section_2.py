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
        lecture_lines = ["Vector addition represents sequential movements.", 
                         "Place the second tail at the first head.", 
                         "The resultant vector connects origin to finish."]
        self.setup_layout("Vector Addition: The Head-to-Tail Rule", lecture_lines)
        
        # Fixing Issue 24 & 26: Axes positioning and origin marker
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], x_length=4, y_length=4, axis_config={"include_tip": True})
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        self.add(axes)
        
        origin_point = Dot(color=WHITE)
        self.place_at_grid(origin_point, 'D4', scale_factor=0.5)
        self.add(origin_point)

        # Assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        v1 = Arrow(axes.c2p(0,0), axes.c2p(3,2), buff=0, color="#00FFFF")
        v2 = Arrow(axes.c2p(0,0), axes.c2p(1,4), buff=0, color="#FF4500")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(pencil, 'C4', scale_factor=0.5)
        self.play(FadeIn(pencil), Create(v1), Create(v2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.place_at_grid(ruler, 'B5', scale_factor=0.5)
        self.play(FadeIn(ruler), v2.animate.shift(axes.c2p(3,2) - axes.c2p(0,0)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        resultant = Arrow(axes.c2p(0,0), axes.c2p(4,6), buff=0, color="#FFD700")
        self.play(Create(resultant))
        self.wait(2)
