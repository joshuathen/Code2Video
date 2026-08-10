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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Gradient Descent: Refining the Model", [
            "Gradient descent refines the model's weights.",
            "Think of a hiker descending in fog.",
            "Small steps avoid overshooting the lowest point."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Use simpler axes and fewer elements to stay within render time.
        axes = Axes(x_range=[-2, 2, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False}, x_length=4, y_length=2).scale(0.6)
        surface = axes.plot(lambda x: 0.25 * x**2, x_range=[-2, 2], color="#8A2BE2")
        fog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fog.svg").scale(0.4)
        
        bowl = VGroup(axes, surface)
        self.place_in_area(bowl, 'B2', 'D5', scale_factor=0.8)
        self.place_in_area(fog, 'D3', 'E5', scale_factor=0.5)
        
        self.play(Create(axes), Create(surface), FadeIn(fog))
        self.lecture[0].set_color("#8A2BE2")

        # === Animation for Lecture Line 2 ===
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg").scale(0.15)
        hiker.move_to(axes.c2p(1.8, 0.25 * 1.8**2))
        
        self.play(FadeIn(hiker))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Path for hiker
        path = axes.plot(lambda x: 0.25 * x**2, x_range=[1.8, 0.2])
        self.play(MoveAlongPath(hiker, path), run_time=1.5)
        
        min_point = Dot(axes.c2p(0, 0), color="#FFD700").scale(0.6)
        self.play(FadeIn(min_point))
        self.lecture[2].set_color("#FFD700")
        self.wait(1)
