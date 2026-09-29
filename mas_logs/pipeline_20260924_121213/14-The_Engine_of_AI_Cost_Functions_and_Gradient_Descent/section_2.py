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
        lecture_lines = ["Cost functions map error across weight landscapes.", "Think of it as a mountain range.", "Goal: reach the valley of minimum error."]
        self.setup_layout("The Cost Function: The Landscape of Errors", lecture_lines)

        # Assets
        mountain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        
        # Axes
        axes = Axes(x_range=[-3, 3], y_range=[0, 4], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, color="#00FFFF")
        
        # Positioning per issues
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.5)
        curve.move_to(axes.get_center())
        self.place_at_grid(mountain_icon, 'B5', scale_factor=0.5)
        
        # Points
        yellow_point = Dot(color="#FFFF00")
        green_point = Dot(color="#00FF00")
        
        self.place_at_grid(yellow_point, 'C4', scale_factor=0.6)
        self.place_at_grid(green_point, 'E4', scale_factor=0.6)
        
        # Initial position
        hiker = Dot(color="#FFFFFF").move_to(axes.c2p(-2, 3))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.play(Create(axes), Create(curve), FadeIn(mountain_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        self.play(FadeIn(hiker))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(FadeIn(yellow_point), FadeIn(green_point))
        self.play(hiker.animate.move_to(green_point.get_center()), run_time=2)
        self.wait(1)
