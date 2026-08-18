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
        lecture_lines = ["Adding two independent variables corresponds to convolution.", 
                         "Two narrow bell curves merge into one wider.", 
                         "Wider curve reflects increased uncertainty."]
        self.setup_layout("Visualizing the 'Convolution' Effect", lecture_lines)
        
        # Elements
        func1 = FunctionGraph(lambda x: 0.5 * np.exp(-x**2), x_range=[-3, 3], color="#00FFFF")
        func2 = FunctionGraph(lambda x: 0.5 * np.exp(-x**2), x_range=[-3, 3], color="#00FFFF")
        result = FunctionGraph(lambda x: 0.3 * np.exp(-x**2 / 2), x_range=[-4, 4], color="#FF00FF")
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(func1, "A1", "B3", scale_factor=0.6)
        self.place_in_area(func2, "C1", "D3", scale_factor=0.6)
        self.place_at_grid(bell_icon, "A5", scale_factor=0.5)
        self.play(Create(func1), Create(func2), FadeIn(bell_icon))
        self.play(func1.animate.shift(RIGHT * 1.0), func2.animate.shift(RIGHT * 1.0), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_in_area(result, "E1", "F6", scale_factor=0.8)
        self.play(FadeOut(func1), FadeOut(func2), FadeOut(bell_icon), Create(result))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        label = Text("Wider Variance", font_size=20, color="#FFFF00")
        self.place_at_grid(label, "E4", scale_factor=0.9)
        self.play(Write(label))
        self.wait(1)
