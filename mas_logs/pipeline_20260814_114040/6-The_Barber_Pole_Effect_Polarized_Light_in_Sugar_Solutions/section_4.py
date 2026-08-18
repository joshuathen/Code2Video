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
            "Malus's Law dictates the light intensity output.",
            "Varying sugar concentration changes the helix pitch.",
            "A graph maps depth to rotation angles.",
            "Periodic curves define the observed spectral colors.",
            "Rotation correlates directly to sugar concentration depth."
        ]
        self.setup_layout("Mathematical Visualization", lecture_lines)
        
        # Colors for lecture lines
        colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#FF9F1C", "#A29BFE"]
        
        # Load Sugar icon asset
        sugar_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sugar.svg")
        
        # Rotating Vector
        vec = Arrow(start=ORIGIN, end=RIGHT, color="#FFFFFF")
        vec_group = VGroup(vec, sugar_asset)
        self.place_at_grid(vec_group, 'B4', scale_factor=0.7)
        
        # Wave Function Graph
        axes = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], axis_config={"include_tip": False})
        wave = axes.plot(lambda x: np.cos(x), color="#FFE66D")
        graph_group = VGroup(axes, wave)
        self.place_in_area(graph_group, 'C3', 'D5', scale_factor=0.7)

        # Equations
        malus_eq = MathTex(r"I = I_0 \cos^2(\theta)", color="#FF6B6B")
        self.place_in_area(malus_eq, 'B3', 'C5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.play(Create(malus_eq))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.play(vec_group.animate.rotate(PI/4))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.play(Create(graph_group))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        self.play(wave.animate.set_color("#FF9F1C"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.play(vec_group.animate.rotate(PI/4))
