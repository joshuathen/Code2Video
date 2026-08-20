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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Normal distributions are fundamental building blocks.", "They appear frequently in nature.", "Parameters define their shape."]
        self.setup_layout("Introduction: The 'Bell Curve' Phenomenon", lecture_lines)
        
        # Bell curve function
        def normal_pdf(x, mu=0, sigma=1):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 0.5, 0.1], axis_config={"include_tip": False})
        bell_curve = axes.plot(lambda x: normal_pdf(x), color="#FFFF00")
        
        # Add assets and labels
        pebble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pebble.svg")
        label = Text("Normal Distribution", font_size=20, color="#00FFFF")
        mean_marker = Dot(color="#FF00FF")
        mean_label = Text("Mean", font_size=16, color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.55)
        self.place_at_grid(pebble, 'F3', scale_factor=0.3)
        self.play(Create(bell_curve), FadeIn(pebble))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(Indicate(bell_curve))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.place_at_grid(label, 'E4', scale_factor=0.7)
        self.place_at_grid(mean_marker, 'C4', scale_factor=0.6)
        self.place_at_grid(mean_label, 'C5', scale_factor=0.7)
        self.play(Write(label), FadeIn(mean_marker), Write(mean_label))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
