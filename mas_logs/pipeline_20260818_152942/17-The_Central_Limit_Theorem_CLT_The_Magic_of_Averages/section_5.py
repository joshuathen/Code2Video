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
        self.setup_layout("Application: Real-World Predictability", [
            "CLT allows population inference from samples.",
            "We predict outcomes without knowing original shapes.",
            "It bridges sample data to real-world certainty."
        ])

        # Assets
        icon_pop = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        icon_bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")

        # Create histogram data
        data = np.random.normal(0, 1, 1000)
        hist = BarChart(values=np.histogram(data, bins=15, range=(-3, 3))[0], bar_names=[str(i) for i in range(15)])
        hist_group = self.place_in_area(hist, "C2", "F6", scale_factor=0.4)

        # Create Normal Curve
        curve = FunctionGraph(lambda x: 15 * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), x_range=[-3, 3])
        curve.set_color(WHITE)
        self.place_in_area(curve, "C2", "F6", scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(icon_pop, "A3", scale_factor=0.4)
        self.play(FadeIn(icon_pop), Write(self.lecture[0]))
        self.lecture[0].set_color("#ADFF2F")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(hist_group), Create(curve))
        self.lecture[1].set_color("#87CEEB")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(icon_bridge, "A5", scale_factor=0.4)
        self.play(FadeIn(icon_bridge), Write(self.lecture[2]))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
