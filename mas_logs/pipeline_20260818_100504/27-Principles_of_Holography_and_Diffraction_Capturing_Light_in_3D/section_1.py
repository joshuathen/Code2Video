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
        self.setup_layout("Prerequisite: The Wave Nature of Light", [
            "Light behaves as a propagating wave.",
            "Waves possess both crests and troughs.",
            "Interference patterns encode crucial information."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg]
        source = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg")
        self.place_at_grid(source, 'B5', scale_factor=0.7)
        
        waves = VGroup()
        for i in range(1, 4):
            wave = Circle(radius=0.4 * i, color=WHITE, stroke_width=2)
            wave.move_to(source.get_center())
            waves.add(wave)
        
        self.play(FadeIn(source), Create(waves))
        self.play(
            waves.animate.scale(2).set_opacity(0),
            run_time=2,
            rate_func=linear
        )
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg]
        wave_graph = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[-2, 2], color="#FFFF00")
        self.place_in_area(wave_graph, 'A2', 'C4', scale_factor=0.6)
        
        # Keep lightbulb as reference
        light_bulb_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg")
        self.place_at_grid(light_bulb_2, 'B5', scale_factor=0.4)
        
        self.play(Create(wave_graph), FadeIn(light_bulb_2))
        self.lecture[1].set_color("#FFFF00")
        
        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg]
        light_bulb_3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg")
        self.place_at_grid(light_bulb_3, 'B5', scale_factor=0.4)
        
        obstacle = Rectangle(height=2, width=0.2, color=WHITE, fill_opacity=0.5)
        self.place_at_grid(obstacle, 'C5', scale_factor=0.5)
        
        interference = VGroup()
        for i in range(-5, 6):
            line = Line(start=np.array([0, i*0.2, 0]), end=np.array([0.5, i*0.2, 0]), color="#00FF00")
            interference.add(line)
        self.place_in_area(interference, 'D4', 'E6', scale_factor=0.5)
        
        self.play(FadeIn(light_bulb_3), FadeIn(obstacle), FadeIn(interference))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
