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
        self.setup_layout("Summary & Application", [
            "ODEs are the language of nature.",
            "They model behavior over time accurately.",
            "Everything from bridges to heart rates."
        ])
        
        # === Animation for Lecture Line 1 ===
        ode_nature = Text("ODEs = Nature", font_size=36, color=WHITE)
        self.place_in_area(ode_nature, 'A3', 'B5', scale_factor=0.9)
        self.play(Write(ode_nature))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Using placeholder icons if SVG load fails, but attempting SVG load
        try:
            bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color=TEAL)
            heart = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg", color=TEAL)
        except:
            bridge = Square(side_length=1.0, color=TEAL)
            heart = Circle(radius=0.5, color=TEAL)

        self.place_at_grid(bridge, 'D1', scale_factor=0.7)
        self.label(bridge, 'Bridge', 'D1', offset_y=-0.8)
        
        self.place_at_grid(heart, 'D4', scale_factor=0.7)
        self.label(heart, 'Heart Rate', 'D4', offset_y=-0.8)
        
        self.play(FadeIn(bridge), FadeIn(heart))
        self.lecture[1].set_color(TEAL)

        # === Animation for Lecture Line 3 ===
        flow = VGroup(*[Line(start=self.grid["D1"] + RIGHT*0.4, end=self.grid["D4"] + LEFT*0.4, color=RED) for _ in range(1)])
        self.play(Create(flow))
        self.lecture[2].set_color(RED)
        
        self.wait(2)
