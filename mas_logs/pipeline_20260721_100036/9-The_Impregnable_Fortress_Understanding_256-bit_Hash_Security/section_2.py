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
        title = "Prerequisite: The Power of Doubling"
        lines = [
            "One switch offers two possible paths.",
            "Two switches double the choices to four.",
            "Adding switches grows the possibilities exponentially."
        ]
        self.setup_layout(title, lines)

        # Assets
        SWITCH_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg"

        # Colors
        color1 = WHITE
        color2 = "#ADD8E6" # Light Blue
        color3 = "#0000FF" # Blue

        # === Animation for Lecture Line 1 ===
        # One switch offers two possible paths.
        self.play(self.lecture[0].animate.set_color(color1))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg]
        switch1 = SVGMobject(SWITCH_PATH).set_color(color1)
        self.place_at_grid(switch1, "C1", scale_factor=0.8)
        
        path1a = Line(self.grid["C1"], self.grid["B2"], color=color1)
        path1b = Line(self.grid["C1"], self.grid["D2"], color=color1)
        
        label0 = Text("0", font_size=20, color=color1)
        label1 = Text("1", font_size=20, color=color1)
        # Issue 36: label0 at B2, label1 at D2
        self.place_at_grid(label0, "B2", scale_factor=0.8)
        self.place_at_grid(label1, "D2", scale_factor=0.8)
        
        self.play(Create(switch1))
        self.play(Create(path1a), Create(path1b))
        self.play(Write(label0), Write(label1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Two switches double the choices to four.
        self.play(self.lecture[1].animate.set_color(color2))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg]
        switch2 = SVGMobject(SWITCH_PATH).set_color(color2)
        switch3 = SVGMobject(SWITCH_PATH).set_color(color2)
        # Issue 35: switch2 at B3, switch3 at D3
        self.place_at_grid(switch2, "B3", scale_factor=0.8)
        self.place_at_grid(switch3, "D3", scale_factor=0.8)
        
        path2a = Line(self.grid["B3"], self.grid["A4"], color=color2)
        path2b = Line(self.grid["B3"], self.grid["B4"], color=color2)
        path2c = Line(self.grid["D3"], self.grid["D4"], color=color2)
        path2d = Line(self.grid["D3"], self.grid["E4"], color=color2)
        
        label00 = Text("00", font_size=18, color=color2)
        label01 = Text("01", font_size=18, color=color2)
        label10 = Text("10", font_size=18, color=color2)
        label11 = Text("11", font_size=18, color=color2)
        self.place_at_grid(label00, "A5", scale_factor=0.8)
        self.place_at_grid(label01, "B5", scale_factor=0.8)
        self.place_at_grid(label10, "D5", scale_factor=0.8)
        self.place_at_grid(label11, "E5", scale_factor=0.8)
        
        self.play(Create(switch2), Create(switch3))
        self.play(Create(path2a), Create(path2b), Create(path2c), Create(path2d))
        self.play(Write(label00), Write(label01), Write(label10), Write(label11))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adding switches grows the possibilities exponentially.
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Group all previous elements to fade out
        group12 = VGroup(switch1, switch2, switch3, path1a, path1b, path2a, path2b, path2c, path2d,
                        label0, label1, label00, label01, label10, label11)
        
        # Wall of 256 switches [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg]
        # Using a representative number to avoid performance issues
        switches_256 = VGroup(*[SVGMobject(SWITCH_PATH).set_color(color3) for _ in range(36)])
        switches_256.arrange_in_grid(rows=6, cols=6, buff=0.2)
        
        # Issue 34: switches_256 and fog in B2 to F5
        self.place_in_area(switches_256, "B2", "F5", scale_factor=0.7)
        
        # Glowing fog
        fog = Rectangle(width=4.0, height=4.0, color=color3, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(fog, "B2", "F5", scale_factor=0.8)
        
        self.play(FadeOut(group12))
        self.play(FadeIn(switches_256))
        self.play(FadeIn(fog))
        self.wait(2)
