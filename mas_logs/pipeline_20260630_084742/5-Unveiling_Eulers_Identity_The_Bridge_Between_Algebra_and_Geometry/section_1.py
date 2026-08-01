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
        # Initialization
        title = "The Mathematical Playground: The Complex Plane"
        # Lecture lines aligned with the updated teaching content
        lines = [
            "Real numbers live on a horizontal number line.",
            "Imaginary numbers create a new, vertical dimension.",
            "We start at the number one on the real axis.",
            "Imaginary unit i sits on the vertical axis.",
            "Multiplying by i rotates our position by 90 degrees."
        ]
        self.setup_layout(title, lines)
        
        # Pivot point (Origin) at D3 for the coordinate system
        origin_pos = self.grid['D3']
        pos_1 = self.grid['D4'] # 1 unit to the right
        pos_i = self.grid['C3'] # 1 unit up

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        real_axis = Line(start=self.grid['D1'], end=self.grid['D6'], color=WHITE)
        real_label = Text("Real", font_size=20, color=WHITE)
        # Fix Issue 32: Position at D5, scale 0.8 to prevent clipping
        self.place_at_grid(real_label, 'D5', scale_factor=0.8).shift(DOWN * 0.4)
        
        # Issue 31: Integrate foundation asset icon
        foundation_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/based.svg")
        self.place_at_grid(foundation_icon, "D1", scale_factor=0.4).shift(LEFT * 0.6)
        
        self.play(Create(real_axis), Write(real_label), FadeIn(foundation_icon), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        imag_axis = Line(start=self.grid['F3'], end=self.grid['A3'], color=WHITE)
        imag_label = Text("Imaginary", font_size=20, color=WHITE)
        # Fix Issue 33: Position at B4, scale 0.8 to avoid top-edge clipping
        self.place_at_grid(imag_label, 'B4', scale_factor=0.8).shift(LEFT * 0.8)
        
        self.play(Create(imag_axis), Write(imag_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3 color matches the dot color (#00FF00)
        self.lecture[2].set_color("#00FF00")
        dot_1 = Dot(pos_1, color="#00FF00")
        label_1 = Text("1", font_size=24, color="#00FF00").next_to(dot_1, DOWN, buff=0.1)
        
        self.play(FadeIn(dot_1), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4 color matches the dot color (#00FFFF)
        self.lecture[3].set_color("#00FFFF")
        dot_i = Dot(pos_i, color="#00FFFF")
        label_i = Text("i", font_size=24, slant=ITALIC, color="#00FFFF").next_to(dot_i, LEFT, buff=0.1)
        
        self.play(FadeIn(dot_i), Write(label_i))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5 color matches the rotation arc (#FF00FF)
        self.lecture[4].set_color("#FF00FF")
        
        # Rotation arc representing the transformation
        rotation_arc = Arc(
            radius=1.0,
            start_angle=0,
            angle=PI/2,
            arc_center=origin_pos,
            color="#FF00FF"
        )
        
        self.play(Create(rotation_arc), run_time=1.2)
        # Animate the dot's rotation to illustrate multiplication by i
        self.play(
            MoveAlongPath(dot_1, rotation_arc),
            run_time=1.5
        )
        self.wait(2)
