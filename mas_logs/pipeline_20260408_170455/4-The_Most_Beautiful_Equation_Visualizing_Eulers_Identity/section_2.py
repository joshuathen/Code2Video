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
        # Define layout
        title_text = "Prerequisite: The 2D Playground"
        lecture_lines = [
            "Real numbers live on a flat horizontal line.",
            "The imaginary unit i adds a vertical dimension.",
            "Multiplying by i creates a ninety-degree turn."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_REAL = "#FFFFFF"
        COLOR_IMAG = "#5555FF"
        COLOR_ROT = "#FFFF00"
        
        # Origin point for the complex plane
        origin_pos = self.grid['D3']
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_REAL))
        
        # Horizontal Real Axis
        real_axis = Line(self.grid['D1'], self.grid['D6'], color=COLOR_REAL)
        real_label = Text("Real", font_size=20, color=COLOR_REAL)
        # Fix for Issue 42: Scale real_label to 0.8
        self.place_at_grid(real_label, 'D6', scale_factor=0.8)
        
        # Ground Asset - Issue 36
        ground_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ground.svg")
        self.place_in_area(ground_asset, 'E1', 'E6', scale_factor=0.5)
        
        self.play(Create(real_axis), FadeIn(real_label), FadeIn(ground_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_IMAG)
        )
        
        # Vertical Imaginary Axis
        imag_axis = Line(self.grid['F3'], self.grid['A3'], color=COLOR_IMAG)
        imag_label = Text("Imaginary", font_size=20, color=COLOR_IMAG)
        # Fix for Issue 41: Scale imag_label to 0.7
        self.place_at_grid(imag_label, 'A3', scale_factor=0.7)
        
        self.play(Create(imag_axis), FadeIn(imag_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_ROT)
        )
        
        # Point at 1 on real axis (1 unit right of origin D3 is D4)
        pos_1 = self.grid['D4']
        point = Dot(pos_1, color=COLOR_REAL)
        label_1 = Text("1", font_size=28, color=COLOR_REAL)
        self.place_at_grid(label_1, 'E4', scale_factor=1.0)
        
        self.play(FadeIn(point), Write(label_1))
        self.wait(0.5)
        
        # Rotation path (90-degree CCW from D4 to C3 around D3)
        arc = Arc(
            radius=1.0,
            start_angle=0,
            angle=PI/2,
            arc_center=origin_pos,
            color=COLOR_ROT
        )
        
        # Target position 'i' (1 unit up from origin D3 is C3)
        label_i = Text("i", font_size=28, color=COLOR_IMAG, slant=ITALIC)
        # Positioning label 'i' to the left of the point at C3
        self.place_at_grid(label_i, 'C2', scale_factor=1.0)
        
        # Animate the turn
        self.play(
            MoveAlongPath(point, arc),
            point.animate.set_color(COLOR_IMAG),
            label_1.animate.set_opacity(0),
            run_time=2,
            rate_func=smooth
        )
        self.play(FadeIn(label_i))
        
        self.wait(2)
        
        # Final cleanup/rest
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
