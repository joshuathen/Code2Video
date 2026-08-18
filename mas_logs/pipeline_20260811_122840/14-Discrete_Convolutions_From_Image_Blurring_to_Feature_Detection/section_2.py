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
        lecture_lines = [
            "The discrete formula defines convolution.",
            "Flip the kernel before sliding.",
            "Weights determine the output transformation.",
            "Sliding across input produces output sequence.",
            "Moving average simplifies 1D data."
        ]
        self.setup_layout("Mathematical Mechanics", lecture_lines)
        
        # Elements
        input_data = [1, 2, 3]
        kernel_data = [0.5, 0.5]
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        stencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stencil.svg")
        
        input_mobj = VGroup(*[Square(side_length=0.6, color=WHITE).add(Text(str(x), font_size=20)) for x in input_data]).arrange(RIGHT, buff=0.1)
        kernel_mobj = VGroup(*[Square(side_length=0.6, color=YELLOW).add(Text(str(x), font_size=20)) for x in kernel_data]).arrange(RIGHT, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(GREEN))
        self.place_in_area(input_mobj, 'B3', 'B5', scale_factor=0.9)
        self.place_at_grid(ruler, 'B6', scale_factor=0.5)
        self.play(FadeIn(input_mobj), FadeIn(ruler))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_in_area(kernel_mobj, 'C3', 'C5', scale_factor=0.9)
        self.play(FadeIn(kernel_mobj))
        self.play(kernel_mobj.animate.rotate(PI))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(calc, 'C6', scale_factor=0.5)
        self.play(FadeIn(calc), Indicate(kernel_mobj))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(TEAL))
        output_cell = Square(side_length=0.6, color=TEAL).add(Text("1.5", font_size=20))
        self.place_at_grid(output_cell, 'D4', scale_factor=0.8)
        self.place_at_grid(stencil, 'D6', scale_factor=0.5)
        self.play(FadeIn(output_cell), FadeIn(stencil))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(TEAL))
        self.play(kernel_mobj.animate.shift(RIGHT * 0.7))
        output_cell2 = Square(side_length=0.6, color=TEAL).add(Text("2.5", font_size=20))
        self.place_at_grid(output_cell2, 'D5', scale_factor=0.8)
        self.play(FadeIn(output_cell2))
        self.wait(1)
