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
        self.setup_layout("Beyond Arrows: Function Spaces", [
            "Functions like polynomials can also be vectors.", 
            "Summing two functions yields another function.", 
            "This expands our horizon beyond physical geometry."
        ])
        
        # Define objects
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True})
        self.place_in_area(axes, 'A2', 'F5', scale_factor=0.65)
        
        func_curve = axes.plot(lambda x: 0.5 * x**2, color="#00FFFF")
        f_x_label = MathTex("f(x)", color="#00FFFF")
        self.place_at_grid(f_x_label, 'B6', scale_factor=0.7)
        
        # Asset: graph icon - Switched from ImageMobject to SVGMobject as it is an .svg file
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        self.place_at_grid(graph_icon, 'A6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(axes), Create(func_curve), Write(f_x_label), FadeIn(graph_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Summing two functions example
        func_g = axes.plot(lambda x: -0.3 * x, color="#FFFF00")
        func_sum = axes.plot(lambda x: 0.5 * x**2 - 0.3 * x, color="#FF00FF")
        fg_label = MathTex("(f+g)(x)", color="#FF00FF")
        self.place_at_grid(fg_label, 'E6', scale_factor=0.7)
        
        self.play(Create(func_g), Create(func_sum), Write(fg_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        # Final decorative element
        bounding_box = SurroundingRectangle(VGroup(func_curve, func_g, func_sum), color="#FF0000")
        self.place_in_area(bounding_box, 'A2', 'F5', scale_factor=0.9)
        self.play(Create(bounding_box))
        self.wait(2)
