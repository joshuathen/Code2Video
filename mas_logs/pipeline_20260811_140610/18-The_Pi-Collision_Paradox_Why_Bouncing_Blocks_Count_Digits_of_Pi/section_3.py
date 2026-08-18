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
        self.setup_layout("The Geometric Transformation", [
            "Mass ratios determine the collision count.",
            "Zigzag paths mirror circular arcs.",
            "The path length traces digits of Pi.",
            "Mass ratio 1 to 100 yields 31.",
            "Higher ratios yield more digits of Pi."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display two mass blocks using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg] showing ratio 1:100.
        block1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        block2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=GREEN)
        ratio_label = Text("Ratio 1 : 100", font_size=24)
        group = VGroup(block1, block2, ratio_label).arrange(DOWN)
        self.place_in_area(group, 'A4', 'C6', scale_factor=0.5)
        self.play(FadeIn(group))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Draw a circular arc inside the phase space grid #FF9900.
        arc = Arc(radius=1.5, start_angle=0, angle=PI, color="#FF9900")
        self.place_at_grid(arc, 'E2', scale_factor=0.7)
        self.play(Create(arc))
        self.lecture[1].set_color("#FF9900")

        # === Animation for Lecture Line 3 ===
        # Animate a vector tracing the path length calculation.
        vec = Vector(direction=RIGHT, color=YELLOW)
        self.place_at_grid(vec, 'E4', scale_factor=0.6)
        self.play(GrowArrow(vec))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Highlight the number 31 on the screen in #FF0000.
        count_num = Text("31", color="#FF0000", font_size=40)
        self.place_at_grid(count_num, 'B6', scale_factor=0.8)
        self.play(Write(count_num))
        self.lecture[3].set_color("#FF0000")

        # === Animation for Lecture Line 5 ===
        # Show a progression of digits 3.1415... appearing near the path.
        pi_digits = Text("3 . 1 4 1 5 ...", font_size=24, color=WHITE)
        self.place_at_grid(pi_digits, 'F6', scale_factor=0.6)
        self.play(FadeIn(pi_digits))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
