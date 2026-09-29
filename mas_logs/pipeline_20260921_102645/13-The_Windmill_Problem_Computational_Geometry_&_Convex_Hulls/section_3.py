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
        self.setup_layout("The Mechanical Logic: State Transitions", [
            "Identify current pivot and point.",
            "Calculate all relative angles.",
            "Choose the smallest clockwise rotation.",
            "Shift pivot to the new point.",
            "Repeat the mechanical process."
        ])
        
        # Define objects
        dots = VGroup(*[Dot(color=BLUE) for _ in range(5)])
        self.place_at_grid(dots[0], "B4")
        self.place_at_grid(dots[1], "C6")
        self.place_at_grid(dots[2], "D3")
        self.place_at_grid(dots[3], "E4")
        self.place_at_grid(dots[4], "B6")
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        pivot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pivot.svg")
        
        line = Line(dots[0].get_center(), dots[1].get_center(), color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(ruler, "A2", scale_factor=0.3)
        self.play(FadeIn(dots), FadeIn(ruler), Create(line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(protractor, "C2", scale_factor=0.3)
        self.play(FadeOut(ruler), FadeIn(protractor))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Indicate(dots[4]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.place_at_grid(pivot, "F3", scale_factor=0.3)
        new_line = Line(dots[4].get_center(), dots[0].get_center(), color=YELLOW)
        self.play(FadeOut(protractor), FadeOut(line), FadeIn(pivot), Create(new_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(Indicate(VGroup(new_line, dots[4], pivot)))
        self.wait(2)
