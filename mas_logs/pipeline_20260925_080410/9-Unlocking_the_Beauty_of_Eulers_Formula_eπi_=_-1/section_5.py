from manim import *
import numpy as np

# Apply configuration to prevent race condition during LaTeX cleanup
config.no_latex_cleanup = True

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
        lecture_lines = [
            "Set the input x to pi.",
            "Rotation of pi radians reaches the left side.",
            "The result is exactly negative one.",
            "We have derived Euler's identity.",
            "e to the pi i is negative one."
        ]
        self.setup_layout("The Grand Finale: e^(πi) = -1", lecture_lines)
        
        compass = SVGMobject('/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg')
        circle = Circle(radius=1.5, color=BLUE)
        self.place_in_area(circle, 'B2', 'E5')
        self.add(circle)
        
        point = Dot(color=WHITE)
        point.move_to(circle.point_from_proportion(0))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color('#FFFFFF')
        self.place_at_grid(compass, 'A1', 0.5)
        self.add(compass, point)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color('#FF0000')
        arc = Arc(radius=1.5, start_angle=0, angle=PI, color='#FF0000')
        self.play(MoveAlongPath(point, arc), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color('#00FF00')
        point.set_color('#00FF00')
        self.play(Flash(point, color='#00FF00'))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color('#FFFF00')
        identity = MathTex(r"e^{\pi i} = -1", color='#FFFF00')
        self.place_at_grid(identity, 'C4', 1.5)
        self.play(Write(identity))
        
        # Frame with compass
        frame = SurroundingRectangle(identity, color=WHITE, buff=0.1)
        self.play(Create(frame), run_time=1)
        self.play(FadeOut(frame), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color('#FFFF00')
        self.wait(1)
