from manim import *
import os

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
        self.setup_layout("The Geometry of Rotation", [
            "Euler's formula describes rotating growth.",
            "Multiplying by i rotates ninety degrees.",
            "Continuous growth creates circular paths.",
            "The robotic arm traces a circle.",
            "The input angle x increases rotation."
        ])
        
        # Elements
        dot = Dot(color=WHITE)
        dot_i = Dot(color="#FF0000")
        dot_2i = Dot(color="#00FF00")
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#FFFF00")
        circle = Circle(radius=0.8, color="#FFFF00")
        
        # Setup positions based on feedback
        self.place_at_grid(dot, "D3")
        self.place_at_grid(dot_i, "D4")
        self.place_at_grid(dot_2i, "C4")
        self.place_at_grid(circle, "D4", scale_factor=1.2)
        
        # Group for area placement
        group = VGroup(dot, dot_i, dot_2i, circle, robot)
        self.place_in_area(group, "C4", "F6")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(FadeIn(dot_i))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(dot_2i))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(Create(circle), FadeIn(robot))
        self.play(Rotate(robot, angle=2*PI, about_point=circle.get_center()))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#0000FF"))
        self.wait(1)
