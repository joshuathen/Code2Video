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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The curve is closed.",
            "Mismatch functions must pass through zero.",
            "This follows from topology logic.",
            "Robot arms trace the boundary.",
            "Zero marks a perfect square."
        ]
        self.setup_layout("Applying the Topology: Why it works", lecture_lines)
        
        # Elements
        curve = Circle(radius=1.0, color=BLUE)
        self.place_in_area(curve, 'B3', 'E4', scale_factor=0.6)
        
        f_x_label = MathTex("f(x)", color="#FFFFFF")
        f_neg_x_label = MathTex("f(-x)", color="#FFFFFF")
        
        robot_svg_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot_svg_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(curve))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_at_grid(f_x_label, 'B5', scale_factor=0.7)
        self.place_at_grid(f_neg_x_label, 'E5', scale_factor=0.7)
        self.place_at_grid(robot_svg_1, 'A5', scale_factor=0.3)
        self.play(Write(f_x_label), Write(f_neg_x_label), FadeIn(robot_svg_1))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        dot1 = Dot(color="#FF00FF").move_to(curve.point_from_proportion(0.1))
        dot2 = Dot(color="#FF00FF").move_to(curve.point_from_proportion(0.6))
        self.play(FadeIn(dot1), FadeIn(dot2))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        line = Line(dot1.get_center(), dot2.get_center(), color="#00FF00")
        self.play(Create(line))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        cross = Cross(color="#FFFF00").scale(0.5).move_to(line.get_center())
        self.place_at_grid(robot_svg_2, 'F5', scale_factor=0.3)
        self.play(FadeIn(cross), FadeIn(robot_svg_2))
        self.wait(1)
