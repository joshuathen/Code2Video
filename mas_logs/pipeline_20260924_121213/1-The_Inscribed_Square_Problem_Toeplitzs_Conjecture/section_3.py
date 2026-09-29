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
        self.setup_layout("The Topological Mapping: Circles to Squares", [
            "We map squares to 3D space.",
            "Potential squares form a surface.",
            "Missing squares indicate an obstruction.",
            "Vertices slide along the curve.",
            "Mismatch error tracks rotation."
        ])
        
        # Load assets
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        icon_circle = SVGMobject(asset_path).set_color(GREEN)
        icon_line = SVGMobject(asset_path).set_color(YELLOW)
        
        # Elements
        circle = Circle(radius=0.5, color=GREEN)
        self.place_at_grid(circle, 'D2', scale_factor=0.8)
        circle_label = Text("Circle", font_size=18, color=GREEN).next_to(circle, UP)
        
        square = Square(side_length=0.7, color="#FF00FF")
        self.place_at_grid(square, 'D5', scale_factor=0.8)
        square_label = Text("Mapping", font_size=18, color="#FF00FF").next_to(square, UP)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.place_at_grid(icon_circle, 'C2', scale_factor=0.5)
        self.play(FadeIn(circle), FadeIn(circle_label), FadeIn(icon_circle))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.play(FadeIn(square), FadeIn(square_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        dot = Dot(color=RED).move_to(circle.get_right())
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(MoveAlongPath(dot, circle), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF8800"))
        connecting_line = Line(circle.get_center(), square.get_center(), color=WHITE)
        self.place_at_grid(icon_line, 'C5', scale_factor=0.5)
        self.play(Create(connecting_line), FadeIn(icon_line))
        self.wait(2)
