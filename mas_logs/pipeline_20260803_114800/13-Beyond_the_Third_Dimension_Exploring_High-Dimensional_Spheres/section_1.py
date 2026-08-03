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
        title = "The Dimension Ladder: From Circles to Spheres"
        lines = [
            "Imagine moving from one dimension to three.",
            "A circle is points at distance 'r' in 2D.",
            "A sphere extends this concept to 3D space.",
            "The distance formula is the heart of these shapes.",
            "All spheres share this fundamental geometric definition."
        ]
        self.setup_layout(title, lines)

        # Assets/Character Setup
        # Asset: Pointy the Penguin character (#FFD700)
        pointy = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pointy.svg")
        pointy.set_color("#FFD700")
        
        # Asset: Bubble for 3D sphere
        bubble_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bubble.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        line_1d = Line(start=LEFT, end=RIGHT, color="#FFFFFF")
        # Fix Issue 28: Positioning line_1d and label_1d
        self.place_in_area(line_1d, 'A2', 'A5', scale_factor=1.0)
        label_1d = Text("1D: Interval", font_size=20, color="#FFFFFF")
        self.place_at_grid(label_1d, 'A1', scale_factor=1.0)
        
        dot = Dot(color="#FFFFFF")
        dot.move_to(line_1d.get_left())
        
        self.play(Create(line_1d), Write(label_1d))
        self.play(dot.animate.move_to(line_1d.get_right()), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        circle_2d = Circle(radius=1.0, color=WHITE)
        # Fix Issue 29: Positioning circle_2d and label_2d
        self.place_in_area(circle_2d, 'B2', 'C5', scale_factor=0.8)
        label_2d = Text("2D: Circle", font_size=20, color=WHITE)
        self.place_at_grid(label_2d, 'B1', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(line_1d.copy(), circle_2d),
            Write(label_2d),
            FadeOut(dot)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # 3D sphere representation using bubble asset and ellipses
        sphere_lines = VGroup(
            Ellipse(width=2.0, height=0.5, color=GRAY_B),
            Ellipse(width=0.5, height=2.0, color=GRAY_B)
        )
        bubble_svg.stretch_to_fit_width(2.0)
        bubble_svg.stretch_to_fit_height(2.0)
        sphere_3d = VGroup(bubble_svg, sphere_lines)
        
        # Fix Issue 29: Positioning sphere_3d and label_3d
        self.place_in_area(sphere_3d, 'D2', 'E5', scale_factor=0.4)
        label_3d = Text("3D: Sphere", font_size=20, color=WHITE)
        self.place_at_grid(label_3d, 'D1', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(circle_2d.copy(), sphere_3d),
            Write(label_3d)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        formula = MathTex(r"\sqrt{x^2 + y^2 + z^2} = r", color="#00FFFF", font_size=32)
        # Fix Issue 30: Positioning formula at F2-F5
        self.place_in_area(formula, 'F2', 'F5', scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Fix Issue 30: Pointy starts at F1
        self.place_at_grid(pointy, 'F1', scale_factor=0.6)
        self.play(FadeIn(pointy))
        
        # Pointy hops between visual areas as per storyboard
        # Hop to 1D line area
        self.play(pointy.animate.move_to(self.grid["A2"]), run_time=1)
        # Hop to 2D circle area
        self.play(pointy.animate.move_to(self.grid["B2"]), run_time=1)
        # Jump inside the 3D bubble/sphere
        self.play(pointy.animate.move_to(sphere_3d.get_center()), run_time=1)
        
        self.wait(2)
