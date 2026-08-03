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

class Section5Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "The Cube and the 'Hiding' Sphere"
        lecture_lines = [
            "Inscribe a sphere inside a unit hypercube.",
            "In three dimensions, the sphere fills most space.",
            "As dimensions grow, the cube's corners stretch away.",
            "The sphere's volume becomes negligible in hyperspace.",
            "Large empty pockets develop in the cube's corners."
        ]
        self.setup_layout(title, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight line 1
        self.play(self.lecture[0].animate.set_color(BLUE_A))
        
        # 2D case: Square with inscribed circle
        square_2d = Square(side_length=1.4, color=BLUE_A)
        circle_2d = Circle(radius=0.7, color=WHITE)
        label_2d = Text("2D", font_size=20, color=BLUE_A)
        group_2d = VGroup(square_2d, circle_2d)
        
        self.place_in_area(group_2d, "A2", "B3", scale_factor=1.0)
        self.place_at_grid(label_2d, "A1", scale_factor=1.0)
        
        self.play(Create(square_2d), Create(circle_2d), Write(label_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight line 2
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(BLUE_B))
        
        # 3D case: Projected Cube with inscribed sphere
        cube_3d = VGroup(
            Square(side_length=1.4, color=BLUE_B),
            Square(side_length=1.4, color=BLUE_B).shift(0.3 * UR),
            Line(ORIGIN, 0.3 * UR).shift(0.7 * UL),
            Line(ORIGIN, 0.3 * UR).shift(0.7 * UR),
            Line(ORIGIN, 0.3 * UR).shift(0.7 * DL),
            Line(ORIGIN, 0.3 * UR).shift(0.7 * DR)
        )
        sphere_3d = Circle(radius=0.7, color=WHITE).shift(0.15 * UR)
        label_3d = Text("3D", font_size=20, color=BLUE_B)
        group_3d = VGroup(cube_3d, sphere_3d)
        
        self.place_in_area(group_3d, "A5", "B6", scale_factor=1.0)
        self.place_at_grid(label_3d, "A4", scale_factor=1.0)
        
        self.play(FadeIn(group_3d), FadeIn(label_3d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FF00FF"))
        
        # Transition to High-D concept
        self.play(
            FadeOut(group_2d), FadeOut(label_2d),
            FadeOut(group_3d), FadeOut(label_3d)
        )
        
        # High-D Representation
        hd_cube = Square(side_length=1.4, color=BLUE_E)
        hd_sphere = Circle(radius=0.7, color=WHITE, fill_opacity=0.3)
        hd_group = VGroup(hd_cube, hd_sphere)
        
        # Issue 41: hd_group in D3-F6, hd_label in D2 (scale 0.9)
        self.place_in_area(hd_group, "D3", "F6", scale_factor=0.8)
        hd_label = Text("High Dimensions (n=100)", font_size=20, color=BLUE_E)
        self.place_at_grid(hd_label, "D2", scale_factor=0.9)
        
        # Highlighting corners
        corner_dots = VGroup(*[Dot(hd_cube.get_corner(c), color="#FF00FF", radius=0.08) for c in [UR, UL, DR, DL]])
        
        self.play(FadeIn(hd_group), FadeIn(hd_label), FadeIn(corner_dots))
        
        # Animate "corners stretch away"
        self.play(
            hd_cube.animate.scale(2.0),
            corner_dots.animate.scale(2.0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight line 4
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Sphere becomes a tiny dot
        island_text = Text("The Tiny Island", font_size=24, color=YELLOW)
        # Issue 39: island_text at F1
        self.place_at_grid(island_text, "F1", scale_factor=0.8)
        
        self.play(
            hd_sphere.animate.scale(0.05).set_opacity(1.0),
            FadeIn(island_text),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight line 5
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#FF00FF"))
        
        # Robot Packer and Pockets
        # Issue 40: robot_packer at C5
        robot_packer = Text("Robot Packer", font_size=24, color=WHITE)
        self.place_at_grid(robot_packer, "C5", scale_factor=0.6)
        
        # Filling corners with small spheres to represent "pockets"
        # We place them at the corners of the now-scaled cube
        pockets = VGroup(*[Circle(radius=0.15, color="#FF00FF", fill_opacity=0.4, stroke_width=1).move_to(dot.get_center()) for dot in corner_dots])
        
        self.play(FadeIn(robot_packer), Create(pockets))
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
