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
        self.setup_layout("Summoning the Dandelin Spheres", [
            "Place a small sphere above the intersecting plane.",
            "Place a larger sphere below the plane.",
            "Both spheres touch the cone along horizontal circles.",
            "Each sphere also touches the plane at one point.",
            "These points of contact are called the Dandelin foci."
        ])

        # === Visual Elements Construction ===
        
        # Load and style SVG Assets
        # Cone [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg]
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(WHITE).set_height(4.5)
        
        # Spheres [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_s = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color("#D3D3D3").set_height(1.4).move_to([0, 0.8, 0])
        sphere_l = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color("#D3D3D3").set_height(2.8).move_to([0, -1.8, 0])

        # Plane (Tilted Line)
        plane_line = Line(np.array([-2.5, 0.2, 0]), np.array([2.5, -0.8, 0])).set_color("#1E90FF")

        # Contact Circles (Represented as dashed lines on the cone cross-section)
        # Positioned where the spheres would touch the sides of the cone SVG
        contact_s = DashedLine(np.array([-0.6, 0.8, 0]), np.array([0.6, 0.8, 0]), color=WHITE)
        contact_l = DashedLine(np.array([-1.2, -1.8, 0]), np.array( [1.2, -1.8, 0]), color=WHITE)

        # Foci (Tangency points on the plane)
        # Based on plane line: y = -0.2x - 0.3
        f1_pos = np.array([0.8, -0.46, 0])
        f2_pos = np.array([-0.8, -0.14, 0])
        f1 = Dot(f1_pos, color="#FF0000")
        f2 = Dot(f2_pos, color="#FF0000")
        
        f1_label = Text("F1", font_size=24, color="#FF0000")
        f2_label = Text("F2", font_size=24, color="#FF0000")

        # Group construction for grid placement
        construction = VGroup(cone, plane_line, sphere_s, sphere_l, contact_s, contact_l, f1, f2)
        
        # Resolving Issue 19: Shift to start at B2 to avoid crowding lecture notes
        self.place_in_area(construction, "B2", "F6", scale_factor=0.8)

        # Positioning labels relative to dots after the construction group is placed
        f1_label.next_to(f1, DOWN+RIGHT, buff=0.1)
        f2_label.next_to(f2, UP+LEFT, buff=0.1)

        # === Animation sequence ===

        # === Animation for Lecture Line 1 ===
        # Place a small sphere above the intersecting plane.
        self.play(self.lecture[0].animate.set_color("#D3D3D3"))
        self.play(Create(cone), Create(plane_line))
        self.play(FadeIn(sphere_s))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place a larger sphere below the plane.
        self.play(self.lecture[1].animate.set_color("#D3D3D3"))
        self.play(FadeIn(sphere_l))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Both spheres touch the cone along horizontal circles.
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.play(Create(contact_s), Create(contact_l))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Each sphere also touches the plane at one point.
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.play(FadeIn(f1), FadeIn(f2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # These points of contact are called the Dandelin foci.
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        self.play(Write(f1_label), Write(f2_label))
        self.wait(3)
