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
        self.setup_layout("Prerequisite: Euler's Planar Formula", [
            "We treat the circle as a planar graph.",
            "Euler's formula states V minus E plus F equals 2.",
            "For regions, R equals E minus V plus 1."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        
        # Planar graph representation (Circle with chords)
        circle = Circle(radius=1.0, color="#87CEEB")
        points = [circle.point_at_angle(a * DEGREES) for a in [45, 135, 225, 315]]
        vertices = VGroup(*[Dot(p, color="#87CEEB", radius=0.08) for p in points])
        chords = VGroup(
            Line(points[0], points[1], color="#87CEEB"),
            Line(points[1], points[2], color="#87CEEB"),
            Line(points[2], points[3], color="#87CEEB"),
            Line(points[3], points[0], color="#87CEEB"),
            Line(points[0], points[2], color="#87CEEB")
        )
        graph_group = VGroup(circle, vertices, chords)
        self.place_in_area(graph_group, "A1", "C3", scale_factor=1.0)
        
        self.play(Create(circle))
        self.play(Create(vertices))
        self.play(Create(chords))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Euler's Formula
        euler_formula = MathTex("V - E + F = 2", color="#FFFF00")
        # Issue 30: Scale 1.2 was too large; adjusted area to B4-B6 for better vertical spacing
        self.place_in_area(euler_formula, "B4", "B6", scale_factor=1.0)
        
        # Triangle Verification
        tri_pts = [
            np.array([0, 0.6, 0]), 
            np.array([-0.6, -0.4, 0]), 
            np.array([0.6, -0.4, 0])
        ]
        triangle = Polygon(*tri_pts, color="#FFFF00")
        tri_v = VGroup(*[Dot(p, color="#FFFF00", radius=0.08) for p in tri_pts])
        
        # Triangle labels
        v_label = MathTex("V=3", font_size=24, color="#FFFF00")
        e_label = MathTex("E=3", font_size=24, color="#FFFF00")
        f_label = MathTex("F=2", font_size=24, color="#FFFF00")
        
        # Position labels
        v_label.next_to(tri_pts[0], UP, buff=0.1)
        e_label.next_to(Line(tri_pts[1], tri_pts[2]), DOWN, buff=0.1)
        f_label.move_to(triangle.get_center())
        
        # Verification calculation text
        verif_calc = MathTex("3 - 3 + 2 = 2", font_size=28, color="#FFFF00")
        verif_calc.next_to(triangle, RIGHT, buff=0.4)
        
        # Add labels to group for placement
        tri_verification = VGroup(triangle, tri_v, v_label, e_label, f_label, verif_calc)
        # Issue 31: Group was too close to derivation; reduced scale factor.
        self.place_in_area(tri_verification, "D1", "F3", scale_factor=0.9)
        
        # Animation sequence
        self.play(Write(euler_formula))
        self.play(Create(triangle), Create(tri_v))
        self.play(Write(v_label), Write(e_label), Write(f_label))
        self.play(Write(verif_calc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Derivation: R = E - V + 1
        derivation = VGroup(
            MathTex("R = F - 1", color="#FFFFFF"),
            MathTex("F = 2 - V + E", color="#FFFFFF"),
            MathTex("R = E - V + 1", color="#FFFFFF")
        ).arrange(DOWN, buff=0.5)
        
        # Issue 32: Sequence was too dense; reduced scale factor for buffer.
        self.place_in_area(derivation, "D4", "F6", scale_factor=0.9)
        
        self.play(Write(derivation[0]))
        self.wait(0.5)
        self.play(Write(derivation[1]))
        self.wait(0.5)
        self.play(Write(derivation[2]))
        self.wait(2)
