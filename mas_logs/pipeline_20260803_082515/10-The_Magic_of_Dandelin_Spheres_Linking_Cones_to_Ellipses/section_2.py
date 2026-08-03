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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Prerequisite: The Two-Tangent Theorem"
        lines = [
            "Two tangents from a point to a sphere are equal.",
            "Think of it like an ice cream cone shape.",
            "These equal lengths are the key to our proof."
        ]
        
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Colors: Sphere (#00BFFF), Point P (#FFFFFF), Tangents (#FFFF00)
        sphere_color = "#00BFFF"
        point_color = "#FFFFFF"
        tangent_color = "#FFFF00"

        # Sphere as a 2D circle with a slight gradient or inner circles to look spherical
        sphere = Circle(radius=1.2, color=sphere_color, fill_opacity=0.3)
        sphere_glow = Circle(radius=1.2, color=sphere_color, fill_opacity=0.1).scale(1.1)
        sphere_group = VGroup(sphere, sphere_glow)
        # Fix for Issue 19: Move sphere group to D4-F6 area
        self.place_in_area(sphere_group, 'D4', 'F6', scale_factor=1.0)
        
        center_point = sphere_group.get_center()
        
        # Point P outside the sphere
        p_dot = Dot(color=point_color)
        # Fix for Issue 18: Move p_dot to B3
        self.place_at_grid(p_dot, 'B3', scale_factor=0.8)
        p_label = Text("P", font_size=20, color=point_color).next_to(p_dot, LEFT, buff=0.1)
        
        # Calculate Tangent points A and B
        p_pos = p_dot.get_center()
        vec_cp = p_pos - center_point
        dist_cp = np.linalg.norm(vec_cp)
        radius = 1.2 # Based on the sphere radius definition
        
        # Angle from CP to Tangents
        alpha = np.arccos(radius / dist_cp)
        angle_cp = np.arctan2(vec_cp[1], vec_cp[0])
        
        pos_a = center_point + np.array([
            radius * np.cos(angle_cp + alpha),
            radius * np.sin(angle_cp + alpha),
            0
        ])
        pos_b = center_point + np.array([
            radius * np.cos(angle_cp - alpha),
            radius * np.sin(angle_cp - alpha),
            0
        ])
        
        dot_a = Dot(pos_a, radius=0.04, color=tangent_color)
        dot_b = Dot(pos_b, radius=0.04, color=tangent_color)
        label_a = Text("A", font_size=18, color=tangent_color).next_to(dot_a, UR, buff=0.05)
        label_b = Text("B", font_size=18, color=tangent_color).next_to(dot_b, DR, buff=0.05)
        
        line_pa = Line(p_pos, pos_a, color=tangent_color)
        line_pb = Line(p_pos, pos_b, color=tangent_color)

        self.play(self.lecture[0].animate.set_color(tangent_color))
        self.play(FadeIn(sphere_group), FadeIn(p_dot), Write(p_label))
        self.play(Create(line_pa), Create(line_pb))
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the "ice cream cone" area
        cone_polygon = Polygon(p_pos, pos_a, center_point, pos_b, color=tangent_color, fill_opacity=0.1, stroke_width=0)
        
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(tangent_color))
        self.play(FadeIn(cone_polygon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show labels 'L' and equality indicators
        # Positions adjusted for better visibility near the center of segments
        label_l1 = Text("L", font_size=24, color=tangent_color).move_to(line_pa.get_center() + UP*0.3 + LEFT*0.1)
        label_l2 = Text("L", font_size=24, color=tangent_color).move_to(line_pb.get_center() + DOWN*0.3 + RIGHT*0.1)
        
        # Tick marks for equality
        tick1 = Line(ORIGIN, UP*0.2, color=tangent_color).rotate(line_pa.get_angle() + PI/2).move_to(line_pa.get_center())
        tick2 = Line(ORIGIN, UP*0.2, color=tangent_color).rotate(line_pb.get_angle() + PI/2).move_to(line_pb.get_center())

        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(tangent_color))
        self.play(Write(label_l1), Write(label_l2), Create(tick1), Create(tick2))
        self.play(Indicate(line_pa), Indicate(line_pb))
        self.wait(2)
        
        # Final state color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
