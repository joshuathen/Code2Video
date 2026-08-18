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
        # Setup layout
        title_text = "Prerequisite: The 'Equal Tangent' Rule"
        lecture_lines = [
            "Consider a point outside a sphere.",
            "Two tangent segments drawn to the sphere are equal.",
            "This simple rule is key to our upcoming proof."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        sphere_color = "#C0C0C0"
        point_color = "#FF0000"
        tangent_point_color = "#00FF00"
        highlight_color = "#FFFF00"
        
        # Grid positions for clarity
        sphere_center_pos = self.grid["C2"]
        p_pos = self.grid["C4"] # As per Issue 25

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)
        
        # Create a sphere (Circle with gradient-like fill)
        sphere = Circle(radius=1.2, color=sphere_color, fill_opacity=0.3)
        sphere.set_fill(sphere_color, opacity=0.3)
        self.place_at_grid(sphere, "C2", scale_factor=1.3) # As per Issue 25
        
        # Point P at C4
        point_p = Dot(p_pos, color=point_color)
        label_p = MathTex("P", color=point_color).next_to(point_p, RIGHT, buff=0.1)
        
        self.play(Create(sphere), FadeIn(point_p), Write(label_p))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)

        # Geometry calculations for tangent points
        r = 1.2 * 1.3 # Effective radius after scaling
        d = np.linalg.norm(p_pos - sphere_center_pos)
        
        # Angle from center to point P
        vec_cp = p_pos - sphere_center_pos
        angle_cp = np.arctan2(vec_cp[1], vec_cp[0])
        
        # Tangent angle: alpha is the angle between CP and CA (or CB)
        # cos(alpha) = r / d
        alpha = np.arccos(r / d)
        
        # Tangent points A and B
        pos_a = sphere_center_pos + r * np.array([np.cos(angle_cp + alpha), np.sin(angle_cp + alpha), 0])
        pos_b = sphere_center_pos + r * np.array([np.cos(angle_cp - alpha), np.sin(angle_cp - alpha), 0])
        
        dot_a = Dot(pos_a, color=tangent_point_color)
        dot_b = Dot(pos_b, color=tangent_point_color)
        label_a = MathTex("A", color=tangent_point_color).next_to(dot_a, UP + LEFT, buff=0.1)
        label_b = MathTex("B", color=tangent_point_color).next_to(dot_b, DOWN + LEFT, buff=0.1)
        
        line_pa = Line(p_pos, pos_a, color=point_color)
        line_pb = Line(p_pos, pos_b, color=point_color)
        
        self.play(
            Create(line_pa), 
            Create(line_pb),
            FadeIn(dot_a),
            FadeIn(dot_b),
            Write(label_a),
            Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        # Highlight PA and PB segments (yellow)
        highlight_pa = Line(p_pos, pos_a, color=highlight_color, stroke_width=6)
        highlight_pb = Line(p_pos, pos_b, color=highlight_color, stroke_width=6)
        
        # Labels for the segments PA and PB
        label_pa_segment = MathTex("PA", color=highlight_color).scale(0.8)
        label_pa_segment.move_to(line_pa.get_center() + UP * 0.3 + LEFT * 0.1)
        
        label_pb_segment = MathTex("PB", color=highlight_color).scale(0.8)
        label_pb_segment.move_to(line_pb.get_center() + DOWN * 0.3 + LEFT * 0.1)
        
        # Equality text
        equality = MathTex("PA = PB", color=highlight_color)
        # Position and scale using place_in_area as per Issue 25
        self.place_in_area(equality, "E3", "E4", scale_factor=1.2)

        self.play(
            Create(highlight_pa),
            Create(highlight_pb),
            Write(label_pa_segment),
            Write(label_pb_segment)
        )
        self.play(Write(equality))
        self.wait(2)
        
        # Final cleanup - reset colors
        self.lecture[2].set_color(WHITE)
        self.wait(2)
