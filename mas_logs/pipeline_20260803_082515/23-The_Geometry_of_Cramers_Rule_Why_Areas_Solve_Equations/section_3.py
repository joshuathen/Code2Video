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
        # Data for vectors
        v1_color = "#FF00FF"
        v2_color = "#00FFFF"
        b_color = "#00FF00"
        
        v1_coords = np.array([2, 1, 0])
        v2_coords = np.array([1, 3, 0])
        b_coords = np.array([7, 11, 0])
        
        x_scale = 2
        y_scale = 3

        lecture_lines = [
            "We scale vectors to reach our target point.",
            "Finding x and y means finding these scales.",
            "It's like navigating a custom, skewed grid."
        ]
        self.setup_layout("The Linear Combination Visual", lecture_lines)

        # Coordinate Plane Setup
        # Fix for Issue 37 & 39: Position in A2-F6 with scale 0.75 to avoid cutoffs and overlap
        plane = NumberPlane(
            x_range=[-1, 9, 2],
            y_range=[-1, 13, 2],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": True, "font_size": 16}
        )
        self.place_in_area(plane, 'A2', 'F6', scale_factor=0.75)
        
        # Origin position for vectors
        origin = plane.coords_to_point(0, 0)
        
        # Vectors and Labels
        v1 = Vector(plane.coords_to_point(*v1_coords) - origin, color=v1_color)
        v1.shift(origin)
        v1_label = MathTex("v_1", color=v1_color, font_size=24).next_to(v1.get_end(), RIGHT, buff=0.1)
        
        v2 = Vector(plane.coords_to_point(*v2_coords) - origin, color=v2_color)
        v2.shift(origin)
        v2_label = MathTex("v_2", color=v2_color, font_size=24).next_to(v2.get_end(), LEFT, buff=0.1)

        b_vec = Vector(plane.coords_to_point(*b_coords) - origin, color=b_color)
        b_vec.shift(origin)
        # b_label to be grid-positioned later (Issue 38)
        b_label = MathTex("b", color=b_color, font_size=24)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(v1), Write(v1_label))
        self.play(GrowArrow(v2), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Show target vector b
        self.play(GrowArrow(b_vec))
        # Fix for Issue 38: Use place_at_grid for b_label to avoid overlap
        self.place_at_grid(b_label, 'A5', scale_factor=0.7)
        self.play(Write(b_label))
        self.wait(0.5)
        
        # Scaling vectors v1 and v2
        scaled_v1_coords = x_scale * v1_coords
        scaled_v2_coords = y_scale * v2_coords
        
        v1_scaled = Vector(plane.coords_to_point(*scaled_v1_coords) - origin, color=v1_color)
        v1_scaled.shift(origin)
        v1_scaled_label = MathTex(f"{x_scale}v_1", color=v1_color, font_size=24).next_to(v1_scaled.get_end(), DOWN, buff=0.1)
        
        v2_scaled = Vector(plane.coords_to_point(*scaled_v2_coords) - origin, color=v2_color)
        v2_scaled.shift(origin)
        # v2_scaled_label to be grid-positioned later (Issue 38)
        v2_scaled_label = MathTex(f"{y_scale}v_2", color=v2_color, font_size=24)
        
        # Animate Scaling v1
        self.play(
            Transform(v1, v1_scaled),
            Transform(v1_label, v1_scaled_label),
            run_time=1.5
        )
        
        # Animate Scaling v2 and position label via grid (Issue 38)
        self.place_at_grid(v2_scaled_label, 'B5', scale_factor=0.7)
        self.play(
            Transform(v2, v2_scaled),
            Transform(v2_label, v2_scaled_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Tip-to-tail positioning
        # We move the scaled v2 to the tip of scaled v1
        tip_v1 = v1.get_end()
        # Note: we don't move the label here as the issue fix uses a fixed grid position (B5) for it.
        self.play(
            v2.animate.shift(tip_v1 - origin),
            run_time=2
        )
        
        # Skewed grid background (Clipped to plane bounds to fix Issue 37)
        grid_lines = VGroup()
        x_min, x_max = -1, 9
        y_min, y_max = -1, 13
        
        # Lines parallel to v2: P(t) = i*v1 + t*v2
        for i in range(-5, 10):
            t_min_x = x_min - 2*i
            t_max_x = x_max - 2*i
            t_min_y = (y_min - i) / 3
            t_max_y = (y_max - i) / 3
            t_start, t_end = max(t_min_x, t_min_y), min(t_max_x, t_max_y)
            if t_start < t_end:
                p_start = plane.coords_to_point(2*i + t_start, i + 3*t_start)
                p_end = plane.coords_to_point(2*i + t_end, i + 3*t_end)
                grid_lines.add(Line(p_start, p_end, color=v2_color, stroke_width=1, stroke_opacity=0.2))

        # Lines parallel to v1: P(t) = j*v2 + t*v1
        for j in range(-5, 10):
            t_min_x = (x_min - j) / 2
            t_max_x = (x_max - j) / 2
            t_min_y = y_min - 3*j
            t_max_y = y_max - 3*j
            t_start, t_end = max(t_min_x, t_min_y), min(t_max_x, t_max_y)
            if t_start < t_end:
                p_start = plane.coords_to_point(j + 2*t_start, 3*j + t_start)
                p_end = plane.coords_to_point(j + 2*t_end, 3*j + t_end)
                grid_lines.add(Line(p_start, p_end, color=v1_color, stroke_width=1, stroke_opacity=0.2))
        
        self.play(Create(grid_lines), run_time=2)
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
