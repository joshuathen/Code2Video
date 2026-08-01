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
        # 1. Sync lecture lines with storyboard
        lecture_lines = [
            "Matrix exponentiation represents a continuous transformation.",
            "It maps an initial state to a future point.",
            "As time flows, the coordinate grid warps smoothly.",
            "The state follows a continuous trajectory through space.",
            "This flow is the heart of linear dynamics."
        ]
        
        self.setup_layout("Geometric Intuition: The Continuous Flow", lecture_lines)
        
        # Colors for highlights
        HIGHLIGHT_COLOR = "#FFFF00"
        VECTOR_COLOR = "#00FF00"
        FORMULA_COLOR = "#00FFFF"
        TRAJECTORY_COLOR = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": False}
        )
        self.place_in_area(plane, 'C2', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Initial vector x0
        x0_val = np.array([1, 0, 0])
        origin_point = plane.get_origin().copy()
        vector = Arrow(origin_point, plane.coords_to_point(x0_val[0], x0_val[1]), buff=0, color=VECTOR_COLOR)
        
        # Replacing MathTex with MarkupText to resolve FileNotFoundError: 'latex'
        vector_label = MarkupText("<b>x</b><sub>0</sub>", color=VECTOR_COLOR).scale(0.8)
        vector_label.next_to(vector.get_end(), RIGHT, buff=0.1)
        
        # Replacing MathTex with MarkupText to resolve FileNotFoundError: 'latex'
        matrix_formula = MarkupText("<b>x</b>(t) = e<sup>At</sup> <b>x</b><sub>0</sub>", color=FORMULA_COLOR)
        self.place_at_grid(matrix_formula, 'B2', scale_factor=0.9)
        
        self.play(Create(vector), Write(vector_label), Write(matrix_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        t_tracker = ValueTracker(0)
        
        def get_transform_matrix(t):
            scale = np.exp(0.1 * t)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            return scale * np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ])

        original_plane_points = plane.copy().points.copy()
        original_vector_points = vector.copy().points.copy()
        
        def update_plane(p):
            t = t_tracker.get_value()
            mat = get_transform_matrix(t)
            new_points = []
            for pt in original_plane_points:
                rel_pt = pt - origin_point
                trans_pt = np.dot(mat, rel_pt)
                new_points.append(trans_pt + origin_point)
            p.points = np.array(new_points)

        def update_vector(v):
            t = t_tracker.get_value()
            mat = get_transform_matrix(t)
            new_points = []
            for pt in original_vector_points:
                rel_pt = pt - origin_point
                trans_pt = np.dot(mat, rel_pt)
                new_points.append(trans_pt + origin_point)
            v.points = np.array(new_points)
            
        plane.add_updater(update_plane)
        vector.add_updater(update_vector)
        
        self.play(FadeOut(vector_label))
        self.play(t_tracker.animate.set_value(1.5), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        trajectory_label = Text("continuous trajectory", font_size=20, color=TRAJECTORY_COLOR)
        self.place_at_grid(trajectory_label, 'B5', scale_factor=0.7)
        
        def trajectory_func(t):
            mat = get_transform_matrix(t)
            rel_pos = np.dot(mat, x0_val)
            return origin_point + rel_pos

        path = ParametricFunction(trajectory_func, t_range=[0, 1.5], color=TRAJECTORY_COLOR)
        
        self.play(Create(path), FadeIn(trajectory_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        flow_text = Text("DYNAMIC FLOW", font_size=36, color=WHITE).set_stroke(BLUE, width=2)
        self.place_at_grid(flow_text, 'E4', scale_factor=1.0)
        
        self.play(Write(flow_text))
        self.play(Indicate(flow_text, color=BLUE))
        
        self.wait(3)
