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
        # Setup the title and lecture lines
        title_text = "Prerequisite: The 2D Playground (Complex Plane)"
        lecture_lines = [
            "Complex numbers live on a 2D plane.",
            "Squaring a number stretches its distance from zero.",
            "This operation also rotates the point around the origin."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # "Complex numbers live on a 2D plane."
        self.lecture[0].set_color("#FFFFFF")
        
        # Create plane
        plane = ComplexPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": "#444444",
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        ).add_coordinates()
        
        # Axis labels
        labels = plane.get_axis_labels(x_label="Re", y_label="Im")
        labels.set_color("#FFFFFF")
        
        plane_group = VGroup(plane, labels)
        # Issue 24 Fix: Scale factor 0.7 to avoid 'Re' label being cramped.
        self.place_in_area(plane_group, 'A2', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(plane_group))
        self.wait(2.0)
        
        # === Animation for Lecture Line 2 ===
        # "Squaring a number stretches its distance from zero."
        self.lecture[0].set_color("#444444")
        self.lecture[1].set_color("#00FFFF")
        
        origin = plane.coords_to_point(0, 0)
        start_point = plane.coords_to_point(1, 1)
        
        vector = Arrow(origin, start_point, buff=0, color="#00FFFF", stroke_width=4)
        
        # Issue 25 Fix: Label '(1, 1)' too large, use place_at_grid at C5 with scale 0.5.
        vec_label = MathTex("(1, 1)", color="#00FFFF")
        self.place_at_grid(vec_label, 'C5', scale_factor=0.5)
        
        self.play(Create(vector), FadeIn(vec_label))
        self.wait(1.5)
        
        # Squaring z=1+i gives z^2=2i. Magnitude sqrt(2) -> 2.
        end_point = plane.coords_to_point(0, 2)
        new_vector = Arrow(origin, end_point, buff=0, color="#00FFFF", stroke_width=4)
        
        # Issue 23 Fix: Label '(0, 2i)' overlaps with 'Im', use place_at_grid at B5 with scale 0.5.
        new_vec_label = MathTex("(0, 2i)", color="#00FFFF")
        self.place_at_grid(new_vec_label, 'B5', scale_factor=0.5)
        
        self.play(
            Transform(vector, new_vector),
            Transform(vec_label, new_vec_label),
            run_time=2
        )
        self.wait(2.0)
        
        # === Animation for Lecture Line 3 ===
        # "This operation also rotates the point around the origin."
        self.lecture[1].set_color("#444444")
        self.lecture[2].set_color("#FF00FF")
        
        # Show multiple points moving in curved paths (Magenta and Yellow)
        p1 = Dot(plane.coords_to_point(1.2, 0), color="#FF00FF")
        p2 = Dot(plane.coords_to_point(0.7, 0.7), color="#FFFF00")
        
        p1_target = plane.coords_to_point(1.44, 0) # 1.2^2 = 1.44
        p2_target = plane.coords_to_point(0, 0.98) # (0.7+0.7i)^2 = 0.98i
        
        self.play(FadeIn(p1), FadeIn(p2))
        self.wait(1.0)
        
        # Points move in unique curved paths
        self.play(
            p1.animate.move_to(p1_target),
            p2.animate.move_to(p2_target),
            run_time=3,
            path_arc=np.pi/4
        )
        
        self.wait(2.0)
