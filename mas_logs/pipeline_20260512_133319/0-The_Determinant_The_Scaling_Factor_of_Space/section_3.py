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
        lecture_lines = [
            'The unit square represents an area of exactly one.',
            'Transformation turns this square into a unique parallelogram.',
            'This new area represents the absolute determinant value.',
            'Any region scales by this same constant factor.',
            "Geometrically, the determinant is the area's scaling factor."
        ]
        self.setup_layout("The Geometric Definition (2D)", lecture_lines)
        
        # Grid Coordinates Reference
        # Origin will be 'D3'
        origin = self.grid['D3']
        unit_x = self.grid['D4'] - origin
        unit_y = self.grid['C3'] - origin

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Simple axes for orientation
        x_axis = Line(origin - unit_x, origin + 3.5*unit_x, color=GRAY, stroke_width=1)
        y_axis = Line(origin - unit_y, origin + 2.5*unit_y, color=GRAY, stroke_width=1)
        
        # Vectors and unit square
        i_vec = Arrow(start=origin, end=origin + unit_x, buff=0, color="#FF4444")
        j_vec = Arrow(start=origin, end=origin + unit_y, buff=0, color="#44FF44")
        i_label = Text("i", font_size=20, color="#FF4444")
        j_label = Text("j", font_size=20, color="#44FF44")
        
        # Fix Issue 40: Moved j_label to C1 from C2 to avoid axis overlap
        self.place_at_grid(i_label, "E4", scale_factor=0.8)
        self.place_at_grid(j_label, "C1", scale_factor=0.8)
        
        unit_square = Polygon(
            origin, 
            origin + unit_x, 
            origin + unit_x + unit_y, 
            origin + unit_y, 
            color=WHITE, 
            stroke_width=2
        )
        
        self.add(x_axis, y_axis)
        self.play(Create(unit_square), GrowArrow(i_vec), GrowArrow(j_vec), FadeIn(i_label), FadeIn(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FFFF00") # Yellow
        
        # New basis vectors forming a slanted parallelogram
        new_i_pt = origin + 2*unit_x + 0.5*unit_y
        new_j_pt = origin + 0.5*unit_x + 1.5*unit_y
        new_sum_pt = origin + 2.5*unit_x + 2.0*unit_y
        
        parallelogram = Polygon(
            origin,
            new_i_pt,
            new_sum_pt,
            new_j_pt,
            color="#FFFF00",
            stroke_width=3
        )
        
        self.play(
            ReplacementTransform(unit_square, parallelogram),
            i_vec.animate.put_start_and_end_on(origin, new_i_pt),
            j_vec.animate.put_start_and_end_on(origin, new_j_pt),
            i_label.animate.move_to(self.grid["E5"]),
            j_label.animate.move_to(self.grid["B3"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE)
        
        # Fill the parallelogram with 50% opacity yellow
        self.play(parallelogram.animate.set_fill("#FFFF00", opacity=0.5))
        
        # Flash Text Area = |Determinant|
        area_text = Text("Area = |Determinant|", font_size=28, color="#FFFFFF", weight=BOLD)
        # Fix Issue 39: Positioned area_text in larger area A2-A5 for better centering
        self.place_in_area(area_text, "A2", "A5", scale_factor=0.8)
        self.play(Flash(area_text, color=WHITE))
        self.add(area_text)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(TEAL)
        
        # Clean up previous formula and reset vectors for a scaling example
        self.play(
            FadeOut(parallelogram),
            FadeOut(area_text),
            i_vec.animate.put_start_and_end_on(origin, origin + unit_x),
            j_vec.animate.put_start_and_end_on(origin, origin + unit_y),
            i_label.animate.move_to(self.grid["E4"]),
            j_label.animate.move_to(self.grid["C1"]), # Issue 40 persistence
            run_time=1
        )
        
        # Show scaling i-vector by 3 to demonstrate consistent scaling
        scaled_i_pt = origin + 3*unit_x
        scaling_rect = Polygon(
            origin,
            scaled_i_pt,
            scaled_i_pt + unit_y,
            origin + unit_y,
            color=TEAL,
            fill_color=TEAL,
            fill_opacity=0.4,
            stroke_width=2
        )
        
        self.play(
            i_vec.animate.put_start_and_end_on(origin, scaled_i_pt),
            i_label.animate.move_to(self.grid["E6"]),
            Create(scaling_rect),
            run_time=2
        )
        
        scale_label = Text("Area triples (3x)", font_size=22, color=TEAL)
        # Fix Issue 38: Use place_in_area A4-B6 to avoid cutoffs/overlaps
        self.place_in_area(scale_label, "A4", "B6", scale_factor=0.8)
        self.play(Write(scale_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color("#FFD700") # Gold
        
        # Show a circle being scaled to emphasize "Geometrically, the determinant is the scaling factor"
        small_circle = Circle(radius=0.2, color=WHITE, stroke_width=2)
        self.place_in_area(small_circle, 'C3', 'D4')
        
        # Scaled circle (ellipse) in the 3x1 rect
        scaled_ellipse = Ellipse(width=1.2, height=0.4, color="#FFD700", fill_opacity=0.3)
        self.place_in_area(scaled_ellipse, 'C3', 'D6')
        
        self.play(Create(small_circle))
        self.play(ReplacementTransform(small_circle, scaled_ellipse))
        self.wait(2)
