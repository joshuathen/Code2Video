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
        # 1. Setup layout
        self.setup_layout("Euler's Formula: The GPS of the Complex Plane", [
            "Euler's formula defines our position on this circular path.",
            "e to the i x equals cosine x plus i sine x.",
            "Choosing an angle x places us on the unit circle.",
            "Cosine tracks horizontal position, while sine tracks the vertical.",
            "It is the ultimate GPS for the complex plane."
        ])

        # Colors
        white_c = "#FFFFFF"
        orange_c = "#FFA500"
        green_c = "#00FF00"
        magenta_c = "#FF00FF"
        gray_c = "#888888"

        # Reference center point for the unit circle
        center_pt = self.grid['D4']
        # The base circle and axes should be scaled to fit comfortably.
        # Issue 38: Circle scaled to 0.7
        # Issue 36: Horizontal axis scaled to 0.7
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(white_c))
        
        # Unit Circle
        circle = Circle(radius=1.5, color=white_c)
        self.place_at_grid(circle, 'D4', scale_factor=0.7)
        
        # Axes
        h_axis = Line(LEFT * 3.0, RIGHT * 3.0, color=gray_c, stroke_opacity=0.4)
        v_axis = Line(UP * 3.0, DOWN * 3.0, color=gray_c, stroke_opacity=0.4)
        self.place_at_grid(h_axis, 'D4', scale_factor=0.7)
        self.place_at_grid(v_axis, 'D4', scale_factor=0.7)
        
        self.play(Create(h_axis), Create(v_axis))
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Issue 37: Formula at B2-B5, scale 0.8
        self.play(self.lecture[1].animate.set_color(white_c))
        formula = Text("e^ix = cos(x) + i sin(x)", color=white_c, font_size=32)
        self.place_in_area(formula, 'B2', 'B5', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(white_c))
        
        # Setup angle tracker and dynamic geometry
        angle_tracker = ValueTracker(PI/4)
        radius_mag = 1.5 * 0.7  # radius * scale_factor

        def get_radial_pos():
            theta = angle_tracker.get_value()
            return center_pt + radius_mag * np.array([np.cos(theta), np.sin(theta), 0])

        radius_vector = always_redraw(lambda: Line(
            center_pt, get_radial_pos(), color=orange_c, stroke_width=4
        ))
        dot = always_redraw(lambda: Dot(get_radial_pos(), color=orange_c))
        
        # Simple angle arc
        arc = always_redraw(lambda: Arc(
            radius=0.4, 
            start_angle=0, 
            angle=angle_tracker.get_value(), 
            arc_center=center_pt, 
            color=orange_c
        ))
        
        angle_label = Text("x", font_size=18, color=orange_c)
        self.place_at_grid(angle_label, 'D5', scale_factor=0.8) # Position relative to center D4

        self.play(Create(radius_vector), Create(dot), Create(arc), Write(angle_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(white_c))
        
        # Horizontal component (Cosine)
        cos_line = always_redraw(lambda: DashedLine(
            [get_radial_pos()[0], center_pt[1], 0], get_radial_pos(), color=green_c
        ))
        cos_label = Text("cos(x)", font_size=18, color=green_c)
        self.place_at_grid(cos_label, 'E4', scale_factor=0.9)

        # Vertical component (Sine)
        sin_line = always_redraw(lambda: DashedLine(
            [center_pt[0], get_radial_pos()[1], 0], get_radial_pos(), color=magenta_c
        ))
        sin_label = Text("i sin(x)", font_size=18, color=magenta_c)
        self.place_at_grid(sin_label, 'C4', scale_factor=0.9)

        self.play(Create(cos_line), Create(sin_line))
        self.play(Write(cos_label), Write(sin_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(white_c))
        
        # Rotating the point like a GPS tracking movement
        self.play(
            angle_tracker.animate.set_value(2 * PI + PI / 4),
            run_time=6,
            rate_func=linear
        )
        self.wait(2)
