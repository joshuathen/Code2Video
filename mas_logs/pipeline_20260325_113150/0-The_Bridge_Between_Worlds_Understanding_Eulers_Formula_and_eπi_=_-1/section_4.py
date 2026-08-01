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
        # Initial Setup
        self.setup_layout("Visualizing the Formula: e^(i\u03b8) = cos\u03b8 + i sin\u03b8", [
            "Euler's formula describes any point on the unit circle.",
            "A vector at angle theta marks our current position.",
            "Cosine gives the horizontal distance on the real axis.",
            "Sine gives the vertical distance on the imaginary axis.",
            "Increasing theta moves the point smoothly around the circle."
        ])

        # Main Formula
        # Issue 36: Scale factor 1.2 in area A2-A5
        formula = Text("e^(i\u03b8) = cos \u03b8 + i sin \u03b8", font_size=32, color=WHITE)
        self.place_in_area(formula, "A2", "A5", scale_factor=1.2)

        # Unit Circle and Axes
        # Issue 34: Use grid D4
        radius_val = 1.3
        unit_circle = Circle(radius=radius_val, color="#444444")
        self.place_at_grid(unit_circle, "D4", scale_factor=1.0)
        circle_center = unit_circle.get_center()

        axes = VGroup(
            Line(circle_center + LEFT * (radius_val + 0.3), circle_center + RIGHT * (radius_val + 0.3), color=GRAY, stroke_width=1),
            Line(circle_center + DOWN * (radius_val + 0.3), circle_center + UP * (radius_val + 0.3), color=GRAY, stroke_width=1)
        )

        # Labels
        # Issue 35: Real label at D6, scale 0.6
        real_label = Text("Real", font_size=18, color=GRAY)
        self.place_at_grid(real_label, "D6", scale_factor=0.6)
        
        imag_label = Text("Imaginary", font_size=18, color=GRAY)
        self.place_at_grid(imag_label, "B4", scale_factor=0.6)

        # Animation Tracker
        theta_tracker = ValueTracker(PI / 4)

        # Dynamic components
        def get_point():
            val = theta_tracker.get_value()
            return circle_center + radius_val * np.array([np.cos(val), np.sin(val), 0])

        def get_cos_point():
            val = theta_tracker.get_value()
            return circle_center + np.array([radius_val * np.cos(val), 0, 0])

        radius_vector = always_redraw(lambda: Line(circle_center, get_point(), color=WHITE, buff=0))
        
        angle_arc = always_redraw(lambda: Arc(
            radius=0.4, 
            start_angle=0, 
            angle=theta_tracker.get_value(), 
            arc_center=circle_center, 
            color=YELLOW
        ))
        
        theta_label = always_redraw(lambda: Text("\u03b8", font_size=20, color=YELLOW).move_to(
            circle_center + 0.55 * np.array([np.cos(theta_tracker.get_value()/2), np.sin(theta_tracker.get_value()/2), 0])
        ))

        cos_line = always_redraw(lambda: Line(
            circle_center, get_cos_point(), color="#00FF00", stroke_width=6
        ))
        
        sin_line = always_redraw(lambda: Line(
            get_cos_point(), get_point(), color="#0000FF", stroke_width=6
        ))

        cos_text = always_redraw(lambda: Text("cos \u03b8", font_size=20, color="#00FF00").next_to(cos_line, DOWN, buff=0.1))
        sin_text = always_redraw(lambda: Text("i sin \u03b8", font_size=20, color="#0000FF").next_to(sin_line, RIGHT, buff=0.1))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Write(formula))
        self.play(Create(unit_circle), Create(axes), Write(real_label), Write(imag_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(radius_vector), Create(angle_arc), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        self.play(Create(cos_line), Write(cos_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#0000FF")
        self.play(Create(sin_line), Write(sin_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(theta_tracker.animate.set_value(0), run_time=1)
        self.play(theta_tracker.animate.set_value(2 * PI), run_time=6, rate_func=linear)
        self.wait(2)
