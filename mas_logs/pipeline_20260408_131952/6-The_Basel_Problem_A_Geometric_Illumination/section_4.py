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
        title = "The Geometric Construction: Circles and Squares"
        lines = [
            "We arrange our lights around a circular path.",
            "Repeatedly applying our theorem merges pairs of sources.",
            "The infinite line transforms into a finite circle.",
            "For small angles, chord length nearly matches the arc.",
            "This geometry links the series to the circle's properties."
        ]
        self.setup_layout(title, lines)

        # Asset path (Placeholder for user SVG)
        # Using a Dot representation if path is inaccessible, but here providing code structure
        light_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/lights.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Define the circle and the observer (sensor)
        circle_center = self.grid["D4"]
        circle = Circle(radius=1.2, color=BLUE_B).move_to(circle_center)
        
        # Observer at bottom (E4)
        sensor = Dot(color=WHITE)
        self.place_at_grid(sensor, "E4", scale_factor=1.0)
        sensor_label = Text("Observer", font_size=16).next_to(sensor, DOWN, buff=0.1)

        # Two light sources on the circle
        light1 = Dot(color=YELLOW).move_to(circle.point_at_angle(PI/3 + PI/2))
        light2 = Dot(color=YELLOW).move_to(circle.point_at_angle(-PI/3 + PI/2))

        self.play(Create(circle), FadeIn(sensor), Write(sensor_label))
        self.play(FadeIn(light1), FadeIn(light2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        # Show lines (hypotenuses) to the sensor
        line1 = Line(light1.get_center(), sensor.get_center(), color=YELLOW_A, stroke_width=2)
        line2 = Line(light2.get_center(), sensor.get_center(), color=YELLOW_A, stroke_width=2)
        
        # Inverse Pythagorean label - Using Text for robustness
        theorem_label = Text("1/h² = 1/a² + 1/b²", font_size=18, color=ORANGE)
        self.place_at_grid(theorem_label, "C4")

        self.play(Create(line1), Create(line2))
        self.play(Write(theorem_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(TEAL)
        # Repeatedly bisect arcs - visualize by adding more dots on the circle
        num_lights = 12
        extra_lights = VGroup(*[
            Dot(radius=0.06, color=YELLOW).move_to(circle.point_at_angle(angle))
            for angle in np.linspace(0, 2*PI, num_lights, endpoint=False)
        ])
        
        self.play(FadeOut(light1), FadeOut(light2), FadeOut(line1), FadeOut(line2), FadeOut(theorem_label))
        self.play(FadeIn(extra_lights))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(PINK)
        # Highlight a small segment
        arc_segment = Arc(radius=1.2, start_angle=PI/2, angle=PI/8, color=PINK).move_to(circle_center)
        chord_segment = Line(
            circle.point_at_angle(PI/2), 
            circle.point_at_angle(PI/2 + PI/8), 
            color=WHITE
        )
        
        self.play(Create(arc_segment), Create(chord_segment))
        self.play(Indicate(arc_segment), Indicate(chord_segment))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GOLD)
        final_formula = Text("Sum = 1 / sin²(θ)", font_size=24, color=GOLD)
        self.place_at_grid(final_formula, "B4")
        
        self.play(Write(final_formula))
        self.play(circle.animate.set_stroke(width=6, color=GOLD))
        self.wait(2)

        # Final cleanup for smooth ending
        self.play(FadeOut(extra_lights), FadeOut(sensor), FadeOut(sensor_label), FadeOut(arc_segment), FadeOut(chord_segment))
        self.wait(1)

# To run this code: manim -pql file_name.py Section4Scene
