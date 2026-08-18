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

class Section6Scene(TeachingScene):
    def construct(self):
        # Section 6: Summary and Practical Application
        lecture_lines = [
            "Derivatives break wholes down; integrals build them back up.",
            "Think of a speedometer versus a car's odometer.",
            "Together, they map the motion of our entire universe."
        ]
        self.setup_layout("Summary and Practical Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Screen splits into 'Derivative' (#FF4500) and 'Integral' (#7CFC00).
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        
        deriv_label = Text("Derivative", color="#FF4500", font_size=32)
        integ_label = Text("Integral", color="#7CFC00", font_size=32)
        
        self.place_in_area(deriv_label, "A1", "A3")
        self.place_in_area(integ_label, "A4", "A6")
        
        self.play(Write(deriv_label), Write(integ_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Left side shows a red needle pulsing like a speedometer (#FF4500).
        # Right side shows a white digital counter like an odometer.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#7CFC00")
        )
        
        # Speedometer components
        speedo_arc = Arc(radius=1.0, start_angle=PI, angle=-PI, color=WHITE)
        speedo_center = Dot(color=WHITE)
        speedo_base = VGroup(speedo_arc, speedo_center)
        self.place_in_area(speedo_base, "B1", "E3")
        
        speedo_center_pos = speedo_base.get_center()
        rotating_needle = Line(speedo_center_pos, speedo_center_pos + LEFT * 0.9, color="#FF4500", stroke_width=6)
        
        # ValueTracker for needle angle (0 to PI)
        angle_tracker = ValueTracker(0)
        def update_needle(m):
            new_angle = PI - angle_tracker.get_value()
            vec = np.array([np.cos(new_angle), np.sin(new_angle), 0]) * 0.9
            m.put_start_and_end_on(speedo_center_pos, speedo_center_pos + vec)
            
        rotating_needle.add_updater(update_needle)

        # Odometer components
        odo_box = Rectangle(width=2.5, height=0.8, color=WHITE)
        odo_val = DecimalNumber(42.0, num_decimal_places=1, include_sign=False, font_size=36)
        odometer = VGroup(odo_box, odo_val)
        self.place_in_area(odometer, "B4", "E6")
        
        self.play(Create(speedo_base), Create(rotating_needle), Create(odo_box), Write(odo_val))
        
        # Animate speedometer and odometer
        self.play(
            angle_tracker.animate.set_value(PI/2),
            odo_val.animate.set_value(42.5),
            run_time=1.5,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.play(
            angle_tracker.animate.set_value(PI/4),
            odo_val.animate.set_value(43.1),
            run_time=1.5,
            rate_func=linear
        )
        self.play(
            angle_tracker.animate.set_value(PI*0.8),
            odo_val.animate.set_value(45.0),
            run_time=1.5,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Both icons merge into a central glowing 'Calculus' icon (#FFFFFF).
        # Final text 'The Language of Change' appears in cyan (#00FFFF).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        calc_icon_circle = Circle(radius=1.2, color=WHITE, stroke_width=4)
        calc_icon_symbol = VGroup(
            MathTex(r"\int", color="#7CFC00", font_size=70),
            MathTex(r"\frac{d}{dx}", color="#FF4500", font_size=70)
        ).arrange(RIGHT, buff=0.3)
        calc_icon = VGroup(calc_icon_circle, calc_icon_symbol)
        self.place_in_area(calc_icon, "B1", "E6")
        
        final_text = Text("The Language of Change", color="#00FFFF", font_size=32)
        self.place_in_area(final_text, "F1", "F6")
        
        # Remove updaters before transform
        rotating_needle.clear_updaters()
        
        self.play(
            FadeOut(deriv_label),
            FadeOut(integ_label),
            ReplacementTransform(VGroup(speedo_base, rotating_needle, odometer), calc_icon),
            run_time=2
        )
        
        # Glowing effect
        self.play(calc_icon.animate.scale(1.2), run_time=0.8, rate_func=there_and_back)
        
        self.play(Write(final_text))
        self.wait(3)
