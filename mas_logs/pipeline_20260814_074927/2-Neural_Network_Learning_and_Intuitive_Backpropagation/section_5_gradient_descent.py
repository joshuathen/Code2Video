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

class Section5GradientDescentScene(TeachingScene):
    def construct(self):
        title = "The Adjustment: Nudging the Knobs"
        lines = [
            "We nudge the knobs based on the blame.",
            "Each weight is adjusted by a small step.",
            "The learning rate controls how far we move.",
            "Small steps prevent overshooting the bottom.",
            "Moving down the slope reduces the total error."
        ]
        self.setup_layout(title, lines)

        # --- Object Initialization ---
        
        # Knobs: Floppy Ear and Metal
        knob_floppy_circle = Circle(radius=0.4, color=WHITE)
        knob_floppy_indicator = Line(ORIGIN, 0.4 * UP, color=WHITE)
        knob_floppy = VGroup(knob_floppy_circle, knob_floppy_indicator)
        self.place_at_grid(knob_floppy, 'B2')
        knob_floppy_label = Text("Floppy Ear", font_size=16).next_to(knob_floppy, UP, buff=0.2)
        
        knob_metal_circle = Circle(radius=0.4, color=WHITE)
        knob_metal_indicator = Line(ORIGIN, 0.4 * UP, color=WHITE)
        knob_metal = VGroup(knob_metal_circle, knob_metal_indicator)
        self.place_at_grid(knob_metal, 'B5')
        knob_metal_label = Text("Metal", font_size=16).next_to(knob_metal, UP, buff=0.2)

        # Gradient Arrows (Yellow)
        arrow_floppy = Arrow(
            start=self.grid['B2'], 
            end=self.grid['B2'] + 0.6 * RIGHT, 
            color="#FFFF00", 
            buff=0, 
            stroke_width=4,
            max_tip_length_to_length_ratio=0.3
        )
        arrow_metal = Arrow(
            start=self.grid['B5'], 
            end=self.grid['B5'] + 0.6 * LEFT, 
            color="#FFFF00", 
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.3
        )

        # Learning Rate Slider
        slider_line = Line(self.grid['D2'], self.grid['D5'], color=WHITE)
        slider_dot = Dot(color="#00FF00").move_to(self.grid['D4']) 
        slider_label = Text("Learning Rate", font_size=16, color="#00FF00").next_to(slider_line, UP, buff=0.2)
        slider = VGroup(slider_line, slider_dot, slider_label)

        # Landscape Curve
        landscape_center = self.grid['F3'] + 0.5 * RIGHT
        landscape_curve = ParametricFunction(
            lambda t: np.array([t, 0.4 * (t - 0)**2 - 0.5, 0]),
            t_range=[-1.5, 1.5],
            color=BLUE
        ).move_to(landscape_center)
        
        # Pixel character
        pixel = Dot(color="#00FF00", radius=0.1)
        pixel.move_to(landscape_curve.point_from_proportion(0.1))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(
            FadeIn(knob_floppy), FadeIn(knob_floppy_label),
            FadeIn(knob_metal), FadeIn(knob_metal_label)
        )
        self.play(GrowArrow(arrow_floppy), GrowArrow(arrow_metal))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(
            arrow_floppy.animate.scale(0.8),
            arrow_metal.animate.scale(0.8)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(slider))
        self.play(
            slider_dot.animate.move_to(self.grid['D2']),
            arrow_floppy.animate.scale(0.4),
            arrow_metal.animate.scale(0.4),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(Create(landscape_curve), FadeIn(pixel))
        self.play(
            knob_floppy_indicator.animate.set_color("#00FF00"),
            knob_metal_indicator.animate.set_color("#FF0000"),
            Rotate(knob_floppy, angle=-PI/3, about_point=self.grid['B2']),
            Rotate(knob_metal, angle=PI/3, about_point=self.grid['B5']),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.play(
            MoveAlongPath(pixel, landscape_curve, rate_func=lambda t: 0.1 + t * 0.38),
            run_time=2.5,
            rate_func=linear
        )
        self.wait(2)
