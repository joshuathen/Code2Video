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
        lecture_lines = [
            "The Fourier formula provides a mathematical recipe for waves.",
            "The constant term represents the wave's average height.",
            "Coefficients a-n and b-n act like volume sliders.",
            "They determine the strength of each cosine and sine.",
            "This blueprint reconstructs any periodic function precisely."
        ]
        self.setup_layout("The Mathematical Blueprint", lecture_lines)
        
        # Trackers for the formula parameters
        a0_val = ValueTracker(0.0)
        a1_val = ValueTracker(0.0)
        b3_val = ValueTracker(0.5)

        # === Animation for Lecture Line 1 ===
        # The Fourier formula provides a mathematical recipe for waves.
        # f(x) = a0/2 + sum [an cos(nx) + bn sin(nx)]
        formula = MathTex(
            "f(x) = ", "\\frac{a_0}{2}", " + \\sum_{n=1}^{\\infty} [", "a_n", "\\cos(nx) + ", "b_n", "\\sin(nx)]",
            font_size=36, color=WHITE
        )
        # Fix for Issue 33: Line 69: self.place_in_area(formula, 'A2', 'A6')
        self.place_in_area(formula, 'A2', 'A6')
        
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The constant term represents the wave's average height.
        # Highlight a0/2 in Gold (#FFD700)
        self.play(
            formula[1].animate.set_color("#FFD700"),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Setup Axes for visualization
        axes = Axes(
            x_range=[0, 2*PI, PI], y_range=[-2, 2, 1], 
            axis_config={"include_tip": False},
            x_length=4.5, y_length=2.5
        )
        # Fix for Issue 31: Line 90: self.place_in_area(axes, 'B1', 'D6')
        self.place_in_area(axes, 'B1', 'D6')
        
        # a0 offset visualization: a horizontal line representing average height
        # Using a0_val for interactive height change
        a0_line = always_redraw(lambda: axes.plot(lambda x: a0_val.get_value(), color="#FFD700"))
        a0_label = MathTex("a_0/2", color="#FFD700", font_size=20)
        # Position label relative to the line
        a0_label.add_updater(lambda m: m.move_to(axes.c2p(0.5, a0_val.get_value() + 0.3)))
        
        self.play(Create(axes), Create(a0_line), FadeIn(a0_label))
        # Animate DC offset changing to demonstrate "average height"
        self.play(a0_val.animate.set_value(1.0), run_time=1)
        self.play(a0_val.animate.set_value(-0.8), run_time=1)
        self.play(a0_val.animate.set_value(0.0), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Coefficients a-n and b-n act like volume sliders.
        # transform formula terms into green (#00FF00) sliders
        
        def create_vertical_slider(tracker, label_text, color="#00FF00"):
            track = Line(DOWN, UP, color=GREY_B).scale(0.8)
            knob = Dot(color=color)
            # Update knob position along the track based on tracker value
            knob.add_updater(lambda m: m.move_to(track.point_from_proportion(np.clip((tracker.get_value() + 1.5) / 3.0, 0, 1))))
            label = MathTex(label_text, font_size=24, color=color)
            slider = VGroup(track, knob, label).arrange(DOWN, buff=0.15)
            return slider

        # Slider for a1 and b3 as examples of an and bn
        s_a1 = create_vertical_slider(a1_val, "a_1")
        s_b3 = create_vertical_slider(b3_val, "b_3")
        sliders = VGroup(s_a1, s_b3).arrange(RIGHT, buff=1.0)
        # Fix for Issue 32: Line 123: self.place_in_area(sliders, 'E1', 'F6')
        self.place_in_area(sliders, 'E1', 'F6')

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            formula[3].animate.set_color("#00FF00"),
            formula[5].animate.set_color("#00FF00"),
            FadeOut(a0_line), FadeOut(a0_label),
            FadeIn(sliders)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # They determine the strength of each cosine and sine.
        # One slider moves up, increasing magnitude of sin(3x) component.
        
        # Component wave b3 * sin(3x)
        comp_wave = always_redraw(lambda: axes.plot(
            lambda x: b3_val.get_value() * np.sin(3*x), 
            color="#87CEEB"
        ))
        comp_label = MathTex("b_3 \\sin(3x)", color="#87CEEB", font_size=24)
        comp_label.add_updater(lambda m: m.move_to(axes.c2p(PI, 1.6)))

        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW),
            Create(comp_wave), FadeIn(comp_label)
        )
        
        # Animate b3 slider movement and corresponding wave magnitude increase
        self.play(b3_val.animate.set_value(1.5), run_time=1.5)
        self.play(b3_val.animate.set_value(-1.2), run_time=1.5)
        self.play(b3_val.animate.set_value(0.6), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This blueprint reconstructs any periodic function precisely.
        # Total sum wave adjusts its shape according to slider movement.
        
        # The sum wave: a0 + a1*cos(x) + b3*sin(3x)
        sum_wave = always_redraw(lambda: axes.plot(
            lambda x: a0_val.get_value() + a1_val.get_value() * np.cos(x) + b3_val.get_value() * np.sin(3*x),
            color=WHITE
        ))
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW),
            FadeOut(comp_wave), FadeOut(comp_label),
            FadeIn(sum_wave)
        )
        
        # Add a0 slider back to show full control
        s_a0 = create_vertical_slider(a0_val, "a_0/2", color="#FFD700")
        # Align s_a0 with the existing sliders
        s_a0.move_to(sliders.get_left() + LEFT * 1.5)
        self.play(FadeIn(s_a0))
        
        # Animate multiple sliders to show dynamic reconstruction
        self.play(
            a0_val.animate.set_value(0.5),
            a1_val.animate.set_value(0.8),
            b3_val.animate.set_value(-0.5),
            run_time=2.5
        )
        self.play(
            a0_val.animate.set_value(-0.3),
            a1_val.animate.set_value(-0.5),
            b3_val.animate.set_value(1.2),
            run_time=2.5
        )
        self.wait(2)
