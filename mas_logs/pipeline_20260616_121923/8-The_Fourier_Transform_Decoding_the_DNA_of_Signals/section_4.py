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
            "The formula captures this winding process mathematically.",
            "Euler's formula rotates our signal around the complex plane.",
            "Integrating over time finds the average center of mass.",
            "Peaks appear on the spectrum at component frequencies.",
            "This transforms time data into a frequency map."
        ]
        self.setup_layout("Connecting to the Math", lecture_lines)

        # Formula components (using Text due to latex system constraints)
        # X(f) = Integral x(t) e^(-i 2pi ft) dt
        f_part1 = Text("X(f) = ", font_size=32, color=WHITE)
        f_part2 = Text("∫", font_size=40, color=WHITE)
        f_part3 = Text(" x(t)", font_size=32, color=WHITE)
        f_part4 = Text(" e", font_size=32, color=WHITE)
        f_part4_exp = Text("-i 2π ft", font_size=20, color=WHITE).shift(UP*0.2 + RIGHT*0.5)
        f_part5 = Text(" dt", font_size=32, color=WHITE)
        
        formula = VGroup(f_part1, f_part2, f_part3, f_part4, f_part4_exp, f_part5).arrange(RIGHT, buff=0.1)
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Euler's formula rotates... Highlight exp in blue.
        # Also highlighting signal x(t) in red as per description.
        signal_label = Text("Signal", font_size=20, color="#FF0000")
        winder_label = Text("Winder", font_size=20, color="#00AAFF")
        
        # Position labels within 1 grid unit of their objects
        signal_label.next_to(f_part3, DOWN, buff=0.2)
        winder_label.next_to(f_part4, DOWN, buff=0.2)

        self.play(self.lecture[1].animate.set_color("#00AAFF"))
        self.play(
            f_part4.animate.set_color("#00AAFF"),
            f_part4_exp.animate.set_color("#00AAFF"),
            f_part3.animate.set_color("#FF0000"),
            FadeIn(signal_label),
            FadeIn(winder_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Integrating over time... Highlight integral and dt in green.
        com_label = Text("Center of Mass", font_size=20, color="#00FF00")
        com_label.next_to(f_part2, UP, buff=0.2)

        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(
            f_part2.animate.set_color("#00FF00"),
            f_part5.animate.set_color("#00FF00"),
            FadeIn(com_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show sharp yellow peak on a frequency graph
        # Resolving Issue 33 and 34
        
        time_axes = Axes(
            x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1],
            x_length=4, y_length=1.5,
            axis_config={"include_tip": False, "color": BLUE_E}
        )
        self.place_in_area(time_axes, 'A2', 'C5', scale_factor=0.6) # Issue 33 Fix: scale 0.6
        
        freq_axes = Axes(
            x_range=[0, 5, 1], y_range=[0, 1.5, 1],
            x_length=4, y_length=1.5,
            axis_config={"color": WHITE}
        )
        self.place_in_area(freq_axes, 'D2', 'F5', scale_factor=0.6) # Issue 34 Fix: scale 0.6
        
        time_signal = time_axes.plot(lambda x: np.sin(2*PI*x), color=RED)
        freq_peak = Line(
            freq_axes.c2p(1, 0), freq_axes.c2p(1, 1.2), 
            color="#FFFF00", stroke_width=4
        )
        peak_label = Text("Frequency Peak", font_size=18, color="#FFFF00")
        peak_label.next_to(freq_peak, UP, buff=0.1)

        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(
            FadeOut(formula), FadeOut(signal_label), FadeOut(winder_label), FadeOut(com_label),
            Create(time_axes), Create(freq_axes)
        )
        self.play(Create(time_signal), Create(freq_peak), Write(peak_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        # Arrow indicating the transformation process
        arrow = Arrow(time_axes.get_bottom(), freq_axes.get_top(), color=WHITE, buff=0.2)
        self.play(GrowArrow(arrow))
        self.wait(2)
