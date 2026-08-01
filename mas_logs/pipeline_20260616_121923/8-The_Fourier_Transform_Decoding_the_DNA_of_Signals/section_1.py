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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the Scene
        title = "The Big Idea: The Smoothie Metaphor"
        lines = [
            "Imagine a complex signal is like a fruit smoothie.",
            "It is a mixture of different ingredients blended together.",
            "The Fourier Transform acts like a magical filter.",
            "It pulls out individual ingredients, like strawberries or bananas.",
            "We move from time-based signals to distinct frequencies."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_COMPLEX = "#FF5555"
        COLOR_CHEF = "#FFFFFF"
        COLOR_FILTER = "#00AAFF"
        COLOR_WAVE1 = "#FF0000"
        COLOR_WAVE2 = "#00FF00"
        COLOR_WAVE3 = "#5555FF"
        COLOR_TIME = "#AAAAAA"
        COLOR_FREQ = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show a complex, jagged red wave #FF5555 titled 'Sound Smoothie'.
        self.play(self.lecture[0].animate.set_color(COLOR_COMPLEX))
        
        def complex_wave_func(x):
            return 0.5 * np.sin(2 * PI * x) + 0.3 * np.sin(4 * PI * x + 0.5) + 0.15 * np.sin(8 * PI * x)

        complex_wave = FunctionGraph(
            complex_wave_func,
            x_range=[-1.5, 1.5],
            color=COLOR_COMPLEX
        )
        self.place_in_area(complex_wave, "B2", "C5", scale_factor=0.6)
        
        smoothie_label = Text("Sound Smoothie", font_size=20, color=COLOR_COMPLEX)
        self.place_at_grid(smoothie_label, "A3", scale_factor=0.8)
        
        self.play(Create(complex_wave), Write(smoothie_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A white icon #FFFFFF representing a chef appears next to the wave.
        self.play(self.lecture[1].animate.set_color(COLOR_CHEF))
        
        # Simple Chef Icon (Hat + Face)
        chef_hat = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.4, width=0.6, fill_opacity=1),
            Circle(radius=0.15, fill_opacity=1).shift(UP*0.2)
        ).set_color(COLOR_CHEF)
        chef_face = Circle(radius=0.2, color=COLOR_CHEF).shift(DOWN*0.3)
        chef_icon = VGroup(chef_hat, chef_face)
        
        self.place_at_grid(chef_icon, "B1", scale_factor=0.7)
        self.play(FadeIn(chef_icon, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A blue filter #00AAFF moves through the red wave, separating it.
        self.play(self.lecture[2].animate.set_color(COLOR_FILTER))
        
        filter_line = Line(UP, DOWN, color=COLOR_FILTER, stroke_width=8).set_height(2.5)
        self.place_at_grid(filter_line, "B2", scale_factor=1.0)
        
        # Movement of filter
        self.play(filter_line.animate.move_to(self.grid["B5"]), run_time=2, rate_func=linear)
        self.play(FadeOut(filter_line))

        # === Animation for Lecture Line 4 ===
        # The red wave splits into three colored sine waves: #FF0000, #00FF00, #5555FF.
        self.play(self.lecture[3].animate.set_color(COLOR_WAVE2))
        
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(2 * PI * x), x_range=[-1, 1], color=COLOR_WAVE1)
        wave2 = FunctionGraph(lambda x: 0.3 * np.sin(4 * PI * x), x_range=[-1, 1], color=COLOR_WAVE2)
        wave3 = FunctionGraph(lambda x: 0.15 * np.sin(8 * PI * x), x_range=[-1, 1], color=COLOR_WAVE3)
        
        self.place_at_grid(wave1, "D3", scale_factor=0.8)
        self.place_at_grid(wave2, "E3", scale_factor=0.8)
        self.place_at_grid(wave3, "F3", scale_factor=0.8)

        # Transition complex wave into components
        self.play(
            complex_wave.animate.scale(0.5).move_to(self.grid["B3"]),
            ReplacementTransform(complex_wave.copy(), wave1),
            ReplacementTransform(complex_wave.copy(), wave2),
            ReplacementTransform(complex_wave.copy(), wave3),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Labels 'Time Domain' #AAAAAA and 'Frequency Domain' #FFFFFF appear.
        self.play(self.lecture[4].animate.set_color(COLOR_FREQ))
        
        time_label = Text("Time Domain", font_size=18, color=COLOR_TIME)
        freq_label = Text("Frequency Domain", font_size=18, color=COLOR_FREQ)
        
        self.place_at_grid(time_label, "A2", scale_factor=0.8)
        self.place_in_area(freq_label, "D4", "F4", scale_factor=0.8)
        
        self.play(Write(time_label), Write(freq_label))
        self.wait(2)
