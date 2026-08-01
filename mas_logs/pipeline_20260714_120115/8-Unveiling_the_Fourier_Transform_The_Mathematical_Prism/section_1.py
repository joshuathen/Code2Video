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
        # Data from storyboard
        title_text = "The Big Idea: From Smoothies to Ingredients"
        lecture_lines = [
            "Think of a complex signal like a mixed fruit smoothie.",
            "The Fourier Transform is the prism that finds the fruits.",
            "It breaks down complex waves into simple, pure notes.",
            "We move from the time domain to the frequency domain.",
            "This reveals the hidden ingredients inside any signal."
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors based on animation description
        COLOR_WAVE = "#FFFFFF"
        COLOR_SMOOTHIE = "#ADD8E6"
        COLOR_PRISM = "#FFFF00"
        COLOR_R = "#FF0000"
        COLOR_G = "#00FF00"
        COLOR_B = "#0000FF"
        
        # Asset path
        PRISM_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/prism.svg"

        # === Animation for Lecture Line 1 ===
        # Show a complex, white #FFFFFF wave.
        complex_wave = FunctionGraph(
            lambda x: 0.5 * np.sin(2 * PI * x) + 0.3 * np.sin(5 * PI * x) + 0.2 * np.cos(10 * PI * x),
            x_range=[-1.0, 1.0],
            color=COLOR_WAVE
        )
        # Fix for Issue 27: Scale factor 1.0 to prevent crowding
        self.place_in_area(complex_wave, "B2", "D3", scale_factor=1.0)
        
        self.play(
            self.lecture[0].animate.set_color(COLOR_WAVE),
            Create(complex_wave),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform wave into a light blue #ADD8E6 smoothie icon.
        smoothie_cup = Polygon([-0.4, -0.6, 0], [0.4, -0.6, 0], [0.5, 0.6, 0], [-0.5, 0.6, 0], 
                              color=COLOR_SMOOTHIE, fill_opacity=0.8)
        smoothie_top = Arc(radius=0.5, start_angle=0, angle=PI, 
                           color=COLOR_SMOOTHIE, fill_opacity=1).move_to([0, 0.6, 0])
        straw = Line([0.1, 0.6, 0], [0.3, 1.1, 0], color=WHITE, stroke_width=4)
        smoothie = VGroup(smoothie_cup, smoothie_top, straw)
        self.place_in_area(smoothie, "B2", "D3", scale_factor=0.7)
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_SMOOTHIE),
            ReplacementTransform(complex_wave, smoothie),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display a yellow #FFFF00 triangle labeled "Mathematical Prism" [Asset: prism.svg].
        # Fix for Issue 25: Load the SVG asset
        prism = SVGMobject(PRISM_ASSET).set_color(COLOR_PRISM)
        self.place_in_area(prism, "C4", "D4", scale_factor=1.5)
        
        prism_label = Text("Mathematical Prism", font_size=18, color=COLOR_PRISM)
        # Fix for Issue 28: Use place_in_area for label to balance horizontally
        self.place_in_area(prism_label, "E3", "E4", scale_factor=0.8)
        
        self.play(
            self.lecture[2].animate.set_color(COLOR_PRISM),
            FadeIn(prism),
            Write(prism_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Move smoothie through the prism [Asset: prism.svg].
        prism_center = prism.get_center()
        
        self.play(
            self.lecture[3].animate.set_color(COLOR_SMOOTHIE),
            smoothie.animate.move_to(prism_center).scale(0.5).set_opacity(0.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show three colored waves: red #FF0000, green #00FF00, blue #0000FF.
        wave_r = FunctionGraph(lambda x: 0.4 * np.sin(2 * PI * x), x_range=[-0.8, 0.8], color=COLOR_R)
        wave_g = FunctionGraph(lambda x: 0.3 * np.sin(5 * PI * x), x_range=[-0.8, 0.8], color=COLOR_G)
        wave_b = FunctionGraph(lambda x: 0.2 * np.sin(10 * PI * x), x_range=[-0.8, 0.8], color=COLOR_B)
        
        waves = VGroup(wave_r, wave_g, wave_b).arrange(DOWN, buff=0.4)
        # Fix for Issue 29: Expand the area to A5-F6 and scale to 0.9
        self.place_in_area(waves, "A5", "F6", scale_factor=0.9)
        
        self.play(
            self.lecture[4].animate.set_color(COLOR_G),
            FadeOut(smoothie),
            FadeOut(prism),
            FadeOut(prism_label),
            Create(waves),
            run_time=2
        )
        self.wait(3)
