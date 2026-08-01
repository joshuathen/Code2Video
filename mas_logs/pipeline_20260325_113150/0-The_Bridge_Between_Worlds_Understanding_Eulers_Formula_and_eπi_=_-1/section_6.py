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
        # Setup layout with title and bullet points
        self.setup_layout("Application: Why Does This Matter?", [
            "This formula simplifies messy signals into clean mathematical waves.",
            "It transforms complex oscillations into easy-to-manage exponential terms.",
            "This bridge enables your smartphone to process digital audio."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # Create a messy wave representing a signal (#00FFFF) and label it 'Complex Signal'.
        messy_wave = FunctionGraph(
            lambda x: 0.25 * np.sin(5 * x) + 0.15 * np.cos(11 * x) + 0.1 * np.sin(21 * x),
            x_range=[-1.5, 1.5],
            color="#00FFFF"
        )
        # Positioned at B2 with scale 0.8 to avoid visual tension with title (Issue 37)
        self.place_at_grid(messy_wave, "B2", scale_factor=0.8)
        
        signal_label = Text("Complex Signal", font_size=20, color="#00FFFF")
        self.place_at_grid(signal_label, "B2", scale_factor=0.8)
        signal_label.shift(UP * 0.8)

        self.play(Create(messy_wave), Write(signal_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the second lecture line
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Show a transformation arrow (#FFFFFF) pointing from the wave to the mathematical expression e^(i*omega*t).
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        self.place_at_grid(arrow, "B4", scale_factor=0.6)
        
        # FIXED: Replaced MathTex with Text to avoid FileNotFoundError: [Errno 2] No such file or directory: 'latex'
        formula = Text("e^iωt", color=WHITE)
        self.place_at_grid(formula, "B5", scale_factor=1.0)

        self.play(GrowArrow(arrow), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the third lecture line
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Show a simplified diagram of a smartphone processing the signal into a clear audio output icon (#FFFF00).
        phone = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/smartphone.svg")
        self.place_at_grid(phone, "D4", scale_factor=0.8)
        
        audio_icon = Text("♪", color="#FFFF00", font_size=40)
        self.place_at_grid(audio_icon, "D5", scale_factor=1.0)
        
        self.play(FadeIn(phone))
        self.play(Indicate(phone, color="#FFFF00"), FadeIn(audio_icon))
        self.wait(2)
