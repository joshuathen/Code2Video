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
        self.setup_layout("Application: Radar and Chirp Signals", [
            "Short pulses resolve time, not frequency.", 
            "Long signals resolve frequency, not time.", 
            "Chirps combine both for better radar."
        ])

        # Assets
        radar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radar.svg")
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")

        # === Animation for Lecture Line 1 ===
        # Represent a short pulse
        pulse = VGroup(
            Line(ORIGIN, UP*0.5, color=BLUE),
            Line(UP*0.5, RIGHT*0.1+UP*0.5, color=BLUE),
            Line(RIGHT*0.1+UP*0.5, RIGHT*0.1, color=BLUE)
        )
        self.place_at_grid(pulse, "B2", scale_factor=0.7)
        self.play(Create(pulse))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent a frequency sweep (Chirp) from radar to airplane
        self.place_at_grid(radar, "C3", scale_factor=0.4)
        self.place_at_grid(airplane, "C5", scale_factor=0.4)
        chirp = FunctionGraph(lambda x: np.sin(x**2), x_range=[0, 3], color="#FF33A8")
        self.place_at_grid(chirp, "C5", scale_factor=0.8)
        
        self.play(FadeIn(radar), FadeIn(airplane), Create(chirp))
        self.lecture[1].set_color("#FF33A8")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Represent pulse compression output peak
        peak = Dot(color=YELLOW)
        label = Text("Peak", font_size=16, color=YELLOW)
        peak_group = VGroup(peak, label).arrange(DOWN)
        self.place_at_grid(peak_group, "E5", scale_factor=0.7)
        
        self.play(FadeIn(peak_group))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
