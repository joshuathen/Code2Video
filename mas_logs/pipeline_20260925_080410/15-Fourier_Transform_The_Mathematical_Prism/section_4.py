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
        self.setup_layout("Practical Application: Digital Noise Reduction", ["Real world signals often contain noise.", "Filter out the static frequencies.", "Voice remains, noise is gone."])
        
        # Define mobjects
        t = np.linspace(0, 4 * np.pi, 200)
        noisy_data = np.sin(t) + 0.5 * np.sin(10 * t) + 0.2 * np.random.normal(size=len(t))
        
        signal = VMobject()
        signal.set_points_smoothly([np.array([x, y, 0]) for x, y in zip(np.linspace(-2, 2, 200), noisy_data)])
        
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        headset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/headset.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(signal, "C3", "E5", scale_factor=0.4)
        signal.set_color("#FF4500")
        self.place_at_grid(mic, "B2", scale_factor=0.5)
        self.play(Create(signal), FadeIn(mic), self.lecture[0].animate.set_color("#FF4500"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        clean_data = np.sin(t)
        clean_signal = VMobject()
        clean_signal.set_points_smoothly([np.array([x, y, 0]) for x, y in zip(np.linspace(-2, 2, 200), clean_data)])
        # Match position and scale to original noisy signal
        clean_signal.move_to(signal.get_center())
        clean_signal.scale(0.4 / 1.0) # adjust for already scaled
        
        self.play(
            ReplacementTransform(signal, clean_signal),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        clean_signal.set_color("#00FF00")
        self.place_at_grid(headset, "B5", scale_factor=0.5)
        self.play(FadeIn(headset), self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
