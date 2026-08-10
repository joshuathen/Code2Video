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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Digital Signal Processor", [
            "Fourier series power modern digital compression.",
            "Adjusting coefficients filters audio noise away.",
            "Equalizers show these principles in action."
        ])
        
        # Load assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        # Data points for waveform
        points = VGroup(*[Dot(radius=0.05, color="#3498DB") for _ in range(20)])
        for i, p in enumerate(points):
            p.move_to(np.array([2.5 + (i * 0.2), 0.5 * np.sin(i * 0.3), 0]))
            
        # Fourier curve
        fourier_curve = FunctionGraph(lambda x: 0.5 * np.sin(x), x_range=[-3, 3], color="#E74C3C")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#3498DB")
        self.place_at_grid(mic, 'A4', scale_factor=0.3)
        self.play(FadeIn(mic), Create(points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E74C3C")
        # Applying requested area/scale constraints from issues 33-35
        self.place_in_area(fourier_curve, 'B3', 'F6', scale_factor=0.65)
        self.play(Create(fourier_curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F1C40F")
        self.place_at_grid(speaker, 'F6', scale_factor=0.3)
        # Final visual element with required color
        final_wave = FunctionGraph(lambda x: 0.5 * np.sin(x), x_range=[-3, 3], color="#2ECC71")
        self.place_in_area(final_wave, 'B3', 'F6', scale_factor=0.65)
        self.play(Transform(fourier_curve, final_wave), FadeIn(speaker))
        self.wait(2)
