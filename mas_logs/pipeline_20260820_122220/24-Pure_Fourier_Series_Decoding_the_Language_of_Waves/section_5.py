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
        self.setup_layout("Conclusion and Real-World Application", 
                         ["Fourier series maps time signals to frequencies.", 
                          "This data enables modern signal compression.", 
                          "Technology discards irrelevant data to optimize storage."])
        
        self.lecture.set_opacity(1)

        # Load assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")

        # 1. Complex Signal / Frequency Spectrum
        time_axes = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], axis_config={"include_numbers": False}).scale(0.3)
        signal = time_axes.plot(lambda x: np.sin(2 * PI * x) + 0.5 * np.sin(4 * PI * x), color="#FFD700")
        
        freq_axes = Axes(x_range=[0, 5, 1], y_range=[0, 2, 1], axis_config={"include_numbers": False}).scale(0.3)
        spectrum = VGroup(
            Line(freq_axes.c2p(1,0), freq_axes.c2p(1,1), color="#FFD700"),
            Line(freq_axes.c2p(2,0), freq_axes.c2p(2,0.5), color="#FFD700")
        )
        
        # 2. Audio Equalizer
        eq = VGroup(*[Rectangle(height=np.random.rand()*1.5, width=0.3, fill_opacity=0.8, color="#00FF00") for _ in range(5)]).arrange(RIGHT, buff=0.1)

        # 3. Clean Reconstructed Signal
        recon_signal = time_axes.plot(lambda x: np.sin(2 * PI * x), color="#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        asset_group1 = VGroup(time_axes, signal, freq_axes, spectrum, mic)
        self.place_in_area(asset_group1, "A4", "C6", scale_factor=0.8)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        # Fixing per critique
        self.place_in_area(eq, 'A3', 'B4', scale_factor=0.9)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Fixing per critique
        recon_group = VGroup(recon_signal, speaker)
        self.place_in_area(recon_group, 'D3', 'E4', scale_factor=0.9)
        self.wait(2)
