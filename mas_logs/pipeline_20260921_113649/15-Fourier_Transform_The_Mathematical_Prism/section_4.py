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
            "Identify noise in the signal.",
            "Frequency domain isolates static.",
            "Remove noise to clean data.",
            "The signal becomes crisp.",
            "Fourier makes this filtering possible."
        ]
        self.setup_layout("Practical Application: Noise Reduction", lecture_lines)
        
        # Assets
        microphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        checkmark = Text("✔").set_color("#2ECC71")

        # Create Signal Representations
        time_axis = NumberLine(x_range=[0, 10, 1], length=3, color=BLUE)
        freq_axis = NumberLine(x_range=[0, 10, 1], length=3, color=RED)
        
        # Positioning based on criticisms
        self.place_in_area(time_axis, 'B4', 'C6', scale_factor=0.8)
        self.place_in_area(freq_axis, 'E4', 'F6', scale_factor=0.8)

        t = np.linspace(0, 10, 200)
        signal_data = np.sin(t) + 0.3 * np.random.normal(0, 1, 200)
        
        def get_signal(data, color=WHITE):
            coords = [
                np.array([0, d*0.5, 0]) for d in data
            ]
            path = VMobject()
            path.set_points_smoothly(coords)
            path.set_color(color)
            return path
            
        noisy_signal = get_signal(signal_data, WHITE)
        self.place_at_grid(microphone, "A4", scale_factor=0.5)
        
        signal_group = VGroup(noisy_signal, time_axis)
        self.place_in_area(signal_group, 'B4', 'F6', scale_factor=0.75)
        
        self.add(microphone, signal_group, freq_axis)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Indicate(noisy_signal))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        spike1 = Line(ORIGIN, UP*1, color="#E74C3C").next_to(freq_axis, UP, buff=0.1)
        spike2 = Line(ORIGIN, UP*2, color="#E74C3C").next_to(freq_axis, UP, buff=0.1).shift(RIGHT*1)
        self.play(Create(spike1), Create(spike2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(FadeOut(spike2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        clean_signal = get_signal(np.sin(t), BLUE)
        clean_signal.move_to(noisy_signal.get_center())
        self.play(Transform(noisy_signal, clean_signal))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.place_at_grid(computer, "D6", scale_factor=0.5)
        self.place_at_grid(checkmark, "E6", scale_factor=0.5)
        self.play(FadeIn(computer), FadeIn(checkmark))
        self.wait(1)
