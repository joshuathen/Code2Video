from manim import *

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
        self.setup_layout("Real-world Application: Noise Cancellation", [
            "Noise cancellation subtracts unwanted signal frequencies.",
            "Isolate specific noise from the main audio.",
            "Recover clear sounds by removing interference."
        ])
        
        # Assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        headphones = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/headphones.svg")
        
        # Define waves
        axes = Axes(x_range=[0, 4, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False}).scale(0.5)
        signal = axes.plot(lambda x: np.sin(2 * PI * x), color="#00FF00")
        noise = axes.plot(lambda x: 0.5 * np.sin(10 * PI * x), color="#FF0000")
        anti_noise = axes.plot(lambda x: -0.5 * np.sin(10 * PI * x), color="#0000FF")
        
        combined = VGroup(axes, signal, noise)
        waveform_group = VGroup(axes, signal, noise, anti_noise)
        signal_labels = VGroup(Text("Signal", color="#00FF00", font_size=20), Text("Noise", color="#FF0000", font_size=20))
        
        # Positioning requested by critic
        self.place_in_area(combined, 'B4', 'E6', scale_factor=0.7)
        self.place_at_grid(waveform_group, 'C4', scale_factor=0.75)
        self.place_in_area(signal_labels, 'D4', 'E5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(mic, "A4", scale_factor=0.5)
        self.play(self.lecture[0].animate.set_color("#00FF00"), Create(mic), Create(signal), Create(noise))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"), Create(anti_noise))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(headphones, "F4", scale_factor=0.5)
        self.play(self.lecture[2].animate.set_color("#0000FF"), 
                  FadeOut(noise), FadeOut(anti_noise), Create(headphones))
        self.wait(1)
