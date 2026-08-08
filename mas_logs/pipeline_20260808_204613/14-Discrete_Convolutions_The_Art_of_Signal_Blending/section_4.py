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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Convolution smooths noisy sensor data.",
            "It underpins modern neural networks.",
            "Signal blending is essential everywhere."
        ]
        self.setup_layout("Applications & Summary", lecture_lines)
        
        # Assets
        icon_audio = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        icon_image = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        icon_signal = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        icon_speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_C)
        self.place_at_grid(icon_audio, 'B4', scale_factor=0.6)
        self.play(FadeIn(icon_audio))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN_C)
        self.place_at_grid(icon_image, 'C4', scale_factor=0.6)
        self.play(FadeIn(icon_image))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW_C)
        self.place_at_grid(icon_signal, 'D4', scale_factor=0.6)
        self.play(FadeIn(icon_signal))
        self.wait(1)
        
        # Add speaker icon for finale
        self.place_at_grid(icon_speaker, 'E4', scale_factor=0.6)
        self.play(FadeIn(icon_speaker))
        self.wait(2)
        
        # Cleanup
        all_mobjects = VGroup(icon_audio, icon_image, icon_signal, icon_speaker, self.lecture, self.title)
        self.play(FadeOut(*all_mobjects))
        self.wait(1)
