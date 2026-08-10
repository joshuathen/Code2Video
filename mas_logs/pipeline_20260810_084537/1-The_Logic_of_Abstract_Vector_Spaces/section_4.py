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
        self.setup_layout("Application: Why Abstract Matters", 
                          ["Abstracting vectors solves complex problems.", 
                           "Think of audio waves as vectors.", 
                           "Simple algebra controls complex reality."])
        
        # Load assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        # Define mobjects
        waveform = FunctionGraph(lambda x: 0.5 * np.sin(4 * x) * np.exp(-0.2 * x**2), 
                                 x_range=[-3, 3], color="#9b59b6")
        
        rules = VGroup(
            Tex(r"$u + v = v + u$", font_size=24, color="#bdc3c7"),
            Tex(r"$c(u + v) = cu + cv$", font_size=24, color="#bdc3c7")
        ).arrange(DOWN, aligned_edge=LEFT)
        
        # Grouped visual components for positioning
        visual_group = VGroup(waveform, mic, speaker)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(mic, 'B2', scale_factor=0.5)
        self.place_in_area(waveform, 'A4', 'C6', scale_factor=0.9)
        self.play(FadeIn(mic), Create(waveform))
        self.lecture[0].set_color("#9b59b6")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(rules, 'E4', scale_factor=0.7)
        self.play(FadeIn(rules))
        self.lecture[1].set_color("#3498db")

        # === Animation for Lecture Line 3 ===
        new_waveform = FunctionGraph(lambda x: 1.2 * 0.5 * np.sin(4 * x) * np.exp(-0.2 * x**2), 
                                     x_range=[-3, 3], color="#e74c3c")
        
        self.place_at_grid(speaker, 'F6', scale_factor=0.5)
        self.play(
            Transform(waveform, new_waveform),
            FadeIn(speaker)
        )
        self.lecture[2].set_color("#e74c3c")
        self.wait(1)
