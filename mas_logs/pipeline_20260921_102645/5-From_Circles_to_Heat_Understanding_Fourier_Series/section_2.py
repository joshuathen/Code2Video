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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Fourier Series: Decomposing Complexity", [
            "Periodic signals decompose into sine waves.", 
            "Harmonics create a signal's unique timbre.", 
            "More harmonics sharpen the signal's edges."
        ])
        
        # Axes for the right-side visualizations
        # Apply fix 26: Reposition axes
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-2, 2, 1],
            axis_config={"include_numbers": False},
            x_length=4, y_length=3
        )
        self.place_at_grid(axes, 'D4', scale_factor=0.7)
        self.add(axes)
        
        # Assets
        synth_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/synthesizer.svg")
        speaker_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        self.place_at_grid(synth_icon, 'B3', scale_factor=0.4)
        self.place_at_grid(speaker_icon, 'F6', scale_factor=0.4)
        
        # Animation Group
        fourier_group = VGroup()
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        wave1 = axes.plot(lambda x: np.sin(x), color="#FF00FF")
        fourier_group.add(wave1)
        self.play(FadeIn(synth_icon), Create(wave1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        wave2 = axes.plot(lambda x: 0.5 * np.sin(3 * x), color="#00FFFF")
        self.lecture[2].set_color("#00FF00")
        wave3 = axes.plot(lambda x: 0.33 * np.sin(5 * x), color="#00FF00")
        fourier_group.add(wave2, wave3)
        self.play(Create(wave2), Create(wave3))
        
        # === Animation for Lecture Line 3 ===
        sum_wave = axes.plot(lambda x: np.sin(x) + 0.5 * np.sin(3 * x) + 0.33 * np.sin(5 * x), color="#FFFF00")
        
        # Apply fix 24: Transform/move animation
        # We transform the component waves to the sum wave
        self.play(
            Transform(fourier_group, sum_wave),
            run_time=2
        )
        self.play(FadeIn(speaker_icon), Indicate(fourier_group, color="#FFFFFF", scale_factor=1.1))
