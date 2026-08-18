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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Magic of Pure Fourier Series", [
            "Complex periodic signals are musical chords.", 
            "Fourier series act like a mathematical prism.", 
            "Signals decompose into simple sine notes."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg] in #FFFF00 representing a pure frequency.
        instrument = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg")
        instrument.set_color("#FFFF00")
        self.place_at_grid(instrument, 'B5', scale_factor=0.6)
        self.play(FadeIn(instrument))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        # Show three sine waves of different frequencies overlaying to create a complex signal, visualized through the [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg].
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        self.place_at_grid(prism, 'C2', scale_factor=0.5)
        
        axes = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], axis_config={"include_tip": False})
        sine1 = axes.plot(lambda x: np.sin(PI * x), color="#00FF00")
        sine2 = axes.plot(lambda x: 0.5 * np.sin(2 * PI * x), color="#00FFFF")
        sine3 = axes.plot(lambda x: 0.3 * np.sin(3 * PI * x), color="#FF00FF")
        
        self.place_in_area(axes, 'D4', 'F6', scale_factor=0.3)
        
        self.play(FadeIn(prism), Create(axes), Create(sine1), Create(sine2), Create(sine3))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
