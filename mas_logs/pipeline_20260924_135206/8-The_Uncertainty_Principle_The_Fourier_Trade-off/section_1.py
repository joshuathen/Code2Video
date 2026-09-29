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
        lecture_lines = ["A sharp click is localized in time.", "But it lacks a precise pitch.", "A long note has precise pitch."]
        self.setup_layout("The Uncertainty Principle: The Fourier Trade-off", lecture_lines)
        
        # Define colors for lecture lines
        c1, c2, c3 = "#FFD700", "#FF4500", "#00CED1"
        
        # Load Assets
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        tuning_fork = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tuningfork.svg")
        
        # === Animation for Lecture Line 1 ===
        # A sharp click is localized in time.
        wave1 = FunctionGraph(lambda x: 0.5 * np.exp(-10 * x**2) * np.sin(20 * np.pi * x), x_range=[-1, 1], color=WHITE)
        self.place_in_area(wave1, 'A3', 'B6', scale_factor=0.6)
        self.place_at_grid(metronome, 'A3', scale_factor=0.3)
        self.play(Create(wave1), FadeIn(metronome))
        self.lecture[0].set_color(c1)
        
        # === Animation for Lecture Line 2 ===
        # But it lacks a precise pitch.
        peak1 = Dot(color="#FF5733")
        self.place_at_grid(peak1, 'C3', scale_factor=0.6)
        # Using simple periodic marker for repetition peaks
        peaks = VGroup(*[Dot(color="#FF5733").move_to(self.grid['C3'] + RIGHT * i * 0.2) for i in range(5)])
        self.play(Create(peak1), *[Create(p) for p in peaks])
        self.lecture[1].set_color(c2)
        
        # === Animation for Lecture Line 3 ===
        # A long note has precise pitch.
        wave2 = FunctionGraph(lambda x: 0.3 * np.sin(2 * np.pi * x), x_range=[-2, 2], color=WHITE)
        self.place_in_area(wave2, 'E3', 'F6', scale_factor=0.6)
        self.place_at_grid(tuning_fork, 'E3', scale_factor=0.3)
        self.play(Create(wave2), FadeIn(tuning_fork))
        self.lecture[2].set_color(c3)
        self.wait(2)
