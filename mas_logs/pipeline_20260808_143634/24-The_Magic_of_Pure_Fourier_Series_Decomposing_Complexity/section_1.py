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
        self.setup_layout("Intuitive Hook: The Symphony of Waves", [
            "Complex signals are sums of simple sine waves.",
            "Think of a digital synthesizer creating sounds.",
            "Complex patterns emerge from overlapping basic tones."
        ])
        
        # Assets
        synthesizer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/synthesizer.svg")
        self.place_at_grid(synthesizer, 'D3', scale_factor=0.5)

        # Waves
        wave1 = FunctionGraph(lambda x: 0.3 * np.sin(2 * PI * x), x_range=[-1.5, 1.5], color="#FF0000")
        wave2 = FunctionGraph(lambda x: 0.3 * np.sin(4 * PI * x), x_range=[-1.5, 1.5], color="#00FF00")
        combined = FunctionGraph(lambda x: 0.3 * np.sin(2 * PI * x) + 0.3 * np.sin(4 * PI * x), x_range=[-1.5, 1.5], color="#FFFF00")
        
        wave_group = VGroup(wave1, wave2, combined)
        self.place_in_area(wave_group, 'D1', 'F6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(wave1))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(synthesizer))
        self.play(FadeIn(wave2))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(ReplacementTransform(wave1.copy(), combined), ReplacementTransform(wave2.copy(), combined))
        self.lecture[2].set_color("#FFFF00")
        self.play(Indicate(combined, color=WHITE))
        self.wait(2)
