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
        lecture_lines = [
            "Any periodic signal is a sum of sine waves.",
            "Think of a guitar chord as individual notes.",
            "We can break complex sounds into simple parts."
        ]
        self.setup_layout("The Hook: The Musical Instrument Analogy", lecture_lines)
        
        # Pre-create elements
        # Waves: complex (#FF00FF), components (#00FFFF)
        complex_wave = FunctionGraph(lambda x: 0.5 * np.sin(x) + 0.3 * np.sin(2*x) + 0.2 * np.sin(3*x), x_range=[-2, 2], color="#FF00FF")
        comp1 = FunctionGraph(lambda x: 0.5 * np.sin(x), x_range=[-2, 2], color="#00FFFF")
        comp2 = FunctionGraph(lambda x: 0.3 * np.sin(2*x), x_range=[-2, 2], color="#00FFFF")
        comp3 = FunctionGraph(lambda x: 0.2 * np.sin(3*x), x_range=[-2, 2], color="#00FFFF")
        wave_group = VGroup(complex_wave, comp1, comp2, comp3)
        
        # Asset
        guitar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg", color=WHITE)
        note = Text("♪", color="#FFFF00", font_size=48)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        self.place_in_area(wave_group, "B4", "D6", scale_factor=0.8)
        self.place_at_grid(guitar_icon, "A5", scale_factor=0.5)
        self.play(Create(wave_group), FadeIn(guitar_icon))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(note, "C3", scale_factor=0.8)
        self.play(FadeIn(note))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF00FF")
        # Synthesize complex wave
        self.play(comp1.animate.shift(UP*0.3), comp2.animate.shift(DOWN*0.3), run_time=1)
        self.wait(2)
