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
        lecture_lines = [
            "We group pulses into equal measures.",
            "Vertical bar lines mark the separation.",
            "These help organize rhythmic flow clearly."
        ]
        self.setup_layout("The Bar Line: Organizing the Chaos", lecture_lines)
        
        # Dots
        dots = VGroup(*[Dot(color=WHITE) for _ in range(12)])
        pulses_group = VGroup(*dots).arrange(RIGHT, buff=0.3)
        self.place_in_area(pulses_group, 'C1', 'E3', scale_factor=0.5)
        
        # Metronome asset
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        self.place_at_grid(metronome, 'F6', scale_factor=0.4)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(pulses_group))
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        bar_lines = VGroup()
        # Bar lines after dots 3 and 7
        for i in [3, 7]:
            bar = Line(UP*0.5, DOWN*0.5, color=RED).next_to(pulses_group[i], RIGHT, buff=0.1)
            bar_lines.add(bar)
        
        self.place_in_area(bar_lines, 'C4', 'F6', scale_factor=0.6)
        self.play(Create(bar_lines), FadeIn(metronome))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Groups of 4
        group1 = VGroup(*pulses_group[0:4])
        group2 = VGroup(*pulses_group[4:8])
        group3 = VGroup(*pulses_group[8:12])
        
        self.play(
            group1.animate.shift(LEFT*0.1),
            group2.animate.shift(RIGHT*0.1),
            group3.animate.shift(RIGHT*0.2)
        )
        self.wait(4)
