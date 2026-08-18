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
        self.setup_layout("Summary & The Universal Language", [
            "Music is just applied math.", 
            "Measures help us organize patterns.", 
            "Rhythm is counting in disguise."
        ])
        
        # Elements
        pulse = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg", color=WHITE)
        rhythmic_structures = VGroup(*[
            Square(side_length=0.3, color="#7FFF00", fill_opacity=0.5).shift(i*0.5*RIGHT) 
            for i in range(5)
        ])
        glow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg", color="#8A2BE2")

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(pulse, 'E2', scale_factor=0.6)
        self.play(FadeIn(pulse), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        self.place_in_area(rhythmic_structures, 'A4', 'B6', scale_factor=0.5)
        self.play(Create(rhythmic_structures), self.lecture[1].animate.set_color("#7FFF00"))

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(glow, 'E2', scale_factor=0.8)
        self.play(FadeIn(glow), self.lecture[2].animate.set_color("#8A2BE2"))
        self.play(FadeOut(pulse), FadeOut(rhythmic_structures), FadeOut(glow))
