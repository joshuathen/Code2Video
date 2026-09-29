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
        self.setup_layout("Measure Theory in Action", [
            "We emphasize beats to create rhythm.",
            "Strong and weak beats define the groove.",
            "Accents follow patterns within the measure."
        ])

        # Define assets
        bar_line = Line(UP, DOWN, color="#FFFFFF", stroke_width=4).scale(1.5)
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        drumsticks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drumsticks.svg")
        
        notes = VGroup(*[Circle(radius=0.2, color="#00FFFF", fill_opacity=0.6) for _ in range(4)])
        notes.arrange(RIGHT, buff=0.3)

        # === Animation for Lecture Line 1 ===
        # Show a bar line and metronome
        self.place_at_grid(bar_line, 'D3', scale_factor=0.6)
        self.place_at_grid(metronome, 'C3', scale_factor=0.5)
        self.play(Create(bar_line), FadeIn(metronome))
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Grouping note symbols
        self.place_at_grid(notes, 'D4', scale_factor=0.6)
        self.play(FadeIn(notes))
        
        # Emphasize downbeat with a color pulse
        downbeat = notes[0]
        pulse = Circle(radius=0.3, color="#FF4500", stroke_opacity=0.5).move_to(downbeat.get_center())
        self.play(FadeIn(pulse), pulse.animate.scale(1.5).set_stroke(opacity=0), run_time=1)
        self.remove(pulse)
        downbeat.set_fill(color="#FF4500", opacity=0.8)
        
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shift emphasis and animate flow with drumsticks
        self.place_at_grid(drumsticks, 'E4', scale_factor=0.5)
        self.play(FadeIn(drumsticks))
        self.play(
            notes[0].animate.set_fill(color="#00FFFF", opacity=0.6),
            notes[2].animate.set_fill(color="#FF4500", opacity=0.8),
            drumsticks.animate.shift(RIGHT * 1)
        )
        
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
