from manim import *
import numpy as np

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
        self.setup_layout("Synthesis & Summary", [
            "Beat is the unit of time.",
            "Measure is the container for notes.",
            "Time signature sets the rhythmic rule."
        ])

        # Assets
        container_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/container.svg", color=WHITE)
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        metronome_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg", color=WHITE)

        # Objects
        beat_label = Text("Beat=Unit", font_size=24, color=YELLOW)
        beat_group = VGroup(beat_label)
        
        measure_label = Text("Measure=Container", font_size=24, color=BLUE)
        measure_group = VGroup(measure_label, container_icon)
        container_icon.next_to(measure_label, UP)
        
        rule_label = Text("Rule=Signature", font_size=24, color=GREEN)
        rule_group = VGroup(rule_label, ruler_icon)
        ruler_icon.next_to(rule_label, UP)

        # Sine wave
        axes = Axes(x_range=[0, 4*PI], y_range=[-1, 1], axis_config={"include_ticks": False}).scale(0.5)
        sine = axes.plot(np.sin, color="#00CED1")
        wave_group = VGroup(sine, metronome_icon)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(beat_group, "B3", scale_factor=0.8)
        self.play(Write(beat_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(measure_group, "B5", scale_factor=0.8)
        self.play(FadeIn(measure_group))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(rule_group, "D4", scale_factor=0.7)
        self.play(FadeIn(rule_group))
        
        # Sine wave and metronome
        self.place_at_grid(wave_group, "E5", scale_factor=0.7)
        self.play(Create(sine), FadeIn(metronome_icon))
        self.play(Indicate(rule_group))
        self.wait(2)
