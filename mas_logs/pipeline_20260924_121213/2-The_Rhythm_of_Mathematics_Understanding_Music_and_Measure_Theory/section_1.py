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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Music is organized by steady pulses called beats.", "A measure is a container for musical time.", "Time signatures define the rhythm of the music."]
        self.setup_layout("Introduction: The Heartbeat of Music", lecture_lines)
        
        # Colors per lecture line
        colors = ["#FF00FF", "#00FFFF", "#FFFF00"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        heart = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg", color=colors[0])
        self.place_at_grid(heart, 'C5', scale_factor=0.6)
        
        # Pulse animation
        pulse = Circle(radius=0.5, color=colors[0]).move_to(heart.get_center())
        self.add(pulse)
        self.play(FadeIn(heart), FadeIn(pulse), run_time=1)
        self.play(pulse.animate.scale(2).set_opacity(0), run_time=1.5)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        # Container (Measure) representation
        container = Rectangle(height=2, width=4, color=colors[1])
        self.place_in_area(container, 'D3', 'E5', scale_factor=0.7)
        
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg", color=colors[1])
        metronome.next_to(container, UP)
        
        self.play(Create(container), FadeIn(metronome), run_time=1)
        
        ticks = VGroup(*[Line(UP*0.5, DOWN*0.5, color=colors[1]) for _ in range(4)]).arrange(RIGHT, buff=0.8)
        ticks.move_to(container.get_center())
        self.play(Create(ticks), run_time=1.5)
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        # Time Signature: "4/4"
        ts = MathTex(r"\\frac{4}{4}", color=colors[2]).scale(2)
        self.place_at_grid(ts, 'B5', scale_factor=0.7)
        self.play(Write(ts), run_time=1.5)
        self.wait(4)
