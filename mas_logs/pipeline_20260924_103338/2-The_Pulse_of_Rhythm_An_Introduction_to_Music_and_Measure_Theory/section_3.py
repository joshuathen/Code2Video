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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Time signatures act as our blueprint.",
            "The top number sets beats per measure.",
            "The bottom defines the beat's value.",
            "Together they guide the rhythmic structure.",
            "Imagine a chef filling musical bowls."
        ]
        self.setup_layout("Time Signature: The Mathematical Blueprint", lecture_lines)
        
        # Define elements using SVG assets
        bowl = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bowl.svg")
        note_template = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/note.svg")
        notes = VGroup(*[note_template.copy() for _ in range(4)])
        bar_line = Line(UP*0.5, DOWN*0.5, color=WHITE)
        time_sig = Text("4/4", color="#00FFFF", font_size=48)

        # Positioning
        self.place_in_area(bowl, 'C4', 'E6', scale_factor=0.6)
        self.place_at_grid(time_sig, 'B4', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(bowl), Write(time_sig))
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        for i in range(4):
            note = notes[i]
            # Drop notes into bowl area
            self.place_at_grid(note, 'C4', scale_factor=0.2)
            self.play(FadeIn(note), run_time=0.5)
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        self.play(time_sig.animate.set_color(RED))
        self.wait(3)
        self.play(time_sig.animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.place_at_grid(bar_line, 'D6', scale_factor=0.8)
        self.play(Create(bar_line))
        self.wait(3)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        for _ in range(3):
            self.play(time_sig.animate.scale(1.2), run_time=0.3)
            self.play(time_sig.animate.scale(1/1.2), run_time=0.3)
        self.wait(3)
