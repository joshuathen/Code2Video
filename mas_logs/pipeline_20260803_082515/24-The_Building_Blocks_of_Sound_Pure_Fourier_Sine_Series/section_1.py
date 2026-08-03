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
        self.setup_layout("The Mystery of the Square Wave", [
            "Meet Oscillo, a robot who sings smooth sine waves.",
            "He wants to mimic this jagged square wave synthesizer.",
            "Can smooth curves ever create such sharp, blocky corners?"
        ])

        # === Animation for Lecture Line 1 ===
        # Oscillo [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] (#ADD8E6)
        oscillo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        oscillo.set_color("#ADD8E6")
        self.place_at_grid(oscillo, "C2", scale_factor=0.8)

        # Smooth sine wave (#00FF00)
        sine_wave = FunctionGraph(lambda x: np.sin(2 * PI * x), x_range=[0, 3], color="#00FF00")
        self.place_in_area(sine_wave, "B3", "E6", scale_factor=0.7)

        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.play(FadeIn(oscillo), Create(sine_wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Square wave (#FF0000) labeled "Target"
        sq_points = []
        for x in np.arange(0, 3.01, 0.01):
            y = 1 if np.sin(2 * PI * x) >= 0 else -1
            if len(sq_points) > 0 and sq_points[-1][1] != y:
                sq_points.append([x, sq_points[-1][1], 0])
            sq_points.append([x, y, 0])
            
        square_wave = VMobject(color="#FF0000").set_points_as_corners([np.array(p) for p in sq_points])
        self.place_in_area(square_wave, "B3", "E6", scale_factor=0.7)
        
        target_label = Text("Target", font_size=24, color="#FF0000")
        self.place_at_grid(target_label, "B5", scale_factor=0.8)

        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(
            FadeOut(sine_wave),
            FadeIn(square_wave),
            FadeIn(target_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Oscillo oscillates between the smooth and sharp waves with confusion.
        question_mark = Text("?", font_size=48, color=YELLOW)
        self.place_at_grid(question_mark, "A4", scale_factor=1.0)

        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(
            FadeIn(question_mark),
            Indicate(oscillo, color=YELLOW)
        )
        
        # Confusion shake using shift to avoid move_to
        for _ in range(3):
            self.play(oscillo.animate.shift(LEFT * 0.2), run_time=0.1)
            self.play(oscillo.animate.shift(RIGHT * 0.2), run_time=0.1)
        
        self.play(Indicate(square_wave, color="#FF0000"))
        self.wait(2)
