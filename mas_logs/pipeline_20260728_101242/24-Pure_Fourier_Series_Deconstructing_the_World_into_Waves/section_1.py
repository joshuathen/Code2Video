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
        self.setup_layout("The Grand Analogy: The Secret Recipe of Sound", [
            "- Complex sounds are like recipes made of simple ingredients.",
            "- Meet Echo the Bat, hearing a complex square wave.",
            "- Every repeating wave is just a sum of pure tones."
        ])
        
        # Initialize lecture colors to gray for highlighting
        self.lecture.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Line 1: Complex sounds are like recipes made of simple ingredients.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bat.svg]
        echo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bat.svg").set_color(WHITE)
        self.place_in_area(echo, 'A4', 'B6', scale_factor=0.8)
        
        self.play(FadeIn(echo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Meet Echo the Bat, hearing a complex square wave.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Square wave helper function
        def get_square_wave(width=4, height=0.8, color="#00FFFF"):
            points = []
            for x in np.arange(-width/2, width/2, 1):
                points.extend([[x, height/2, 0], [x+0.5, height/2, 0], [x+0.5, -height/2, 0], [x+1, -height/2, 0]])
            return VMobject(color=color).set_points_as_corners(points)
        
        sq_wave = get_square_wave()
        # Fix Issue 21: self.place_in_area(sq_wave, 'C2', 'C6')
        self.place_in_area(sq_wave, 'C2', 'C6', scale_factor=1.0)
        
        self.play(Create(sq_wave))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Every repeating wave is just a sum of pure tones.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FF0000")
        )
        
        # Pure tones (Sine waves of varying amplitude and frequency)
        # Using approximations of Fourier series components for a square wave
        # Fixed range and scale to fit grid areas nicely
        s1 = FunctionGraph(lambda x: 0.8 * np.sin(x * PI), x_range=[-2.0, 2.0], color="#FF0000")
        s2 = FunctionGraph(lambda x: 0.25 * np.sin(3 * x * PI), x_range=[-2.0, 2.0], color="#00FF00")
        s3 = FunctionGraph(lambda x: 0.15 * np.sin(5 * x * PI), x_range=[-2.0, 2.0], color="#0000FF")
        
        # Fix Issue 22: s1 at 'D2', 'D6'
        self.place_in_area(s1, 'D2', 'D6')
        s1_final_center = s1.get_center()
        
        # Fix Issue 23: s2 at 'E2', 'E6', s3 at 'F2', 'F6'
        self.place_in_area(s2, 'E2', 'E6')
        s2_final_center = s2.get_center()
        
        self.place_in_area(s3, 'F2', 'F6')
        s3_final_center = s3.get_center()
        
        # Positions relative to the square wave for initial appearance
        sq_wave_center = sq_wave.get_center()
        
        # Start at C2-C6 area center for "decomposition" effect
        s1.move_to(sq_wave_center)
        s2.move_to(sq_wave_center)
        s3.move_to(sq_wave_center)
        
        # Deconstruct Square Wave
        self.play(
            ReplacementTransform(sq_wave, s1),
            FadeIn(s2),
            FadeIn(s3)
        )
        
        # Separate vertically into their assigned slots
        self.play(
            s1.animate.move_to(s1_final_center),
            s2.animate.move_to(s2_final_center),
            s3.animate.move_to(s3_final_center),
            run_time=2.0
        )
        self.wait(3)
