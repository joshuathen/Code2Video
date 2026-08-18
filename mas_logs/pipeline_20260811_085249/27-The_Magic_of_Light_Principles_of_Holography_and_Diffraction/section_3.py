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
            "Holography uses a two-beam interference technique.",
            "A laser splits into object and reference beams.",
            "They combine on a sensitive plate.",
            "The plate captures the interference pattern.",
            "This is the hologram's coded data."
        ]
        self.setup_layout("The Holographic Recording Process", lecture_lines)
        
        # Load Assets
        laser_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        object_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/object.svg")
        plate_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg")
        
        # Position using revised grid coordinates
        self.place_at_grid(laser_icon, 'B2', scale_factor=0.7)
        self.place_at_grid(object_icon, 'B5', scale_factor=0.8)
        self.place_at_grid(plate_icon, 'E5', scale_factor=1.0)
        
        # Interference Pattern
        interference = VGroup(*[
            Line(plate_icon.get_top(), plate_icon.get_bottom()).set_stroke(YELLOW, width=2)
            for _ in range(10)
        ]).arrange(RIGHT, buff=0.1).move_to(plate_icon.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(laser_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        beam_obj = Line(laser_icon.get_right(), object_icon.get_left(), color=RED)
        beam_ref = Line(laser_icon.get_right(), plate_icon.get_left(), color=RED)
        self.play(Create(beam_obj), Create(beam_ref))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(FadeIn(object_icon), FadeIn(plate_icon))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(FadeIn(interference))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.play(
            FadeOut(laser_icon), FadeOut(beam_obj), FadeOut(beam_ref), 
            FadeOut(object_icon)
        )
        self.wait(2)
