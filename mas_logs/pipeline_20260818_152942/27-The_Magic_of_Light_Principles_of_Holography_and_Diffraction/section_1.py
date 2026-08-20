from manim import *
import os

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
        self.setup_layout("Prerequisite: The Wave Nature of Light", [
            "Light waves undergo superposition and interference.",
            "Diffraction is light bending around obstacles.",
            "This is the foundation for holography."
        ])
        
        # Load assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        hologram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")

        # --- Visual Objects ---
        # Wave crests (for line 1) - using asset
        waves = VGroup(*[
            Line(LEFT*2, RIGHT*2, color=WHITE).shift(UP*i*0.5) 
            for i in range(-2, 3)
        ])
        self.place_at_grid(waves, 'D2', scale_factor=0.6)
        laser.scale(0.5).next_to(waves, LEFT)

        # Beams for line 2
        beam1 = Line(ORIGIN, RIGHT*3, color=WHITE).shift(UP*0.5)
        beam2 = Line(ORIGIN, RIGHT*3, color=WHITE).shift(DOWN*0.5)
        self.place_at_grid(lens, 'D5', scale_factor=0.5)
        interference_zone = Rectangle(height=2, width=1, color=BLUE, fill_opacity=0.3).next_to(lens, RIGHT, buff=0)
        beams = VGroup(beam1, beam2, interference_zone)
        self.place_at_grid(beams, 'D5', scale_factor=0.7)
        
        overall_animation_group = VGroup(waves, laser, beams, lens, interference_zone)
        self.place_in_area(overall_animation_group, 'A3', 'B5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(laser), Create(waves), run_time=1)
        self.play(waves.animate.shift(RIGHT*0.5), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Create(beams), Create(lens), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        hologram.scale(0.5).next_to(interference_zone, RIGHT)
        self.play(FadeIn(hologram), interference_zone.animate.set_color(BLUE), run_time=1)
        self.wait(1)
