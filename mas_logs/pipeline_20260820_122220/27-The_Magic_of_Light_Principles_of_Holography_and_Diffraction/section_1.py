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
        self.setup_layout("Prerequisite: The Wave Nature of Light", [
            "Light waves act like ripples in a pond.", 
            "Coherence means stable, rhythmic wave relationships.", 
            "Laser beams create perfectly rhythmic waves."
        ])
        
        # Assets: Use SVGMobject instead of ImageMobject for .svg files
        pond_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pond.svg")
        laser_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(pond_icon, "A4", scale_factor=0.3)
        wave = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[-2, 2], color="#FFD700")
        self.place_in_area(wave, "B4", "C6", scale_factor=0.6)
        
        self.play(FadeIn(pond_icon), Create(wave))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        
        crests = VGroup(*[Dot(color="#00CED1").move_to(wave.point_from_proportion(i/10)) for i in range(1, 10, 2)])
        troughs = VGroup(*[Dot(color="#00CED1").move_to(wave.point_from_proportion(i/10)) for i in range(0, 10, 2)])
        self.play(FadeIn(crests), FadeIn(troughs))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.place_at_grid(laser_icon, "D4", scale_factor=0.3)
        
        # Interference representation
        interf = VGroup(*[
            FunctionGraph(lambda x: 0.3 * np.sin(4 * x + i), x_range=[-2, 2], color="#FF4500")
            for i in [0, 0.5]
        ])
        self.place_in_area(interf, "E4", "F6", scale_factor=0.6)
        
        self.play(FadeIn(laser_icon), Create(interf))
        
        # Central bright spot highlight
        bright_spot = Dot(color="#FFD700").scale(1.5)
        self.place_at_grid(bright_spot, "D6")
        self.play(Flash(bright_spot))
        
        self.wait(2)
