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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Fact Storage: The Embedding Space", 
                          ["Learning shifts weights via gradient descent.", 
                           "Facts are etched into the network.", 
                           "Weights align to map input to output."])
        
        # Elements
        hammer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hammer.svg")
        chisel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chisel.svg")
        
        vec_start = Vector(RIGHT * 2, color=BLUE)
        self.place_at_grid(vec_start, 'C4')
        vec_target = Vector(UP * 2, color=GREEN)
        self.place_at_grid(vec_target, 'C4')
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        hammer_mob = self.place_at_grid(hammer.copy(), 'B3', scale_factor=0.3)
        self.play(FadeIn(vec_start), FadeIn(hammer_mob))
        self.play(Rotate(vec_start, angle=PI/2, about_point=self.grid['C4']), FadeOut(hammer_mob))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        chisel_mob = self.place_at_grid(chisel.copy(), 'D5', scale_factor=0.3)
        self.play(FadeIn(chisel_mob))
        # Simple particle animation represented by a path dash
        path = DashedLine(vec_start.get_end(), vec_target.get_end(), color=RED)
        self.play(Create(path), chisel_mob.animate.move_to(self.grid['D3']))
        self.play(FadeOut(chisel_mob))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        arc = ArcBetweenPoints(vec_start.get_end(), vec_target.get_end(), angle=PI/6, color=YELLOW)
        self.play(Create(arc))
        chisel_final = self.place_at_grid(chisel.copy(), 'E4', scale_factor=0.4)
        self.play(FadeIn(chisel_final))
        self.wait(1)
