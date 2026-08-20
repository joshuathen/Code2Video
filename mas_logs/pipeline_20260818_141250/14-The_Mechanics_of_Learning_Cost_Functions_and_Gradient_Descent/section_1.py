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
        self.setup_layout("The Learning Objective (Prerequisite)", [
            "Neural networks aim to minimize prediction error.",
            "Parameters are knobs we turn to adjust.",
            "Weights and biases control the network's output."
        ])
        
        # Elements
        error_label = Text("Goal: Minimize Error", font_size=24, color=WHITE)
        weight_label = Text("Weight", font_size=24, color="#FFCC00")
        bias_label = Text("Bias", font_size=24, color="#00CCFF")
        
        # Use assets as requested
        knobs = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knobs.svg")
        archer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/archer.svg")
        target = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        arrow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg")
        
        bullseye = Dot(color=YELLOW)
        hit_mark = Dot(color=RED)
        error_dist = Line(color=RED)
        error_text = Text("Error", font_size=20, color=RED)
        
        # Applying requested positions (Issues 21, 22, 23)
        self.place_at_grid(error_label, 'B2', scale_factor=0.7)
        self.place_at_grid(weight_label, 'C2', scale_factor=0.8)
        self.place_at_grid(bias_label, 'C4', scale_factor=0.8)
        self.place_in_area(target, 'D2', 'F4', scale_factor=0.9)
        self.place_at_grid(bullseye, 'E3', scale_factor=1.0)
        
        # Position icons
        self.place_at_grid(knobs, 'C3', scale_factor=0.5)
        self.place_at_grid(archer, 'D1', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), FadeIn(error_label))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), FadeIn(weight_label), FadeIn(bias_label), FadeIn(knobs))
        self.lecture[1].set_color("#FFCC00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]), FadeIn(archer), FadeIn(target), FadeIn(bullseye))
        arrow.next_to(archer, RIGHT)
        self.play(FadeIn(arrow))
        hit_mark.move_to(target.get_center() + RIGHT*0.2 + UP*0.1)
        self.play(FadeIn(hit_mark))
        
        error_dist.put_start_and_end_on(hit_mark.get_center(), bullseye.get_center())
        self.play(Create(error_dist), Write(error_text.next_to(error_dist, UP)))
        self.lecture[2].set_color("#00CCFF")
        self.wait(2)
