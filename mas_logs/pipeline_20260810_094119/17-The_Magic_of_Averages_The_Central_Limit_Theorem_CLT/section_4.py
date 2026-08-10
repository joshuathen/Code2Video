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
        self.setup_layout("The Theorem Defined: The 'Magic' Rule", [
            "Regardless of the initial population shape.",
            "As sample size 'n' increases steadily.",
            "The distribution of means approaches normality.",
            "It centers perfectly at the population mean.",
            "This is the Central Limit Theorem."
        ])
        
        # Load assets
        pop_dist = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        pop_dist.set_color("#F1C40F")
        
        # Create other dynamic mobjects
        sample_dist = Dot(color="#FF5733").scale(2)
        final_dist = Dot(color="#3498DB").scale(3)
        label_norm = Text("Normal Distribution", font_size=20, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#F1C40F")
        self.place_in_area(pop_dist, 'A4', 'C6', scale_factor=0.6)
        self.play(FadeIn(pop_dist))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.place_in_area(sample_dist, 'D4', 'F6', scale_factor=0.6)
        self.play(FadeIn(sample_dist))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3498DB")
        self.play(Transform(sample_dist, final_dist))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#3498DB")
        self.place_at_grid(label_norm, 'D5', scale_factor=0.8)
        self.play(Write(label_norm))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
