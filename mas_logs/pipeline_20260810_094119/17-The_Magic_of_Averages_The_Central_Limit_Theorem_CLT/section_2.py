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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Refresher: The Building Blocks", 
                          ["First, understand the population distribution.", 
                           "It shows all possible individual outcomes.", 
                           "Next, visualize our sampling distribution."])
        
        # Setup population icons
        pop_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        bar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bars.svg")
        
        # Combine into a group styled as #F1C40F
        population_group = VGroup(pop_icon, bar_icon).set_color("#F1C40F")
        
        label = Text("Skewed Distribution", font_size=24, color=WHITE)
        display_group = VGroup(population_group, label).arrange(DOWN, buff=0.3)
        self.place_in_area(display_group, 'B1', 'E3', scale_factor=0.4)
        
        # Sampler icons
        sampler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sampler.svg")
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        sampler_combined = VGroup(sampler, box)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(display_group))
        self.lecture[0].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        self.place_at_grid(sampler_combined, 'E5', scale_factor=0.6)
        self.play(FadeIn(sampler_combined))
        self.play(sampler_combined.animate.move_to(self.grid['D3']), run_time=1.5)
        self.wait(1)
