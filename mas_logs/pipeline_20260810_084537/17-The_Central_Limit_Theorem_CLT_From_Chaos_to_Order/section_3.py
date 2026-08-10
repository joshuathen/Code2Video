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
        self.setup_layout("The Experiment: Building the Sampling Distribution", [
            "Take many samples of size n.",
            "Calculate the mean for each sample.",
            "Plot means as size n increases.",
            "Observe the shape transforming into curves.",
            "Bell shapes emerge from chaotic data."
        ])

        # Assets
        population_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        curve_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg")

        # Animation Setup
        population = VGroup(population_icon)
        self.place_in_area(population, 'A4', 'C6', scale_factor=0.4)

        axes = Axes(x_range=[0, 10, 1], y_range=[0, 5, 1], axis_config={"include_tip": False})
        sampling_dist = VGroup(axes)
        self.place_in_area(sampling_dist, 'D4', 'F6', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(population))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        samples = VGroup(*[Dot(color="#FF4500") for _ in range(5)])
        self.play(FadeIn(samples.arrange(RIGHT)))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        dot = Dot(color="#FFD700")
        self.place_at_grid(dot, 'D3', scale_factor=0.6)
        self.play(Create(dot))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00CED1")
        self.play(Indicate(sampling_dist))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF69B4")
        self.play(FadeOut(samples), FadeOut(dot))
        # Final display simulation
        final_curve = curve_icon
        self.place_at_grid(final_curve, 'E5', scale_factor=0.6)
        final_curve.set_color("#f1c40f")
        self.play(FadeIn(final_curve))
        self.wait(2)
