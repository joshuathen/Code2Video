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
        # Set seed for deterministic randomness
        np.random.seed(42)
        
        title = "Prerequisite: The Concept of a Population vs. Sample"
        lines = [
            "A population represents every individual in a group.",
            "A sample is just a small portion of them.",
            "We describe groups using means and standard deviations."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_POP = WHITE
        COLOR_SAMPLE = "#58C4DD"
        COLOR_DIST = "#FFFF00"
        
        # Asset Path
        PERSON_SVG = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/people.svg"

        # === Animation for Lecture Line 1 ===
        # Visual: Draw a large cluster of white dots representing 'Population'.
        # Using SVGMobject as requested by storyboard [Asset: people.svg]
        
        population_group = VGroup()
        for _ in range(30):
            p = SVGMobject(PERSON_SVG)
            p.set_color(COLOR_POP)
            p.scale(0.15)
            population_group.add(p)
            
        # Randomly position within area B3-D5 (Issue 37 fix)
        self.place_in_area(population_group, "B3", "D5", scale_factor=1.0)
        for p in population_group:
            p.shift(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]))

        pop_label = Text("Population", font_size=24, color=COLOR_POP)
        self.place_at_grid(pop_label, "A3", scale_factor=0.8) # Issue 36 fix

        self.play(
            self.lecture[0].animate.set_color(COLOR_POP),
            LaggedStart(*[FadeIn(p) for p in population_group], lag_ratio=0.05),
            Write(pop_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: Circle a small subset of dots (#58C4DD) and label 'Sample'.
        
        self.lecture[0].set_color(WHITE)
        
        # Select a subset of the population (e.g., indices 5, 12, 18, 22, 27)
        sample_indices = [5, 12, 18, 22, 27]
        sample_mobjects = VGroup(*[population_group[i] for i in sample_indices])
        
        # Create a circle around the sample center
        sample_center = sample_mobjects.get_center()
        sample_circle = Circle(radius=0.8, color=COLOR_SAMPLE, stroke_width=4)
        sample_circle.move_to(sample_center)
        
        sample_label = Text("Sample", font_size=24, color=COLOR_SAMPLE)
        self.place_at_grid(sample_label, "C3", scale_factor=0.8) # Issue 36 fix
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_SAMPLE),
            sample_mobjects.animate.set_color(COLOR_SAMPLE),
            Create(sample_circle),
            Write(sample_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: Transform the cluster into a jagged, bimodal distribution curve (#FFFF00).
        # Plus mu and sigma labels.
        
        self.lecture[1].set_color(WHITE)
        
        # Define Axes for the distribution
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=2,
            axis_config={"include_tip": False, "color": COLOR_DIST}
        )
        self.place_in_area(axes, "D3", "F5", scale_factor=0.8)
        
        # Jagged bimodal distribution curve
        bimodal_graph = axes.plot(
            lambda x: 1.5 * np.exp(-0.5 * (x - 3)**2) + 2.5 * np.exp(-0.5 * (x - 7)**2) + 0.3 * np.sin(10 * x),
            x_range=[0.5, 9.5],
            color=COLOR_DIST
        )
        
        # mu and sigma labels (Issue 38 fix)
        mu_label = Text("μ (Mean)", font_size=22, color=COLOR_DIST)
        sigma_label = Text("σ (Std Dev)", font_size=22, color=COLOR_DIST)
        self.place_in_area(mu_label, "E1", "E2", scale_factor=0.8)
        self.place_in_area(sigma_label, "F1", "F2", scale_factor=0.8)

        self.play(
            self.lecture[2].animate.set_color(COLOR_DIST),
            ReplacementTransform(population_group, bimodal_graph),
            FadeOut(sample_circle),
            FadeOut(pop_label),
            FadeOut(sample_label),
            Create(axes),
            Write(mu_label),
            Write(sigma_label),
            run_time=2
        )
        self.wait(2)
