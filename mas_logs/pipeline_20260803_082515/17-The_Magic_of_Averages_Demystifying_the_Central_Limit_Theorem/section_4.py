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
        self.setup_layout(
            "The Core Reveal: The Shape-Shifting Property", 
            [
                "Start with any population shape, even highly skewed ones.",
                "As sample size n increases, something magical happens.",
                "The distribution of sample means begins to shift shape.",
                "It always settles into a symmetrical normal distribution.",
                "This is the core of the Central Limit Theorem."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Color: Magenta (#FF00FF)
        self.lecture[0].set_color("#FF00FF")
        
        # Shark Fin Population Shape using Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/shark.svg]
        shark_fin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shark.svg")
        shark_fin.set_color("#FF00FF")
        self.place_in_area(shark_fin, "A4", "B6", scale_factor=0.8)
        
        pop_label = Text("Population", font_size=20, color="#FF00FF")
        self.place_at_grid(pop_label, "A2", scale_factor=1.0)
        
        self.play(DrawBorderThenFill(shark_fin), Write(pop_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color: Gray (#AAAAAA)
        self.lecture[1].set_color("#AAAAAA")
        
        # Jagged Plot for small n=2
        # Area moved to D4-F6 (Issue 32)
        jagged_points = [
            [-2, 0, 0], [-1.5, 0.3, 0], [-1.1, 0.9, 0], [-0.7, 0.5, 0], 
            [-0.2, 1.4, 0], [0.3, 0.6, 0], [0.8, 1.1, 0], [1.4, 0.4, 0], [2, 0, 0]
        ]
        jagged_plot = VMobject(color="#AAAAAA").set_points_as_corners([np.array(p) for p in jagged_points])
        self.place_in_area(jagged_plot, "D4", "F6", scale_factor=0.8)
        
        n_label = Text("n = 2", font_size=20, color=WHITE)
        self.place_at_grid(n_label, "D2", scale_factor=1.0)
        
        self.play(Create(jagged_plot), Write(n_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color: White (#FFFFFF)
        self.lecture[2].set_color("#FFFFFF")
        
        # n increases to 30
        n30_label = Text("n = 30", font_size=20, color=WHITE)
        self.place_at_grid(n30_label, "D2", scale_factor=1.0)
        
        # Medium smoothing
        medium_points = [
            [-2, 0, 0], [-1.2, 0.5, 0], [-0.6, 1.3, 0], [0, 1.6, 0], 
            [0.6, 1.3, 0], [1.2, 0.5, 0], [2, 0, 0]
        ]
        medium_plot = VMobject(color="#AAAAAA").set_points_smoothly([np.array(p) for p in medium_points])
        self.place_in_area(medium_plot, "D4", "F6", scale_factor=0.8)

        self.play(Transform(n_label, n30_label), Transform(jagged_plot, medium_plot))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color: Cyan (#00FFFF)
        self.lecture[3].set_color("#00FFFF")
        
        # Smooth Bell Curve
        bell_curve = FunctionGraph(
            lambda x: 2.2 * np.exp(-x**2),
            x_range=[-2.2, 2.2],
            color="#00FFFF"
        )
        self.place_in_area(bell_curve, "D4", "F6", scale_factor=0.8)
        
        self.play(Transform(jagged_plot, bell_curve))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Color: Green (#00FF00)
        self.lecture[4].set_color("#00FF00")
        
        # Issue 30: CLT Text positioning
        clt_text = Text("Central Limit Theorem", font_size=22, color=WHITE)
        self.place_in_area(clt_text, 'C1', 'C3', scale_factor=0.8)
        
        # Issue 31: Normal Distribution text positioning
        normal_text = Text("Normal Distribution", font_size=22, color="#00FF00")
        self.place_in_area(normal_text, 'F1', 'F3', scale_factor=0.8)
        
        self.play(Write(clt_text), Write(normal_text))
        self.wait(2)
