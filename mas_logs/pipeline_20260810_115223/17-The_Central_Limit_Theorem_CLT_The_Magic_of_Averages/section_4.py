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
        self.setup_layout("Real-World Application: Why This Matters", [
            "CLT lets us estimate population parameters accurately.",
            "It enables reliable hypothesis testing and confidence intervals.",
            "Engineers use it to infer quality from samples."
        ])
        
        # --- Create Visuals ---
        # 1. Chaotic stock data (erratic lines)
        axes = Axes(x_range=[0, 10, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        data = FunctionGraph(lambda x: np.sin(x**2) + np.random.normal(0, 0.2), color="#FF3333")
        chart = VGroup(axes, data)
        # Applying fix for Issue 34, 36: D2 to F6, scale 0.4
        self.place_in_area(chart, "D2", "F6", scale_factor=0.4)
        
        # 2. Bell Curve for CLT
        # Applying fix for Issue 35: C4 to E6, scale 0.5
        bell = FunctionGraph(lambda x: 1.5 * np.exp(-x**2), x_range=[-2, 2], color="#33FF33")
        self.place_in_area(bell, "C4", "E6", scale_factor=0.5)
        bell.set_opacity(0)
        
        # 3. Engineering Icon
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        # Apply B045: emphasis scale 0.8-1.2
        self.place_at_grid(gear, "B5", scale_factor=1.0)
        gear.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(Create(chart))
        self.lecture[0].set_color("#FF3333")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(bell))
        self.lecture[1].set_color("#33FF33")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(gear))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
