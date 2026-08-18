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
        self.setup_layout("Prerequisite Review: Rates & Accumulation", [
            "Derivatives measure the slope of a curve.",
            "Integrals calculate the area under a curve.",
            "They are inverse operations of each other."
        ])

        # Assets
        slope_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slope.svg")
        area_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/area.svg")

        # Visualization container
        axes = Axes(x_range=[-1, 5], y_range=[-1, 4], axis_config={"include_tip": True})
        func = lambda x: 0.1 * x**3 - 0.5 * x**2 + x + 1
        curve = axes.plot(func, color=BLUE)
        grid_container = VGroup(axes, curve)
        
        # Apply positioning constraints
        self.place_in_area(grid_container, 'B4', 'F6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_container), run_time=0.5)
        self.place_at_grid(slope_icon, 'B5', scale_factor=0.6)
        slope_icon.set_color("#FF0000")
        self.play(FadeIn(slope_icon), self.lecture[0].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        area = axes.get_area(curve, x_range=[0, 3], color="#00FF00", opacity=0.3)
        self.place_at_grid(area_icon, 'D5', scale_factor=0.6)
        area_icon.set_color("#00FF00")
        self.play(FadeOut(slope_icon), FadeIn(area), FadeIn(area_icon), self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(area), self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)
