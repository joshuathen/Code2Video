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
        self.setup_layout("Visualizing Integration as Accumulation", [
            "Integration measures the total area under a curve.",
            "Visualize this as summing infinite thin rectangles.",
            "Bar graphs transform into a smooth curve."
        ])
        
        # Create axes and function
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        func = axes.plot(lambda t: 0.5 * t + 0.5, color=WHITE)
        
        # Fix: Move axes to area B2-E5 with scale 0.5 as requested by Orchestrator
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.5)
        func.match_y(axes)
        
        # Asset for sweeping bar
        sweeping_bar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg")
        self.place_at_grid(sweeping_bar, 'B2', scale_factor=0.3)
        sweeping_bar.set_opacity(0)
        self.add(sweeping_bar)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        area = axes.get_area(func, x_range=[0, 4], color="#FFFF00", opacity=0.3)
        self.play(Create(axes), Create(func), FadeIn(area))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        # Visualizing summing rectangles
        rects = axes.get_riemann_rectangles(func, x_range=[0, 4], dx=0.5, color=BLUE, fill_opacity=0.5)
        
        # Use the asset to sweep across the area
        self.play(
            FadeOut(area), 
            FadeIn(sweeping_bar.set_opacity(1)),
            Create(rects),
            sweeping_bar.animate.shift(RIGHT * 3)
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        # Transforming to curve
        smooth_area = axes.get_area(func, x_range=[0, 4], color="#FF00FF", opacity=0.5)
        self.play(FadeOut(sweeping_bar), ReplacementTransform(rects, smooth_area))
        self.wait(2)
