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
        lecture_lines = [
            "A cat walking across hilly terrain.",
            "Elevation change depends on movement direction.",
            "Slicing maps out partial derivatives."
        ]
        self.setup_layout("The 'Snapshot' Visualization: Partial Derivatives", lecture_lines)
        
        # Load asset
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        # --- Prepare objects ---
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        surface_curve = axes.plot(lambda x: 0.2 * (x**2), color="#00FF00")
        surface_group = VGroup(axes, surface_curve)
        
        # Line 1: Surface
        self.place_in_area(surface_group, 'C3', 'E5', scale_factor=0.5)
        cat_1 = cat_icon.copy()
        self.place_at_grid(cat_1, 'C3', scale_factor=0.1)

        # Line 2: Axes and plane
        axis_system = axes.copy()
        self.place_at_grid(axis_system, 'D4', scale_factor=0.7)
        plane = Line(start=self.grid['C3'] + DOWN*0.5, end=self.grid['C5'] + UP*0.5, color="#FF00FF")

        # Line 3: Curve and cat
        intersection_curve = axes.plot(lambda x: 0.2 * (x**2), color="#FFFF00")
        graph_labels = VGroup(intersection_curve)
        self.place_in_area(graph_labels, 'C4', 'D5', scale_factor=0.6)
        cat_2 = cat_icon.copy()
        self.place_at_grid(cat_2, 'D5', scale_factor=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(surface_group), FadeIn(cat_1))
        self.lecture[0].set_color("#00FF00")
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(plane))
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        self.play(Create(graph_labels), FadeIn(cat_2))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
