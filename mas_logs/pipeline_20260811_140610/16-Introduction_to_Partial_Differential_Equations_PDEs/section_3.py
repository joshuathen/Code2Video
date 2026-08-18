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
        lecture_lines = [
            "The heat equation models how energy spreads.",
            "Visualize heat diffusing through a metal bar.",
            "The center cools as heat moves outward."
        ]
        self.setup_layout("The Heat Equation: Modeling Diffusion", lecture_lines)
        
        # Use assets as per storyboard/instruction
        bar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg")
        metal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metal.svg")
        
        # Setup heat distribution graph elements
        self.t = ValueTracker(0.1)
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 2, 0.5], axis_config={"include_tip": False})
        # Use curve_updater for animation
        curve = axes.plot(lambda x: 1.5 * np.exp(-x**2 / (0.2 + self.t.get_value())), color="#4287F5")
        
        # Graph Group
        graph_group = VGroup(axes, curve)
        
        # Positioning in grid (using A4-D6 for the graph)
        self.place_in_area(graph_group, 'A4', 'D6', scale_factor=0.6)
        
        # Heat flux vectors
        arrows = VGroup(*[Arrow(start=axes.c2p(x, 0.1), end=axes.c2p(x + 0.3*np.sign(x), 0.1), buff=0, color=RED) for x in [-1.5, 1.5]])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: .../bar.svg]
        self.place_at_grid(bar_icon, 'B4', scale_factor=0.5)
        self.play(FadeIn(bar_icon), FadeIn(graph_group))
        self.lecture[0].set_color("#4287F5")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF9F00")
        
        # Update curve color and position/data
        new_curve = always_redraw(lambda: axes.plot(lambda x: 1.5 * np.exp(-x**2 / (0.2 + self.t.get_value())), color="#FF9F00"))
        self.play(ReplacementTransform(curve, new_curve), self.t.animate.set_value(2.0), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF9F00")
        
        # Transform into [Asset: .../metal.svg]
        self.place_at_grid(metal_icon, 'D6', scale_factor=0.5)
        self.play(FadeIn(arrows), FadeOut(bar_icon), ReplacementTransform(new_curve, metal_icon), self.t.animate.set_value(5.0), run_time=3)
        self.wait(1)
