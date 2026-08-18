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
        self.setup_layout("The Conceptual Bridge: Shrinking the Interval", [
            "What happens if the interval gets smaller?",
            "The secant line approaches the tangent line.",
            "As h nears zero, we find instantaneous rate."
        ])
        
        # Grid positioning as requested
        self.place_at_grid(self.title, 'A2', scale_factor=1.0)
        self.place_at_grid(self.lecture, 'A1', scale_factor=0.9)
        
        # Load asset
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        asset_icon = SVGMobject(asset_path).scale(0.5)

        # Define function and points
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2 * x**3, color=BLUE)
        
        a = 1
        h = ValueTracker(2)
        
        dot1 = Dot(color=RED)
        dot2 = Dot(color=GREEN)
        
        def update_dots(d):
            val = h.get_value()
            dot1.move_to(axes.c2p(a, 0.2 * a**3))
            dot2.move_to(axes.c2p(a + val, 0.2 * (a + val)**3))
            
        dot1.add_updater(update_dots)
        dot2.add_updater(update_dots)
        
        secant = always_redraw(lambda: Line(dot1.get_center(), dot2.get_center(), color=YELLOW))
        
        # Position axes
        self.place_in_area(axes, 'B2', 'F6', scale_factor=0.6)
        
        self.add(axes, curve, dot1, dot2, secant, asset_icon.next_to(axes, UP))

        # Animate lecture reveal
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 1 ===
        self.play(h.animate.set_value(0.5), run_time=2)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_opacity(1)
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(h.animate.set_value(0.1), run_time=2)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_opacity(1)
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        self.play(h.animate.set_value(0.01), run_time=2)
        self.wait(1)
