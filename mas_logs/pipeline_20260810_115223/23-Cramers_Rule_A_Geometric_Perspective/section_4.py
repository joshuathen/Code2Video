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
        self.setup_layout("The Visual Mechanism", [
            "Vary target b shifts areas.",
            "Observe how areas expand and contract.",
            "Scaling factor yields the solution."
        ])
        
        # Asset Loading (placeholder icons as specified)
        icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        asset_1 = SVGMobject(icon_path)
        asset_2 = SVGMobject(icon_path)
        
        # Grid setup (reduced scale per B048)
        grid = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#888888", "stroke_width": 1}
        )
        self.place_in_area(grid, 'B2', 'E5', scale_factor=0.5)
        self.add(grid)
        
        b = ValueTracker(1.0)
        
        # Parallelogram elements
        # Using persistent mobjects + updaters per constraint 10/11
        poly = Polygon(
            [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
            color="#FF00FF", fill_opacity=0.3
        )
        def update_para(p):
            val = b.get_value()
            p.become(Polygon(
                [0, 0, 0], [2, 0, 0], [2 + 0.5 * val, 1 * val, 0], [0.5 * val, 1 * val, 0],
                color="#FF00FF", fill_opacity=0.3
            ).scale(0.5).move_to(grid.get_center()))
            
        poly.add_updater(update_para)
        self.add(poly)
        
        # Blue dot marker
        dot = Dot(color="#00FFFF")
        def update_dot(d):
            d.move_to(grid.c2p(0.5 * b.get_value(), 1 * b.get_value()))
        dot.add_updater(update_dot)
        self.add(dot)

        # Apply Critic/Asset positioning fixes
        self.place_in_area(poly, 'D2', 'F5', scale_factor=0.5) # Fix 29
        self.place_at_grid(dot, 'E4', scale_factor=0.4)       # Fix 30
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(b.animate.set_value(2.0), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(b.animate.set_value(0.5), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(b.animate.set_value(1.5), run_time=1)
