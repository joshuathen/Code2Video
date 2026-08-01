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
        title = "The Mystery of Turbulent Flow"
        lines = [
            "Turbulence transforms smooth, laminar flow into chaotic motion.",
            "It's not just random noise; it's a multiscale phenomenon.",
            "Chaotic fluids actually follow strict statistical laws."
        ]
        self.setup_layout(title, lines)

        # Asset Paths
        fluid_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg"
        eddy_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg"

        # Colors
        laminar_color = "#ADD8E6"
        highlight_color = "#FFFFE0"

        # === Animation for Lecture Line 1 ===
        # "Turbulence transforms smooth, laminar flow into chaotic motion."
        self.play(self.lecture[0].animate.set_color(laminar_color))

        # 1. Laminar Flow
        fluid_asset = SVGMobject(fluid_path).set_color(laminar_color)
        self.place_in_area(fluid_asset, "A3", "C5", scale_factor=1.5)
        
        laminar_label = Text("Laminar Flow", font_size=20, color=WHITE)
        self.place_at_grid(laminar_label, "B2", scale_factor=0.8) # Fix Issue 21: Moved from B1 to B2
        
        self.play(FadeIn(fluid_asset), Write(laminar_label))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "It's not just random noise; it's a multiscale phenomenon."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(laminar_color)
        )

        # 2. Transition to Turbulent Flow
        turbulent_label = Text("Turbulent Flow", font_size=20, color=WHITE)
        self.place_at_grid(turbulent_label, "E2", scale_factor=0.8) # Fix Issue 22: Moved from E1 to E2

        # Large Eddies (Turbulent Flow)
        eddy_centers = ["D3", "D5", "E4", "F3", "F5"]
        large_eddies = VGroup(*[
            SVGMobject(eddy_path).set_color(laminar_color).scale(0.4).move_to(self.grid[c])
            for c in eddy_centers
        ])

        self.play(
            FadeOut(fluid_asset),
            FadeOut(laminar_label),
            FadeIn(large_eddies),
            Write(turbulent_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "Chaotic fluids actually follow strict statistical laws."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color)
        )

        # 3. Multiscale Phenomenon / Highlight
        # Select one eddy for focus
        focus_eddy = large_eddies[2] # E4
        
        # Add small eddies around the focus eddy
        small_eddies = VGroup(
            SVGMobject(eddy_path).set_color(laminar_color).scale(0.15).move_to(focus_eddy.get_center() + UP * 0.2 + RIGHT * 0.2),
            SVGMobject(eddy_path).set_color(laminar_color).scale(0.15).move_to(focus_eddy.get_center() + DOWN * 0.2 + LEFT * 0.2)
        )

        flash = Flash(focus_eddy.get_center(), color=highlight_color, line_length=0.3, num_lines=12, flash_radius=0.5)
        highlight_circle = Circle(radius=0.5, color=highlight_color, stroke_width=2).move_to(focus_eddy.get_center())

        self.play(
            focus_eddy.animate.set_color(highlight_color).scale(1.2),
            FadeIn(small_eddies),
            Create(highlight_circle),
            flash,
            run_time=2
        )
        self.play(FadeOut(highlight_circle), run_time=0.5)
        
        self.wait(3)

        # Reset colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
