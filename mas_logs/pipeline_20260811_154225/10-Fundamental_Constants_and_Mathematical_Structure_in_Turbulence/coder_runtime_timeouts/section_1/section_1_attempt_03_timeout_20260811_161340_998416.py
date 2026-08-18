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
        self.setup_layout("Introduction: The Chaos in the Fluid", [
            "Turbulence is a complex, multi-scale energy process. [Asset: A_01]", 
            "Reynolds number compares inertial to viscous forces. [Asset: A_02]", 
            "Math describes chaotic flow within this regime. [Asset: A_03]"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create a large fluid block labeled 'Fluid'
        fluid_rect = Rectangle(width=4, height=4, color=WHITE)
        fluid_label = Text("Fluid", font_size=24, color=WHITE)
        fluid_group = VGroup(fluid_rect, fluid_label).arrange(UP)
        self.place_in_area(fluid_group, "A3", "D6", scale_factor=0.7)
        self.play(FadeIn(fluid_group))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Animate turbulent vortices using asset
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg
        vortex_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg")
        vortices = VGroup(*[vortex_asset.copy().scale(0.3).move_to(fluid_rect.get_center() + np.random.uniform(-1, 1, 3)) for _ in range(5)])
        self.add(vortices)
        self.play(Rotating(vortices, about_point=fluid_rect.get_center(), angle=2*PI, run_time=2))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Highlight vortices
        self.play(vortices.animate.set_color("#FF00FF"))
        vortices_label = Text("Vortices", font_size=20, color="#FF00FF")
        self.place_at_grid(vortices_label, "D3", scale_factor=0.9)
        self.play(Write(vortices_label))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)
