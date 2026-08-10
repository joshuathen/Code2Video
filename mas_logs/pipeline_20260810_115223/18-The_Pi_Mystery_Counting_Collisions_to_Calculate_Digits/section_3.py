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
        self.setup_layout("Mapping Physics to Geometry: The Phase Space", [
            "Map velocities to a phase plane.", 
            "Collisions trace a circular arc.", 
            "The path reflects against boundaries.", 
            "Geometric paths encode the collisions."
        ])
        
        # Define objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True}).scale(0.5)
        phase_space_grid = VGroup(axes)
        
        # Assets
        particle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        wall_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        barrier_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barrier.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.place_in_area(phase_space_grid, "C3", "F5", scale_factor=0.8)
        self.add(phase_space_grid)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        arc = Arc(radius=1.5, start_angle=0, angle=PI/2, color="#00FFFF")
        self.place_at_grid(arc, "D3", scale_factor=0.5)
        self.place_at_grid(particle_icon, "D4", scale_factor=0.5)
        self.play(Create(arc), FadeIn(particle_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF4500")
        rect = Rectangle(width=1, height=1, color="#FF4500", fill_opacity=0.3)
        self.place_at_grid(rect, "D4", scale_factor=0.8)
        self.place_at_grid(wall_icon, "E4", scale_factor=0.5)
        self.play(FadeIn(rect), FadeIn(wall_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FF00")
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, "C5", scale_factor=0.7)
        self.place_at_grid(barrier_icon, "C6", scale_factor=0.5)
        self.play(dot.animate.move_to(self.grid["E5"]), FadeIn(barrier_icon), run_time=2)
        self.wait(1)
