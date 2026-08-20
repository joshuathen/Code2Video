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
            "We plot velocities as coordinates on a graph.",
            "Collision paths trace a perfect circular arc.",
            "Physics transforms into beautiful, predictable geometry.",
            "The 'Aha!' moment connects motion to circles.",
            "Velocity arcs map the collision states."
        ]
        self.setup_layout("The Geometry of Phase Space", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": True}).scale(0.4)
        # Apply fix for issue 25/36: Change area to A3-D6
        self.place_in_area(axes, "A3", "D6", scale_factor=0.6)
        self.play(Create(axes), self.lecture[0].animate.set_color("#8A2BE2"))
        
        # === Animation for Lecture Line 2 ===
        # Apply fix for issue 17: Load particle asset
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        arc = Arc(radius=1.5, start_angle=0, angle=PI/2, color="#FF1493")
        # Apply fix for issue 25/36
        self.place_in_area(arc, "A3", "D6", scale_factor=0.6)
        self.place_at_grid(particle, "A3", scale_factor=0.2) 
        self.play(Create(arc), MoveAlongPath(particle, arc), self.lecture[1].animate.set_color("#FF1493"))
        
        # === Animation for Lecture Line 3 ===
        sector = Sector(radius=1.5, angle=PI/2, color="#00FFFF", fill_opacity=0.3)
        # Apply fix for issue 25/36
        self.place_in_area(sector, "A3", "D6", scale_factor=0.6)
        self.play(FadeIn(sector), self.lecture[2].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 4 ===
        label = Text("Aha!", font_size=24, color=YELLOW)
        # Apply fix for issue 26/36: E4
        self.place_at_grid(label, "E4", scale_factor=0.9 * 0.8) # Adjust scale as per B020
        self.play(Write(label), self.lecture[3].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 5 ===
        dot = Dot(color=WHITE)
        # Apply fix for issue 27/36: B5
        self.place_at_grid(dot, "B5", scale_factor=0.7)
        self.play(GrowFromCenter(dot), self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
