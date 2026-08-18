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
        lecture_lines = [
            "Collisions relate to circular arc length.",
            "Arc length maps to the ratio pi.",
            "More mass yields more pi digits."
        ]
        self.setup_layout("The Geometric Connection", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg]
        circle = Circle(radius=1.5, color="#FFFFFF")
        self.place_in_area(circle, "B2", "F5", scale_factor=0.85)
        
        # Animate point moving along the circle. Color #FF0000.
        point = Dot(color="#FF0000")
        point.add_updater(lambda m: m.move_to(circle.point_from_proportion((self.time * 0.2) % 1)))
        
        # Incorporate asset for "blocks"
        blocks1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color="#FFFFFF")
        self.place_at_grid(blocks1, "A3", scale_factor=0.3)
        
        self.play(Create(circle), FadeIn(point), FadeIn(blocks1))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show normal vector to the circle. Color #00FF00.
        # Fade in arc showing collision angle. Color #0000FF.
        normal = Line(ORIGIN, UP*0.8, color="#00FF00")
        normal.add_updater(lambda m: m.put_start_and_end_on(circle.get_center(), circle.point_from_proportion((self.time * 0.2) % 1) + UP*0.5))
        
        arc = Arc(radius=0.5, start_angle=0, angle=PI/4, color="#0000FF")
        self.place_at_grid(arc, "C4", scale_factor=0.6)
        
        self.play(Create(normal), Create(arc))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Demonstrate mapping collision to circle geometry. Color #FFFF00.
        label = Text("collisions ~ pi digits", font_size=20, color="#FFFF00")
        self.place_at_grid(label, "E4", scale_factor=0.7)
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg]
        blocks2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color="#FFFF00")
        self.place_at_grid(blocks2, "F4", scale_factor=0.3)
        
        self.play(FadeIn(label), FadeIn(blocks2))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
