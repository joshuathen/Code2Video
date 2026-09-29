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
            "Continuous functions map spheres to Euclidean space.",
            "At least one pair maps to one value.",
            "This is the Borsuk-Ulam theorem.",
            "Visualize a circle mapping to a line.",
            "Opposite points land on the same spot."
        ]
        self.setup_layout("The Borsuk-Ulam Theorem (Formalized)", lecture_lines)
        
        # Setup assets
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        sphere = SVGMobject(sphere_asset, color=WHITE)
        label = Text("Sphere", font_size=20, color=WHITE).next_to(sphere, UP)
        
        line = Line(start=np.array([-1.5, 0, 0]), end=np.array([1.5, 0, 0]), color=WHITE)
        
        # Grouping for area placement
        visual_group = VGroup(sphere, label, line)
        self.place_in_area(visual_group, 'A4', 'F6', scale_factor=0.7)
        
        # Positioning overrides
        self.place_at_grid(sphere, 'C4', scale_factor=0.8)
        self.place_at_grid(line, 'D4', scale_factor=0.8)
        label.next_to(sphere, UP)

        p1 = Dot(color="#FF00FF")
        p2 = Dot(color="#FF00FF")
        p1.move_to(sphere.get_top())
        p2.move_to(sphere.get_bottom())
        
        self.add(sphere, label, p1, p2, line)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Mapping
        mapped_p1 = Dot(color="#00FF00").move_to(line.point_from_proportion(0.3))
        mapped_p2 = Dot(color="#00FF00").move_to(line.point_from_proportion(0.3))
        
        self.play(
            ReplacementTransform(p1.copy(), mapped_p1),
            ReplacementTransform(p2.copy(), mapped_p2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.play(p1.animate.set_color("#00FFFF"), p2.animate.set_color("#00FFFF"))
        self.wait(2)
