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
        lecture_lines = ["Necklace beads represent intervals.", "Continuous paths model discrete objects.", "Topology solves combinatorics tasks."]
        self.setup_layout("Transition: From Spheres to Necklaces", lecture_lines)
        
        # --- Visual Elements ---
        # Note: SVG assets aren't rendered in simple environments, using placeholder shapes
        # as requested in instructions to maintain simplicity.
        
        # Using SVG placeholder (as per instruction: "If provided, MUST use elements")
        necklace = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")
        beads = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beads.svg")
        
        # Corrected positioning per issue 26
        self.place_at_grid(necklace, 'C4', scale_factor=0.6)
        
        # Line representation
        line = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color=WHITE)
        # Corrected positioning per issue 27
        self.place_in_area(line, 'C5', 'D6', scale_factor=0.7)
        line.set_opacity(0)
        
        # Markers
        markers = VGroup(*[Line(UP*0.2, DOWN*0.2, color=RED) for _ in range(4)])
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(necklace))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        # Transition the necklace into a flattened line
        self.play(
            FadeIn(line),
            necklace.animate.move_to(line.get_center()).set_opacity(0.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        # Show markers on the line
        for i, m in enumerate(markers):
            m.move_to(line.point_from_proportion(0.2 * (i+1)))
        self.play(Create(markers))
        self.wait(1)
