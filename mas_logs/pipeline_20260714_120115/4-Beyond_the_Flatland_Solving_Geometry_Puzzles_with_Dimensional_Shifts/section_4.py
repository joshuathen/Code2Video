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
        # Lecture lines from storyboard
        lecture_lines = [
            "Linked rings appear fused in two dimensions.",
            "They cannot be separated without breaking.",
            "Adding a Z-axis provides a new direction.",
            "We lift one ring into the third dimension.",
            "This extra degree of freedom separates the rings."
        ]
        
        self.setup_layout("Puzzle 2: The Impossible Knot (2D to 3D Shift)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show two overlapping 2D rings [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rings.svg], white #FFFFFF and blue #0000FF.
        # Position: ring1 at C4, ring2 at C5 for better balance (Issue 63)
        ring_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rings.svg"
        ring1 = SVGMobject(ring_path).set_color(WHITE)
        ring2 = SVGMobject(ring_path).set_color(BLUE)
        
        self.place_at_grid(ring1, 'C4')
        self.place_at_grid(ring2, 'C5')
        
        # Scale them slightly to ensure they overlap visibly while at C4 and C5 (centers 1 unit apart)
        # SVGMobjects usually have height/width around 2. Scaling by 0.7 makes them roughly 1.4.
        ring1.scale(0.8)
        ring2.scale(0.8)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            FadeIn(ring1),
            FadeIn(ring2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The overlap points flash red #FF0000 to show "stuck."
        # Intersection between C4 (3.5, 0.2) and C5 (4.5, 0.2)
        # We can place dots at approx x=4.0
        dot_top = Dot(point=[4.0, 0.6, 0], color=RED, radius=0.1)
        dot_bottom = Dot(point=[4.0, -0.2, 0], color=RED, radius=0.1)
        
        self.play(
            self.lecture[1].animate.set_color(RED),
            FadeIn(dot_top),
            FadeIn(dot_bottom),
            run_time=1.0
        )
        self.play(Flash(dot_top, color=RED), Flash(dot_bottom, color=RED))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Introduce a green #00FF00 arrow for the Z-axis depth.
        # Fix layout: place z_axis_group at E6, scale_factor=0.8 (Issue 63)
        z_start = self.grid['E6']
        z_end = z_start + np.array([0.5, 0.5, 0])
        z_arrow = Arrow(z_start, z_end, color=GREEN, buff=0)
        z_label = Text("Z", font_size=20, color=GREEN).next_to(z_end, UR, buff=0.1)
        z_axis_group = VGroup(z_arrow, z_label)
        # Already used grid for start, so just scaling the group
        z_axis_group.scale(0.8)
        
        self.play(
            self.lecture[2].animate.set_color(GREEN),
            FadeIn(z_axis_group),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Lift the blue ring [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rings.svg] along the Z-axis.
        # Lift along the same direction as the Z-axis arrow
        lift_vector = np.array([0.3, 0.3, 0])
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            ring2.animate.shift(lift_vector).scale(1.1), # Scale up to show depth
            FadeOut(dot_top),
            FadeOut(dot_bottom),
            run_time=2.0
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Move the rings [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rings.svg] apart, resolving the intersection.
        # Fix layout: ring1 at B4, ring2 at D5 (Issue 63)
        # We need to use place_at_grid but within an animation. 
        # We'll calculate the target positions from the grid.
        
        target_pos1 = self.grid['B4']
        target_pos2 = self.grid['D5']
        
        self.play(
            self.lecture[4].animate.set_color(BLUE),
            ring1.animate.move_to(target_pos1),
            ring2.animate.move_to(target_pos2).scale(1/1.1), # Scale back to original
            FadeOut(z_axis_group),
            run_time=2.0
        )
        self.wait(2)
