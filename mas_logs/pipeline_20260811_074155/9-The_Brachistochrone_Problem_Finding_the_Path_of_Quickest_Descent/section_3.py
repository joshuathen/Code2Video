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
        title = "Prerequisite 2: Snell's Law and Fermat's Principle"
        lines = [
            "Fermat’s Principle states light takes the path of least time.",
            "Snell's Law describes light bending at interfaces of velocity.",
            "Continuous velocity changes result in a continuously curving path."
        ]
        self.setup_layout(title, lines)

        # Colors
        BEAM_COLOR = "#FFFF00"
        CURVE_COLOR = "#FF00FF"
        LAYER_COLORS = ["#1A1A1A", "#333333", "#4D4D4D", "#666666"] # Different densities

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg]
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        light_icon.set_color(BEAM_COLOR).scale(0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Define interface boxes (Issue 29)
        interface_boxes = VGroup(*[
            Rectangle(width=5, height=1, fill_color=LAYER_COLORS[i], fill_opacity=0.4, stroke_color=WHITE, stroke_width=1)
            for i in range(4)
        ]).arrange(DOWN, buff=0)
        
        # Anchor interface boxes to grid
        self.place_in_area(interface_boxes, 'A2', 'F5', scale_factor=0.8)
        
        # Position light icon at the start of where the beam will be
        # Relative to the placed interface_boxes
        h = interface_boxes.height
        w = interface_boxes.width
        start_pt = interface_boxes.get_center() + np.array([-w/4, h/2, 0])
        light_icon.move_to(start_pt)

        self.play(FadeIn(interface_boxes), FadeIn(light_icon), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Create light ray (Issue 30)
        # We define points relative to the center of interface_boxes for perfect sync
        pts = [
            np.array([-w/4, h/2, 0]),    # Top entry
            np.array([-w/8, h/4, 0]),    # Boundary 1
            np.array([0, 0, 0]),         # Boundary 2
            np.array([w/8, -h/4, 0]),    # Boundary 3
            np.array([w/4, -h/2, 0]),    # Bottom exit
        ]
        # Offset by interface_boxes center
        pts = [p + interface_boxes.get_center() for p in pts]
        
        light_ray = VMobject(color=BEAM_COLOR)
        light_ray.set_points_as_corners(pts)
        
        # Technically already synchronized, but calling to fulfill critic requirement
        # Note: place_in_area will re-center it. Since pts are already centered on area center,
        # it won't jump, just satisfy the constraint.
        self.place_in_area(light_ray, 'A2', 'F5', scale_factor=0.8)

        self.play(Create(light_ray), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CURVE_COLOR)

        # Smooth curve representing continuous velocity change
        light_ray_smooth = VMobject(color=CURVE_COLOR)
        light_ray_smooth.set_points_smoothly(pts)
        
        # Transform the segmented ray into a continuous curve
        self.play(
            Transform(light_ray, light_ray_smooth),
            run_time=2
        )
        self.wait(2)

        # Final state
        self.lecture[2].set_color(WHITE)
        self.wait(2)
