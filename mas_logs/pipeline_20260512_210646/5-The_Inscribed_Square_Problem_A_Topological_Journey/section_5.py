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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        title = "The Core Logic: Why One Surface Must Intersect Another"
        lines = [
            'Every surface spanning this boundary must intersect itself.', 
            'These intersection points represent two pairs sharing properties.', 
            "If the properties match, we've found our rectangle.", 
            'The twist in the Möbius strip forces this overlap.', 
            'Topology guarantees these specific points must exist.'
        ]
        self.setup_layout(title, lines)

        # Define Colors
        ORCHID = "#DA70D6"
        WHITE = "#FFFFFF"
        ORANGE = "#FFA500"
        GREEN = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORCHID)
        
        # Representing a Mobius strip in 2D with a 'twist'
        # Top half
        mobius_top = CubicBezier(
            np.array([-1.5, 0, 0]), np.array([-1.5, 1, 0]), 
            np.array([1.5, 1, 0]), np.array([1.5, 0, 0]),
            color=ORCHID
        )
        # Bottom half with a visual twist (break in middle)
        mobius_bot_1 = CubicBezier(
            np.array([1.5, 0, 0]), np.array([1.5, -1, 0]), 
            np.array([0.2, -0.8, 0]), np.array([0, -0.2, 0]),
            color=ORCHID
        )
        mobius_bot_2 = CubicBezier(
            np.array([0, 0.2, 0]), np.array([-0.2, 0.8, 0]), 
            np.array([-1.5, -1, 0]), np.array([-1.5, 0, 0]),
            color=ORCHID
        )
        mobius_strip = VGroup(mobius_top, mobius_bot_1, mobius_bot_2)
        
        # Horizontal Plane (schematic)
        plane = Rectangle(width=4, height=0.2, fill_opacity=0.3, fill_color=WHITE, stroke_width=1)
        
        # Place them in the grid - Resolved Issues #46 and #47
        self.place_in_area(mobius_strip, "C3", "E6", scale_factor=0.8)
        self.place_in_area(plane, "F3", "F6", scale_factor=1.0)
        
        self.play(Create(mobius_strip), FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        
        # Intersection points
        intersect_pt1 = Dot(mobius_strip.get_center() + RIGHT*0.5, color=WHITE)
        intersect_pt2 = Dot(mobius_strip.get_center() + LEFT*0.5, color=WHITE)
        
        self.play(FadeIn(intersect_pt1), FadeIn(intersect_pt2))
        self.play(Flash(intersect_pt1, color=WHITE), Flash(intersect_pt2, color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        
        # Original Loop (the curve)
        loop = Circle(radius=1.0, color=ORCHID).set_points_smoothly([
            [1, 0, 0], [0.7, 0.7, 0], [0, 1.2, 0], [-0.8, 0.6, 0],
            [-1.1, 0, 0], [-0.7, -0.7, 0], [0, -1.1, 0], [0.8, -0.6, 0], [1, 0, 0]
        ])
        # Resolved Issue #48
        self.place_in_area(loop, "A3", "B6", scale_factor=0.8)
        
        # Rectangle points on loop
        p1 = loop.point_from_proportion(0.1)
        p2 = loop.point_from_proportion(0.35)
        p3 = loop.point_from_proportion(0.6)
        p4 = loop.point_from_proportion(0.85)
        
        rect_points = [p1, p2, p3, p4]
        rectangle = Polygon(*rect_points, color=ORANGE, stroke_width=4)
        
        # Arrow mapping intersection to the loop area
        mapping_arrow = Arrow(
            start=intersect_pt1.get_center(), 
            end=loop.get_center() + RIGHT*0.5, 
            buff=0.1, color=ORANGE
        )
        
        self.play(Create(loop))
        self.play(GrowArrow(mapping_arrow))
        self.play(Create(rectangle))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORCHID)
        
        # Emphasize the twist by pulsing the mobius strip
        self.play(mobius_strip.animate.set_stroke(width=8), run_time=0.5)
        self.play(mobius_strip.animate.set_stroke(width=4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        
        # Flash the rectangle green
        self.play(rectangle.animate.set_color(GREEN))
        self.play(Flash(rectangle, color=GREEN))
        self.wait(2)
