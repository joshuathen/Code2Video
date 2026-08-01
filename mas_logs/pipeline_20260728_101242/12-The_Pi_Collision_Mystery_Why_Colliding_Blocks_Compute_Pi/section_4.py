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
        # Setup layout with title and lecture lines
        self.setup_layout("Collisions as Geometric Reflections", [
            "A wall collision reflects the velocity point vertically.",
            "Block collisions reflect the point across a tilted line.",
            "Each collision moves the point along the circle's edge.",
            "The path traced looks like a light beam bouncing.",
            "Every 'clack' of blocks is a geometric reflection."
        ])
        
        # Colors based on storyboard
        DOT_COLOR = "#FF00FF"
        LINE_COLOR = "#FFA500"
        CIRCLE_COLOR = WHITE
        PATH_COLOR = "#FFFF00"

        # 1. Geometry setup (Circle and reference grid)
        # Issue 28: Scale up the circle to use the full A1-F6 area
        circle = Circle(radius=2.2, color=CIRCLE_COLOR) 
        self.place_in_area(circle, "A1", "F6", scale_factor=0.9)
        circle_center = circle.get_center()
        radius = circle.radius
        
        # Reference horizontal axis for the wall collision visualization
        h_axis = Line(circle_center + LEFT * radius, circle_center + RIGHT * radius, color=GREY, stroke_opacity=0.4)
        
        # === Animation for Lecture Line 1 ===
        # "A wall collision reflects the velocity point vertically."
        self.lecture[0].set_color(DOT_COLOR)
        
        start_angle = 45 * DEGREES
        dot_start_pos = circle_center + radius * np.array([np.cos(start_angle), np.sin(start_angle), 0])
        dot = Dot(dot_start_pos, color=DOT_COLOR)
        
        self.add(circle, h_axis)
        self.play(FadeIn(dot))
        
        # Reflect vertically across the horizontal axis (y -> -y relative to center)
        p1 = circle_center + radius * np.array([np.cos(start_angle), -np.sin(start_angle), 0])
        reflection_v = DashedLine(dot_start_pos, p1, color=DOT_COLOR, stroke_width=2)
        
        self.play(Create(reflection_v))
        self.play(dot.animate.move_to(p1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Block collisions reflect the point across a tilted line."
        self.lecture[1].set_color(LINE_COLOR)
        
        # Define the tilted line (reflection boundary for mass-ratio dependent collisions)
        angle_tilted = 30 * DEGREES
        # Issue 29: Ensure tilted line is bounded within the circle to avoid clutter
        tilted_line = Line(
            circle_center - radius * 1.05 * np.array([np.cos(angle_tilted), np.sin(angle_tilted), 0]),
            circle_center + radius * 1.05 * np.array([np.cos(angle_tilted), np.sin(angle_tilted), 0]),
            color=LINE_COLOR
        )
        
        # Reflection logic: P' = P - 2*(P.n)*n where n is the normal to the line
        n = np.array([-np.sin(angle_tilted), np.cos(angle_tilted), 0])
        p_rel = p1 - circle_center
        p2 = circle_center + (p_rel - 2 * np.dot(p_rel, n) * n)
        
        reflection_t = DashedLine(p1, p2, color=LINE_COLOR, stroke_width=2)
        
        self.play(Create(tilted_line))
        self.play(Create(reflection_t))
        self.play(dot.animate.move_to(p2))
        self.wait(1)
        
        # Clean up helpers for the tracing phase
        self.play(FadeOut(reflection_v), FadeOut(reflection_t), FadeOut(h_axis))

        # === Animation for Lecture Line 3 ===
        # "Each collision moves the point along the circle's edge."
        self.lecture[2].set_color(PATH_COLOR)
        
        # Start tracing the path from the current position
        # First, add the history (from start to p1 to p2)
        history = VMobject(color=PATH_COLOR, stroke_width=2)
        history.set_points_as_corners([dot_start_pos, p1, p2])
        self.add(history)
        
        trace = TracedPath(dot.get_center, stroke_color=PATH_COLOR, stroke_width=2)
        self.add(trace)
        
        # Perform one more wall collision (vertical reflection)
        p_rel_2 = p2 - circle_center
        p3 = circle_center + np.array([p_rel_2[0], -p_rel_2[1], 0])
        
        self.play(dot.animate.move_to(p3), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "The path traced looks like a light beam bouncing."
        self.lecture[3].set_color(PATH_COLOR)
        
        # Two more bounces to show the pattern
        # Bounce 4 (Block collision)
        p_rel_3 = p3 - circle_center
        p4 = circle_center + (p_rel_3 - 2 * np.dot(p_rel_3, n) * n)
        
        # Bounce 5 (Wall collision)
        p_rel_4 = p4 - circle_center
        p5 = circle_center + np.array([p_rel_4[0], -p_rel_4[1], 0])

        self.play(dot.animate.move_to(p4), run_time=0.7)
        self.play(dot.animate.move_to(p5), run_time=0.7)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # "Every 'clack' of blocks is a geometric reflection."
        self.lecture[4].set_color(WHITE)
        
        # Rapid series of reflections to simulate the high frequency 'clacks'
        p_curr = p5
        for _ in range(4):
            # Reflect across tilted line
            pr = p_curr - circle_center
            pn = circle_center + (pr - 2 * np.dot(pr, n) * n)
            self.play(dot.animate.move_to(pn), run_time=0.3)
            p_curr = pn
            
            # Reflect across horizontal line
            pr = p_curr - circle_center
            pn = circle_center + np.array([pr[0], -pr[1], 0])
            self.play(dot.animate.move_to(pn), run_time=0.3)
            p_curr = pn

        self.wait(2)
