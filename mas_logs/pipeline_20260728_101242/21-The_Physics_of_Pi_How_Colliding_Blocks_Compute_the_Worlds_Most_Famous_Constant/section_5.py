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
        # Title and Lecture Lines
        title_text = "Collisions as Geometric Reflections"
        lecture_lines = [
            "A block-block collision reflects the point across a line.",
            "A block-wall collision reflects it across the axis.",
            "The path bounces like light inside a circular mirror.",
            "Each collision adds one segment to the path.",
            "We simply count these geometric reflections."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Helper for reflection across a line passing through center with angle alpha
        def get_reflection(point, angle_rad, center):
            p = point - center
            # Normal vector to the line
            n = np.array([-np.sin(angle_rad), np.cos(angle_rad), 0])
            # Reflection formula: p' = p - 2(p.n)n
            p_prime = p - 2 * np.dot(p, n) * n
            return p_prime + center

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF69B4")
        
        # Draw a circle with a slanted 'collision line' (#FF69B4).
        circle = Circle(radius=1.8, color=WHITE)
        self.place_in_area(circle, "B2", "E4")
        center = circle.get_center()
        
        # Slanted line: alpha angle
        alpha = 25 * DEGREES
        collision_line = Line(
            center + 2.0 * np.array([-np.cos(alpha), -np.sin(alpha), 0]),
            center + 2.0 * np.array([np.cos(alpha), np.sin(alpha), 0]),
            color="#FF69B4"
        )
        collision_label = Text("Collision Line", font_size=16, color="#FF69B4")
        # Fixed: Issue 30 - use place_in_area for label
        self.place_in_area(collision_label, "B5", "B6", scale_factor=0.7)
        
        self.play(Create(circle))
        self.play(Create(collision_line), Write(collision_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#1E90FF")
        
        # Draw a horizontal 'wall line' (#1E90FF) and reflect the point across it.
        wall_line = Line(
            center + 2.0 * LEFT,
            center + 2.0 * RIGHT,
            color="#1E90FF"
        )
        wall_label = Text("Wall Line", font_size=16, color="#1E90FF")
        # Fixed: Issue 31 - use place_in_area for label
        self.place_in_area(wall_label, "D5", "D6", scale_factor=0.7)
        
        self.play(Create(wall_line), Write(wall_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Starting point on the circle
        start_angle = -15 * DEGREES
        start_pt = center + 1.8 * np.array([np.cos(start_angle), np.sin(start_angle), 0])
        dot = Dot(point=start_pt, color=YELLOW, radius=0.08)
        
        # The path bounces like light inside a circular mirror.
        self.play(FadeIn(dot))
        
        # Reflection 1: Block-Block (across collision line)
        ref1 = get_reflection(start_pt, alpha, center)
        path1 = Line(start_pt, ref1, color=YELLOW, stroke_width=2)
        
        self.play(Create(path1), dot.animate.move_to(ref1))
        self.wait(0.5)
        
        # Reflection 2: Wall (across horizontal line)
        ref2 = get_reflection(ref1, 0, center)
        path2 = Line(ref1, ref2, color=YELLOW, stroke_width=2)
        
        self.play(Create(path2), dot.animate.move_to(ref2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Each collision adds one segment to the path.
        # Continue sequence for a few steps
        current_pt = ref2
        for _ in range(3):
            # Reflection across collision line
            next_pt = get_reflection(current_pt, alpha, center)
            path = Line(current_pt, next_pt, color=YELLOW, stroke_width=2)
            self.play(Create(path), dot.animate.move_to(next_pt), run_time=0.4)
            current_pt = next_pt
            
            # Reflection across wall line
            next_pt = get_reflection(current_pt, 0, center)
            path = Line(current_pt, next_pt, color=YELLOW, stroke_width=2)
            self.play(Create(path), dot.animate.move_to(next_pt), run_time=0.4)
            current_pt = next_pt

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # We simply count these geometric reflections.
        # Speed up the process to show many steps.
        for _ in range(5):
            # Reflection across collision line
            next_pt = get_reflection(current_pt, alpha, center)
            path = Line(current_pt, next_pt, color=YELLOW, stroke_width=2)
            self.add(path)
            current_pt = next_pt
            
            # Reflection across wall line
            next_pt = get_reflection(current_pt, 0, center)
            path = Line(current_pt, next_pt, color=YELLOW, stroke_width=2)
            self.add(path)
            current_pt = next_pt
            
        self.play(dot.animate.move_to(current_pt), run_time=1)
        self.wait(2)
