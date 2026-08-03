from manim import *
import numpy as np

CYAN = "#00FFFF"

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
        lecture_lines = [
            "Each collision is a transformation of the state.",
            "A wall bounce reflects the point across an axis.",
            "A block bounce reflects it across a tilted line.",
            "The system behaves like a beam of light.",
            "Every bounce is a reflection inside this circular mirror."
        ]
        self.setup_layout("Collisions as Geometric Reflections", lecture_lines)

        # Assets
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE)
        mirror_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg", color=CYAN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        circle = Circle(radius=2.2, color=CYAN, stroke_width=4)
        h_axis = Line(LEFT*2.6, RIGHT*2.6, color=GRAY, stroke_opacity=0.4)
        v_axis = Line(UP*2.6, DOWN*2.6, color=GRAY, stroke_opacity=0.4)
        
        # Tilted line (45 degrees)
        angle = 45 * DEGREES
        tilted_line = Line(
            start=2.6 * np.array([-np.cos(angle), -np.sin(angle), 0]),
            end=2.6 * np.array([np.cos(angle), np.sin(angle), 0]),
            color=WHITE,
            stroke_width=3
        )
        t_label = Text("Block Bounce Line", font_size=16, color=WHITE)
        
        system_group = VGroup(circle, h_axis, v_axis, tilted_line)
        # Resolved Issue 33: Scale system_group to 0.9
        self.place_in_area(system_group, 'A1', 'F6', scale_factor=0.9)
        
        # Resolved Issue 32: Reposition t_label to avoid overlap
        self.place_in_area(t_label, 'A5', 'B6', scale_factor=0.7)
        
        self.play(Create(circle), Create(h_axis), Create(v_axis))
        self.play(Create(tilted_line), Write(t_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(YELLOW)
        
        center = system_group.get_center()
        radius = 2.2 * 0.9 # accounting for scale_factor=0.9
        
        start_angle = 30 * DEGREES
        p1 = center + radius * np.array([np.cos(start_angle), np.sin(start_angle), 0])
        point = Dot(p1, color=YELLOW, radius=0.08)
        
        # Wall bounce: reflect across vertical axis (x -> -x relative to center)
        rel_p1 = p1 - center
        p2 = center + np.array([-rel_p1[0], rel_p1[1], 0])
        
        v_axis_highlight = Line(center + UP*2.34, center + DOWN*2.34, color=YELLOW, stroke_width=5)
        
        self.play(FadeIn(point))
        self.play(Create(v_axis_highlight), run_time=0.4)
        self.play(point.animate.move_to(p2), run_time=1)
        self.play(FadeOut(v_axis_highlight), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(YELLOW)
        
        # Reflection across 45-degree line: (x, y) -> (y, x) relative to center
        rel_p2 = p2 - center
        p3 = center + np.array([rel_p2[1], rel_p2[0], 0])
        
        # Block Asset Integration (Issue 26)
        self.place_at_grid(block_icon, "B6", scale_factor=0.4)
        
        self.play(tilted_line.animate.set_stroke(color=YELLOW, width=6), FadeIn(block_icon), run_time=0.4)
        self.play(point.animate.move_to(p3), run_time=1)
        self.play(tilted_line.animate.set_stroke(color=WHITE, width=3), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(YELLOW)
        
        # Trace trajectory as light beam
        path = VMobject(color=YELLOW, stroke_width=2)
        path.set_points_as_corners([p1, p2, p3])
        
        # More bounces to simulate "beam of light"
        # Reflection 4: Wall (y-axis)
        rel_p3 = p3 - center
        p4 = center + np.array([-rel_p3[0], rel_p3[1], 0])
        # Reflection 5: Tilted
        rel_p4 = p4 - center
        p5 = center + np.array([rel_p4[1], rel_p4[0], 0])
        # Reflection 6: Wall
        rel_p5 = p5 - center
        p6 = center + np.array([-rel_p5[0], rel_p5[1], 0])
        
        full_points = [p1, p2, p3, p4, p5, p6]
        path.set_points_as_corners(full_points)
        
        # We'll use a Successive-like sequence of lines or just Create(path)
        self.add(path)
        self.play(Create(path), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(CYAN)
        
        # Pulse the mirror and add mirror icon
        self.place_at_grid(mirror_icon, "E1", scale_factor=0.4)
        
        self.play(FadeIn(mirror_icon))
        self.play(
            circle.animate.set_stroke(width=10),
            run_time=0.5,
            rate_func=there_and_back
        )
        self.play(
            circle.animate.set_stroke(width=10),
            run_time=0.5,
            rate_func=there_and_back
        )
        self.wait(2)
