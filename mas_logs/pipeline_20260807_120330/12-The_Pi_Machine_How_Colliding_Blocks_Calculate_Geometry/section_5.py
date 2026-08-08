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
        # Data from Storyboard
        title = "The Geometry of 'Clacks'"
        lines = [
            "Collisions cause the point to jump along the circle.",
            "The mass ratio determines the distance of each jump.",
            "Momentum and energy reflections guide these geometric steps.",
            "The blocks trace a path covering exactly pi radians.",
            "Many tiny jumps eventually reveal the digits of pi."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors from Storyboard/Plan
        color_cyan = "#00FFFF"
        color_yellow = "#FFFF00"
        color_orange = "#FF8C00"
        color_grey = "#888888"
        color_gold = "#FFD700"

        # Circle for visualization
        # Cyan circle at 'D4'
        circle = Circle(radius=1.5, color=color_cyan)
        self.place_at_grid(circle, 'D4')
        center_pos = circle.get_center()
        circle_radius = 1.5

        # Initial point on circle at angle PI (9 o'clock)
        start_angle = PI
        state_point = Dot(point=center_pos + circle_radius * np.array([np.cos(start_angle), np.sin(start_angle), 0]), color=color_yellow)

        # === Animation for Lecture Line 1 ===
        # Collisions cause the point to jump along the circle.
        self.play(self.lecture[0].animate.set_color(color_cyan))
        
        self.add(circle)
        self.play(FadeIn(state_point))
        
        # First jump
        jump_angle = 0.5 # radians
        new_angle_1 = start_angle - jump_angle # moving clockwise along top
        p1 = center_pos + circle_radius * np.array([np.cos(new_angle_1), np.sin(new_angle_1), 0])
        
        jump_arc_1 = ArcBetweenPoints(state_point.get_center(), p1, radius=circle_radius, color=color_yellow)
        
        self.play(
            MoveAlongPath(state_point, jump_arc_1),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # The mass ratio determines the distance of each jump.
        self.play(self.lecture[1].animate.set_color(color_orange))
        
        # Thick orange arc for the jump
        thick_orange_arc = Arc(radius=circle_radius, start_angle=start_angle, angle=-jump_angle, arc_center=center_pos, color=color_orange, stroke_width=8)
        self.play(Create(thick_orange_arc))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Momentum and energy reflections guide these geometric steps.
        self.play(self.lecture[2].animate.set_color(color_grey))
        
        # Grey dashed lines intersecting the circle
        line_1 = DashedLine(
            center_pos + circle_radius * np.array([np.cos(start_angle), np.sin(start_angle), 0]),
            center_pos + circle_radius * np.array([np.cos(new_angle_1), np.sin(new_angle_1), 0]),
            color=color_grey
        )
        
        # Second jump
        new_angle_2 = new_angle_1 - jump_angle
        p2 = center_pos + circle_radius * np.array([np.cos(new_angle_2), np.sin(new_angle_2), 0])
        jump_arc_2 = ArcBetweenPoints(state_point.get_center(), p2, radius=circle_radius, color=color_yellow)
        line_2 = DashedLine(p1, p2, color=color_grey)
        
        self.play(
            Create(line_1),
            MoveAlongPath(state_point, jump_arc_2),
            Create(line_2),
            run_time=1.2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # The blocks trace a path covering exactly pi radians.
        self.play(self.lecture[3].animate.set_color(color_gold))
        
        # Gold arc on top half (from PI to 0)
        gold_arc = Arc(radius=circle_radius + 0.1, start_angle=PI, angle=-PI, arc_center=center_pos, color=color_gold, stroke_width=6)
        
        # Issue 33: Place pi_label at B4, scale_factor=0.8
        pi_label = MathTex(r"\pi", color=color_gold)
        self.place_at_grid(pi_label, 'B4', scale_factor=0.8)
        
        self.play(
            Create(gold_arc),
            Write(pi_label)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Many tiny jumps eventually reveal the digits of pi.
        self.play(self.lecture[4].animate.set_color(color_cyan))
        
        # Many tiny jumps filling the gold arc
        small_jump_angle = 0.15
        current_angle = new_angle_2
        small_jumps = VGroup()
        
        # Jumps from current_angle down to 0
        while current_angle > small_jump_angle:
            next_angle = current_angle - small_jump_angle
            p_start = center_pos + circle_radius * np.array([np.cos(current_angle), np.sin(current_angle), 0])
            p_end = center_pos + circle_radius * np.array([np.cos(next_angle), np.sin(next_angle), 0])
            small_jumps.add(ArcBetweenPoints(p_start, p_end, radius=circle_radius, color=color_cyan, stroke_width=3))
            current_angle = next_angle
        
        # Final bit to 0
        p_last = center_pos + circle_radius * np.array([1, 0, 0])
        small_jumps.add(ArcBetweenPoints(center_pos + circle_radius * np.array([np.cos(current_angle), np.sin(current_angle), 0]), p_last, radius=circle_radius, color=color_cyan, stroke_width=3))

        self.play(
            LaggedStart(*[Create(sj) for sj in small_jumps], lag_ratio=0.05),
            run_time=2.5
        )
        self.wait(2)
