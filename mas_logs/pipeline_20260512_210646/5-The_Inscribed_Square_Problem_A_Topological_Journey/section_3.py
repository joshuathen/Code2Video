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
        # Setup the scene
        title_text = "The Stepping Stone: The Inscribed Rectangle"
        lines = [
            "Let's first look for rectangles instead of squares.",
            "Any two points on the loop define a possible side.",
            "We track the midpoint and distance between these points."
        ]
        self.setup_layout(title_text, lines)

        # Colors
        LOOP_COLOR = "#ADD8E6"
        DOT_COLOR = "#FFA500"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(LOOP_COLOR))

        # Asset Integration (Issue 36)
        loop = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/loop.svg")
        loop.set_color(LOOP_COLOR)
        
        # Fix loop position (Issue 43)
        self.place_in_area(loop, 'A2', 'F6', scale_factor=0.8)
        
        # Alignment with grid midpoint (Issue 45 anchor)
        loop.move_to(self.grid['C4'])
        loop_center = loop.get_center()

        # Geometry logic for points staying on the loop
        def loop_geom_func(t):
            # Centrally symmetric distortion to guarantee rectangles exist
            r = 1.6 + 0.3 * np.cos(2 * t)
            return np.array([
                r * np.cos(t) * 1.3, 
                r * np.sin(t) * 1.0, 
                0
            ])

        def get_pt(t):
            return loop_geom_func(t) + loop_center

        self.play(FadeIn(loop))
        
        # Initial points A and B
        t_init = 0.6
        dot_a = Dot(get_pt(t_init), color=DOT_COLOR)
        dot_b = Dot(get_pt(t_init + PI), color=DOT_COLOR)
        label_a = Text("A", font_size=20, color=WHITE).next_to(dot_a, UR, buff=0.1)
        label_b = Text("B", font_size=20, color=WHITE)
        
        # Fix label B position (Issue 44)
        self.place_at_grid(label_b, 'B2', scale_factor=0.5)

        self.play(FadeIn(dot_a, dot_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(DOT_COLOR))
        
        t_tracker = ValueTracker(t_init)

        def get_rect_points(t):
            # 4 points forming a rectangle on symmetric loop
            return [get_pt(t), get_pt(-t), get_pt(t + PI), get_pt(-t + PI)]

        rect = always_redraw(lambda: Polygon(*get_rect_points(t_tracker.get_value()), color=WHITE, stroke_width=2))
        
        # Dots C and D
        dot_c = always_redraw(lambda: Dot(get_rect_points(t_tracker.get_value())[1], color=DOT_COLOR, radius=0.08))
        dot_d = always_redraw(lambda: Dot(get_rect_points(t_tracker.get_value())[3], color=DOT_COLOR, radius=0.08))
        label_c = Text("C", font_size=20, color=WHITE)
        label_d = Text("D", font_size=20, color=WHITE)

        # Updaters for A, C, D and labels (B stays fixed per Issue 44)
        dot_a.add_updater(lambda m: m.move_to(get_rect_points(t_tracker.get_value())[0]))
        dot_b.add_updater(lambda m: m.move_to(get_rect_points(t_tracker.get_value())[2]))
        label_a.add_updater(lambda m: m.next_to(dot_a, UR, buff=0.1))
        label_c.add_updater(lambda m: m.next_to(dot_c, DL, buff=0.1))
        label_d.add_updater(lambda m: m.next_to(dot_d, DR, buff=0.1))

        self.play(Create(rect), FadeIn(dot_c, dot_d, label_c, label_d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))

        # Midpoint dot fixed to grid (Issue 45)
        midpoint_dot = Dot(color=WHITE)
        self.place_at_grid(midpoint_dot, 'C4', scale_factor=0.3)

        # Visualizing diagonals
        diagonals = always_redraw(lambda: VGroup(
            Line(get_rect_points(t_tracker.get_value())[0], get_rect_points(t_tracker.get_value())[2], color=GRAY, stroke_opacity=0.6),
            Line(get_rect_points(t_tracker.get_value())[1], get_rect_points(t_tracker.get_value())[3], color=GRAY, stroke_opacity=0.6)
        ))

        self.play(FadeIn(midpoint_dot, diagonals))
        self.wait(1)

        # The rectangle rotates and scales while vertices stay on the loop
        self.play(t_tracker.animate.set_value(1.3), run_time=3, rate_func=there_and_back)
        self.wait(2)

        # Cleanup
        dot_a.clear_updaters()
        dot_b.clear_updaters()
        label_a.clear_updaters()
        label_c.clear_updaters()
        label_d.clear_updaters()
