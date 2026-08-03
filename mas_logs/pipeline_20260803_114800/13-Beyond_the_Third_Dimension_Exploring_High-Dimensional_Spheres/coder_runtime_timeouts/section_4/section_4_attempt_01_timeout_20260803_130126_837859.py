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
        # Data from storyboard and script
        title = "The 'Crust' Phenomenon (Surface Concentration)"
        lines = [
            "- High-dimensional spheres are mostly made of crust.",
            "- Almost all volume lies near the outer surface.",
            "- The interior of the sphere is effectively empty."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        ORANGE_COLOR = "#FFA500"
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        
        # A circle with a shaded outer ring (thickness epsilon). Outer ring color #FFA500.
        # Sphere center at D4 roughly
        outer_circle = Circle(radius=1.8, color=WHITE)
        annulus_initial = Annulus(inner_radius=1.6, outer_radius=1.8, color=ORANGE_COLOR, fill_opacity=0.6, stroke_width=0)
        
        sphere_group = VGroup(outer_circle, annulus_initial)
        self.place_in_area(sphere_group, 'B2', 'E5', scale_factor=1.0)
        center_point = sphere_group.get_center()
        
        self.play(Create(outer_circle))
        self.play(FadeIn(annulus_initial))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(ORANGE_COLOR))
        
        # 3D sphere with a shell. The inner core shrinks as n increases.
        n_tracker = ValueTracker(2)
        
        # Dynamic crust and core using parametric radii
        # Radius of core = R * (0.95)^n (visual heuristic for "most volume at surface")
        def get_core_radius():
            val = n_tracker.get_value()
            # Start at ~1.6 for n=2, end at ~0.1 for n=100
            # 1.8 * (0.96)^2 = 1.65
            # 1.8 * (0.96)^100 = 0.03
            return 1.8 * (0.96 ** val)

        dynamic_crust = always_redraw(lambda: Annulus(
            inner_radius=get_core_radius(),
            outer_radius=1.8,
            color=ORANGE_COLOR,
            fill_opacity=0.7,
            stroke_width=0
        ).move_to(center_point))
        
        dynamic_core = always_redraw(lambda: Circle(
            radius=get_core_radius(),
            color=WHITE,
            fill_opacity=0.2,
            stroke_width=1
        ).move_to(center_point))
        
        # Progress indicator for dimension n
        # Use Integer to avoid Text recreation inside add_updater/always_redraw
        n_label_text = Text("Dimension n = ", font_size=20, color=WHITE)
        self.place_at_grid(n_label_text, "B4")
        n_val = Integer(2, font_size=20, color=WHITE).next_to(n_label_text, RIGHT)
        n_val.add_updater(lambda m: m.set_value(int(n_tracker.get_value())))
        
        self.remove(annulus_initial)
        self.add(dynamic_crust, dynamic_core, n_label_text, n_val)
        
        # Transition from n=2 to n=100
        self.play(n_tracker.animate.set_value(100), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(ORANGE_COLOR))
        
        # Highlight the final n=100 and the "empty" center
        self.play(n_val.animate.set_color(ORANGE_COLOR))
        
        # Add a small dot at the center and a label pointing to it
        center_dot = Dot(center_point, radius=0.04, color=WHITE)
        label_empty = Text("Practically Empty", font_size=20, color=WHITE)
        self.place_at_grid(label_empty, "D6")
        arrow_to_center = Arrow(start=label_empty.get_left(), end=center_point, color=WHITE, buff=0.1)
        
        self.play(FadeIn(center_dot))
        self.play(Create(arrow_to_center), FadeIn(label_empty))
        self.wait(2)
