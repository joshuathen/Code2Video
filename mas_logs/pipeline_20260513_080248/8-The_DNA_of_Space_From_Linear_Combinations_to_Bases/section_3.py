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
        # Define teaching content
        title_text = "The Reach: What is a Span?"
        lecture_lines = [
            'What if we use every possible scaling factor?',
            "All resulting points form the vector's span.",
            'Two non-parallel vectors can span a plane.',
            'Span represents the reach of your set.',
            'Varying coefficients paints the entire reachable space.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color Constants
        V_COLOR = "#00FFFF"
        W_COLOR = "#FFFF00"
        DOT_COLOR = "#888888"
        SPAN_COLOR = "#ADD8E6"

        # Anchors
        # We will use D3 as our origin for the vector space visualization
        origin_point = self.grid["D3"]
        
        def to_space(x, y):
            # Scale visual units slightly
            return origin_point + np.array([x * 0.7, y * 0.7, 0])

        # === Animation for Lecture Line 1 ===
        # Vectors v (#00FFFF) and w (#FFFF00) appear. Scale them by random factors.
        self.lecture[0].set_color(V_COLOR)
        
        v_dir = np.array([1.5, 0.4, 0])
        w_dir = np.array([0.4, 1.5, 0])
        
        v_arrow = Arrow(start=origin_point, end=to_space(v_dir[0], v_dir[1]), color=V_COLOR, buff=0)
        w_arrow = Arrow(start=origin_point, end=to_space(w_dir[0], w_dir[1]), color=W_COLOR, buff=0)
        
        v_label = Text("v", font_size=18, color=V_COLOR).next_to(v_arrow.get_end(), RIGHT, buff=0.1)
        w_label = Text("w", font_size=18, color=W_COLOR).next_to(w_arrow.get_end(), UP, buff=0.1)

        self.play(GrowArrow(v_arrow), GrowArrow(w_arrow), FadeIn(v_label), FadeIn(w_label))
        
        # Random scaling demonstration
        v_scaled = Arrow(start=origin_point, end=to_space(v_dir[0]*2.0, v_dir[1]*2.0), color=V_COLOR, buff=0)
        w_scaled = Arrow(start=origin_point, end=to_space(w_dir[0]*-1.2, w_dir[1]*-1.2), color=W_COLOR, buff=0)
        
        self.play(
            Transform(v_arrow, v_scaled),
            Transform(w_arrow, w_scaled),
            v_label.animate.move_to(to_space(v_dir[0]*2.0 + 0.2, v_dir[1]*2.0)),
            w_label.animate.move_to(to_space(w_dir[0]*-1.2, w_dir[1]*-1.2 + 0.2)),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A grid of points (#888888) starts appearing at the tips of various linear combinations.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(DOT_COLOR)
        
        dots = VGroup()
        for c1 in np.linspace(-1.5, 2.5, 6):
            for c2 in np.linspace(-1.5, 2.5, 6):
                pos = origin_point + c1 * (v_dir * 0.7) + c2 * (w_dir * 0.7)
                dot = Dot(point=pos, radius=0.04, color=DOT_COLOR)
                dots.add(dot)
        
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.03))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A paintbrush moves across as points merge into a continuous light-blue region.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(SPAN_COLOR)
        
        paintbrush = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/paintbrush.svg")
        paintbrush.set_height(0.6)
        self.place_at_grid(paintbrush, "B1")
        
        # Shaded region
        # We'll approximate the visible portion of the plane
        p1 = origin_point + 3 * (v_dir * 0.7) + 3 * (w_dir * 0.7)
        p2 = origin_point - 3 * (v_dir * 0.7) + 3 * (w_dir * 0.7)
        p3 = origin_point - 3 * (v_dir * 0.7) - 3 * (w_dir * 0.7)
        p4 = origin_point + 3 * (v_dir * 0.7) - 3 * (w_dir * 0.7)
        
        span_plane = Polygon(p1, p2, p3, p4, stroke_width=0, fill_color=SPAN_COLOR, fill_opacity=0.4)
        span_plane.set_z_index(-1) # Ensure it is behind labels
        
        self.add(paintbrush)
        self.play(
            paintbrush.animate.move_to(self.grid["E6"]),
            FadeIn(span_plane),
            FadeOut(dots),
            run_time=2.5
        )
        self.play(FadeOut(paintbrush))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display the text label 'Span' in the center of the shaded region.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(SPAN_COLOR)
        
        span_text = Text("Span", font_size=32, color=WHITE, weight=BOLD)
        # Use place_in_area to center it in the right-side visualization zone
        self.place_in_area(span_text, "C3", "D4")
        
        self.play(Write(span_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The shaded region expands outwards to fill the entire screen.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(SPAN_COLOR)
        
        infinite_region = Rectangle(
            width=20, 
            height=20, 
            fill_color=SPAN_COLOR, 
            fill_opacity=0.3, 
            stroke_width=0
        ).move_to(origin_point)
        infinite_region.set_z_index(-2) # Put it behind everything including the initial plane

        self.play(
            ReplacementTransform(span_plane, infinite_region),
            span_text.animate.scale(1.5),
            run_time=2
        )
        self.wait(2)
