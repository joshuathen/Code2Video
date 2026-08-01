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
        # TEACHING CONTENT
        lecture_lines = [
            "Predictable orbits belong to the stable Fatou set.",
            "Other points escape to infinity in chaotic paths.",
            "The boundary between stability and chaos is the Julia set.",
            "These boundaries are infinitely complex and jagged.",
            "Small changes on this edge lead to massive shifts."
        ]
        self.setup_layout("Stability vs. Chaos: Fatou and Julia Sets", lecture_lines)
        
        # Colors (L008)
        COLOR_FATOU = "#0000FF"
        COLOR_JULIA = "#FFFFFF"
        COLOR_STABLE = "#00FF00"
        COLOR_UNSTABLE = "#FF0000"
        COLOR_SMOOTH = "#FFFF00"
        COLOR_TEXT_HL = "#00FFFF" # Light cyan for final highlight

        # Helper for deterministic jagged shape (visual anchor for Julia set)
        def get_jagged_points(center, radius, noise_amp=0.2, n_points=120):
            points = []
            for i in range(n_points):
                angle = i * TAU / n_points
                # Fixed noise frequencies to ensure deterministic geometry
                r = radius + noise_amp * (np.sin(7 * angle) + 0.5 * np.sin(13 * angle) + 0.3 * np.cos(19 * angle))
                points.append(center + np.array([r * np.cos(angle), r * np.sin(angle), 0]))
            return points

        # === Animation for Lecture Line 1 ===
        # Fade in a solid blue region (#0000FF) representing the stable 'Fatou Set'.
        self.lecture[0].set_color(COLOR_FATOU)
        
        # Initialize at origin then place in grid area C2-F6 as per Issue 30 (prevents occlusion)
        fatou_origin_pts = get_jagged_points(ORIGIN, radius=1.0)
        fatou_region = Polygon(*fatou_origin_pts, stroke_width=0, fill_opacity=0.6, fill_color=COLOR_FATOU)
        self.place_in_area(fatou_region, "C2", "F6", scale_factor=0.9)
        
        self.play(FadeIn(fatou_region))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Show two nearby points (#00FF00 and #FF0000); one stays in the blue, the other exits the screen quickly.
        self.lecture[1].set_color(COLOR_UNSTABLE)
        self.wait(1.5)

        center_pos = fatou_region.get_center()
        fatou_pts = fatou_region.get_vertices()
        
        # Pick a point on the jagged edge for demonstration
        edge_pt = fatou_pts[18]
        vec_out = (edge_pt - center_pos) / np.linalg.norm(edge_pt - center_pos)
        
        # Position points slightly inside and outside the boundary
        p_stable = Dot(edge_pt - vec_out * 0.2, color=COLOR_STABLE, radius=0.08)
        p_unstable = Dot(edge_pt + vec_out * 0.1, color=COLOR_UNSTABLE, radius=0.08)
        
        self.play(FadeIn(p_stable), FadeIn(p_unstable))
        self.wait(0.5)
        
        # Movement: Stable stays bounded, Unstable escapes (metaphor for chaos)
        path_s = TracedPath(p_stable.get_center, stroke_color=COLOR_STABLE, stroke_opacity=0.4)
        path_u = TracedPath(p_unstable.get_center, stroke_color=COLOR_UNSTABLE, stroke_opacity=0.4)
        self.add(path_s, path_u)
        
        self.play(
            p_stable.animate.shift(LEFT * 0.6 + DOWN * 0.5),
            p_unstable.animate.shift(RIGHT * 6 + UP * 4),
            run_time=2.5,
            rate_func=rate_functions.linear # L024: module prefix for rate functions
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Draw a complex, jagged white outline (#FFFFFF) around the blue region to represent the 'Julia Set'.
        self.lecture[2].set_color(COLOR_JULIA)
        self.wait(1.5)

        # The Julia set is the boundary of the Fatou set
        julia_boundary = Polygon(*fatou_pts, color=COLOR_JULIA, stroke_width=3)
        self.play(Create(julia_boundary))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Use a morphing transformation to transition the jagged Julia boundary into a smooth shoreline (#FFFF00) and back.
        self.lecture[3].set_color(COLOR_SMOOTH)
        self.wait(1.5)

        smooth_circle = Circle(radius=1.2, color=COLOR_SMOOTH, stroke_width=3).move_to(center_pos)
        
        # Morphing highlights the difference between smooth geometry and fractal complexity
        self.play(Transform(julia_boundary, smooth_circle))
        self.wait(1.5)
        
        # Morph back to the actual jagged boundary
        jagged_restore = Polygon(*fatou_pts, color=COLOR_JULIA, stroke_width=3)
        self.play(Transform(julia_boundary, jagged_restore))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Perform a 3-second 'smooth' zoom into a small section of the Julia Set boundary to reveal detail.
        self.lecture[4].set_color(COLOR_TEXT_HL)
        self.wait(1.5)

        # Grouping objects to scale them relative to a point on the boundary
        visual_group = VGroup(fatou_region, julia_boundary, p_stable, path_s)
        zoom_target = fatou_pts[18]
        
        self.play(
            visual_group.animate.scale(2.5, about_point=zoom_target),
            run_time=3.0
        )
        self.wait(3.0)
