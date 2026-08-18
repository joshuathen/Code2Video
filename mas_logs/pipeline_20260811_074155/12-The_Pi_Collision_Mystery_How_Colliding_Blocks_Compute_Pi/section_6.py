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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Grand Reveal", [
            "As mass increases, collision counts match Pi more closely.",
            "Geometry explains why these digits appear in our world.",
            "Physics and Pi are connected through this simple setup."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # 1. Fade in four small circles (#FFFFFF)
        circles = VGroup(*[Circle(radius=0.35, color=WHITE) for _ in range(4)])
        self.place_at_grid(circles[0], 'B2')
        self.place_at_grid(circles[1], 'B5')
        self.place_at_grid(circles[2], 'E2')
        self.place_at_grid(circles[3], 'E5')

        # 2. Label them with ratios: 1:1, 1:100, 1:10,000, 1:1,000,000
        ratios = VGroup(
            Text("1:1", font_size=20),
            Text("1:100", font_size=20),
            Text("1:10,000", font_size=20),
            Text("1:1,000,000", font_size=20)
        )
        self.place_at_grid(ratios[0], 'A2', scale_factor=1.0)
        self.place_at_grid(ratios[1], 'A5', scale_factor=1.0)
        self.place_at_grid(ratios[2], 'D2', scale_factor=1.0)
        self.place_at_grid(ratios[3], 'D5', scale_factor=1.0)

        self.play(FadeIn(circles), FadeIn(ratios))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # 3. Each circle fills with collision paths of different densities
        def get_path(num_points, radius=0.35):
            if num_points > 100:
                # Use a circular fill to represent extreme density for high Pi approximations
                return Circle(radius=radius, color=BLUE_B, fill_opacity=0.4, stroke_width=1)
            points = []
            for i in range(num_points + 1):
                angle = i * (PI / num_points)
                points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
            path = VMobject(color=BLUE_B, stroke_width=2)
            path.set_points_as_corners(points)
            return path

        paths = VGroup(
            get_path(3).move_to(circles[0]),
            get_path(31).move_to(circles[1]),
            get_path(100).move_to(circles[2]), # Representative points
            get_path(500).move_to(circles[3])  # Representative density
        )

        # 4. Counters below them stop at: 3, 31, 314, 3141
        # Resolving Issue 35 and 41 by explicitly setting positions at C2, C5, F2, F5
        counters = VGroup(
            Text("3", font_size=24, color=YELLOW),
            Text("31", font_size=24, color=YELLOW),
            Text("314", font_size=24, color=YELLOW),
            Text("3141", font_size=24, color=YELLOW)
        )
        self.place_at_grid(counters[0], 'C2')
        self.place_at_grid(counters[1], 'C5')
        self.place_at_grid(counters[2], 'F2')
        self.place_at_grid(counters[3], 'F5')

        self.play(
            LaggedStart(*[Create(p) for p in paths], lag_ratio=0.3),
            LaggedStart(*[Write(c) for c in counters], lag_ratio=0.3),
            run_time=3
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # 5. The digits of Pi (#FFFF00) glow and move to the center
        # Resolving Issue 34 and 41: use area C1-D6 and scale 1.5
        pi_text = Text("3.14159...", font_size=48, color="#FFFF00")
        self.place_in_area(pi_text, 'C1', 'D6', scale_factor=1.5)
        
        pi_glow = pi_text.copy().set_stroke(YELLOW, width=8, opacity=0.4)
        
        # Clear the setup and reveal the grand result
        # Transitioning cleanly to avoid overlap clutter (Issue 35)
        self.play(
            FadeOut(circles),
            FadeOut(paths),
            FadeOut(ratios),
            FadeOut(counters, shift=DOWN),
            FadeIn(pi_text, shift=UP),
            FadeIn(pi_glow),
            run_time=2
        )
        
        self.play(Indicate(pi_text, color=YELLOW, scale_factor=1.1))
        self.wait(3)
