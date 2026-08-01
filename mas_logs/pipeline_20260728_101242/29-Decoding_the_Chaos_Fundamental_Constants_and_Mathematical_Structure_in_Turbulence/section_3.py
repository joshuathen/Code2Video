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
        self.setup_layout(
            "The Energy Cascade: Richardson’s Vision",
            [
                "Kinetic energy enters the system through large-scale eddies.",
                "These big whorls break down into smaller, secondary whorls.",
                "Energy transfers downward across a wide range of scales.",
                "This process continues until the eddies reach microscopic sizes.",
                "Richardson famously described this as an infinite fractal cascade."
            ]
        )
        
        # Colors
        LARGE_EDDY_COLOR = "#0000FF"
        MEDIUM_EDDY_COLOR = "#4169E1"
        SMALL_EDDY_COLOR = "#1E90FF"
        DISSIPATION_COLOR = "#CD5C5C"
        LABEL_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(LARGE_EDDY_COLOR)
        
        large_eddy = Circle(radius=1.5, color=LARGE_EDDY_COLOR, stroke_width=6)
        self.place_in_area(large_eddy, "B2", "E5")
        
        # Visual spiral to emphasize rotation
        spirals = VGroup(*[
            Arc(radius=1.5 * (i/3), start_angle=TAU*i/3, angle=PI, color=LARGE_EDDY_COLOR, stroke_width=2)
            for i in range(1, 4)
        ])
        large_group = VGroup(large_eddy, spirals)
        large_group.add_updater(lambda m, dt: m.rotate(dt * 0.4))
        
        input_label = Text("Energy Input", font_size=20, color=LABEL_COLOR)
        # Fix Issue 26
        self.place_in_area(input_label, "A2", "A5", scale_factor=0.8)
        
        self.play(Create(large_group), Write(input_label))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(MEDIUM_EDDY_COLOR)
        
        medium_eddies = VGroup()
        m_pos = ["B2", "B5", "E2", "E5"]
        for p in m_pos:
            me = Circle(radius=0.6, color=MEDIUM_EDDY_COLOR, stroke_width=4)
            self.place_at_grid(me, p)
            me.add_updater(lambda m, dt: m.rotate(-dt * 0.8)) # Reverse rotation
            medium_eddies.add(me)
            
        breakdown_label = Text("Breakdown", font_size=20, color=LABEL_COLOR)
        # Fix Issue 27
        self.place_in_area(breakdown_label, "A2", "A5", scale_factor=0.8)

        self.play(
            FadeOut(large_group, scale=0.5),
            ReplacementTransform(input_label, breakdown_label),
            LaggedStart(*[Create(me) for me in medium_eddies], lag_ratio=0.2)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(MEDIUM_EDDY_COLOR)
        
        self.play(
            *[me.animate.scale(1.2) for me in medium_eddies],
            run_time=0.5, rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(SMALL_EDDY_COLOR)
        
        small_eddies = VGroup()
        grid_points = ["B2", "B3", "B4", "B5", "C2", "C3", "C4", "C5", "D2", "D3", "D4", "D5", "E2", "E3", "E4", "E5"]
        for gp in grid_points:
            for d in [UP*0.2, DOWN*0.2, LEFT*0.2, RIGHT*0.2]:
                se = Circle(radius=0.12, color=SMALL_EDDY_COLOR, stroke_width=2)
                se.move_to(self.grid[gp] + d)
                se.add_updater(lambda m, dt: m.rotate(dt * 1.5))
                small_eddies.add(se)

        self.play(
            FadeOut(medium_eddies, scale=0.3),
            FadeOut(breakdown_label),
            FadeIn(small_eddies)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(DISSIPATION_COLOR)
        
        dissipation_label = Text("Dissipation as Heat", font_size=20, color=LABEL_COLOR)
        # Fix Issue 25
        self.place_in_area(dissipation_label, "F2", "F5", scale_factor=0.8)

        self.play(
            small_eddies.animate.set_color(DISSIPATION_COLOR),
            Write(dissipation_label)
        )
        self.wait(2)
        self.play(
            FadeOut(small_eddies, shift=DOWN*0.5),
            FadeOut(dissipation_label)
        )
        self.wait(1)
        self.lecture[4].set_color(WHITE)
