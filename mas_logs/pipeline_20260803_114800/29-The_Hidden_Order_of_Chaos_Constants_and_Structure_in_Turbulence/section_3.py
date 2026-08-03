from manim import *

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
    def create_doll_silhouette(self, color, height=1.0):
        # A simple doll shape: small circle on top of a larger ellipse
        head = Circle(radius=0.15 * height, color=color, fill_opacity=0.3, stroke_width=2)
        body = Ellipse(width=0.4 * height, height=0.6 * height, color=color, fill_opacity=0.3, stroke_width=2)
        body.next_to(head, DOWN, buff=-0.05 * height)
        doll = VGroup(head, body)
        return doll

    def construct(self):
        title_text = "The Energy Cascade: Richardson\u2019s Vision"
        lecture_lines = [
            "Large eddies capture kinetic energy from the flow.",
            "These primary eddies break into smaller spinning structures.",
            "Energy transfers downward through a hierarchical cascade.",
            "Like nesting dolls, eddies fragment into tinier versions.",
            "This process continues until scales become extremely small."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        LARGE_COLOR = "#0000FF"
        MEDIUM_COLOR = "#008080"
        SMALL_COLOR = "#00FFFF"
        ENERGY_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(LARGE_COLOR)
        
        # Large eddy (blue circle) - Fix from Issue 29
        large_eddy = Circle(radius=1.8, color=LARGE_COLOR, stroke_width=6)
        self.place_in_area(large_eddy, "B3", "E4", scale_factor=0.8)
        
        # Swirl arrows inside/around
        arrows = VGroup(*[
            Arrow(start=large_eddy.get_center() + 2.0 * direction, 
                  end=large_eddy.get_center() + 1.2 * direction, 
                  color=ENERGY_COLOR, buff=0, stroke_width=3)
            for direction in [UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL]
        ])
        
        # Rotating updater
        large_eddy.add_updater(lambda m, dt: m.rotate(0.5 * dt))
        
        self.play(Create(large_eddy), FadeIn(arrows, shift=IN), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(MEDIUM_COLOR)
        
        # Medium eddies - Fix from Issue 30
        medium_eddy_1 = Circle(radius=0.9, color=MEDIUM_COLOR, stroke_width=4)
        medium_eddy_2 = Circle(radius=0.9, color=MEDIUM_COLOR, stroke_width=4)
        
        self.place_at_grid(medium_eddy_1, "C1", scale_factor=0.6)
        self.place_at_grid(medium_eddy_2, "C6", scale_factor=0.6)
        
        # Transition: Splitting
        self.play(
            ReplacementTransform(large_eddy.copy(), medium_eddy_1),
            ReplacementTransform(large_eddy, medium_eddy_2),
            FadeOut(arrows),
            run_time=2
        )
        
        # Rotating updaters
        medium_eddy_1.add_updater(lambda m, dt: m.rotate(0.8 * dt))
        medium_eddy_2.add_updater(lambda m, dt: m.rotate(-0.8 * dt))
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(SMALL_COLOR)
        
        # Small eddies hierarchy
        small_positions = ["B1", "D1", "B6", "D6"]
        small_eddies = VGroup(*[
            Circle(radius=0.5, color=SMALL_COLOR, stroke_width=2) for _ in range(4)
        ])
        
        for i, pos in enumerate(small_positions):
            self.place_at_grid(small_eddies[i], pos, scale_factor=0.7)
            
        self.play(
            ReplacementTransform(medium_eddy_1.copy(), small_eddies[0]),
            ReplacementTransform(medium_eddy_1.copy(), small_eddies[1]),
            ReplacementTransform(medium_eddy_2.copy(), small_eddies[2]),
            ReplacementTransform(medium_eddy_2.copy(), small_eddies[3]),
            run_time=2
        )
        
        for i, eddy in enumerate(small_eddies):
            eddy.add_updater(lambda m, dt, idx=i: m.rotate((1.2 + 0.2*idx) * dt))
            
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ENERGY_COLOR)
        
        # Matryoshka Doll silhouettes - Fixes from Issue 29 and 31
        doll_l = self.create_doll_silhouette(LARGE_COLOR, height=3.5)
        self.place_in_area(doll_l, "B3", "E4", scale_factor=0.8)
        
        doll_m1 = self.create_doll_silhouette(MEDIUM_COLOR, height=1.8)
        self.place_at_grid(doll_m1, "C1", scale_factor=0.5)
        doll_m2 = self.create_doll_silhouette(MEDIUM_COLOR, height=1.8)
        self.place_at_grid(doll_m2, "C6", scale_factor=0.5)
        
        doll_s = VGroup(*[self.create_doll_silhouette(SMALL_COLOR, height=1.0) for _ in range(4)])
        for i, pos in enumerate(small_positions):
            self.place_at_grid(doll_s[i], pos, scale_factor=0.4)
            
        # Pulse animation and reveal dolls
        self.play(
            LaggedStart(
                Indicate(small_eddies, color=ENERGY_COLOR),
                FadeIn(doll_l, scale=0.5),
                FadeIn(doll_m1, scale=0.5),
                FadeIn(doll_m2, scale=0.5),
                FadeIn(doll_s, scale=0.5),
                lag_ratio=0.2
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(SMALL_COLOR)
        
        # Tiny eddies for the end of cascade
        tiny_eddies = VGroup()
        offsets = [UP * 0.2, DOWN * 0.2]
        for i, pos in enumerate(small_positions):
            t1 = Circle(radius=0.15, color=SMALL_COLOR, stroke_width=1)
            t2 = Circle(radius=0.15, color=SMALL_COLOR, stroke_width=1)
            self.place_at_grid(t1, pos, scale_factor=1.0).shift(offsets[0])
            self.place_at_grid(t2, pos, scale_factor=1.0).shift(offsets[1])
            tiny_eddies.add(t1, t2)
        
        self.play(FadeIn(tiny_eddies, scale=0.2), run_time=1.5)
        
        # Update tiny eddies to rotate fast
        for tiny in tiny_eddies:
            tiny.add_updater(lambda m, dt: m.rotate(3.0 * dt))
            
        self.wait(3)
