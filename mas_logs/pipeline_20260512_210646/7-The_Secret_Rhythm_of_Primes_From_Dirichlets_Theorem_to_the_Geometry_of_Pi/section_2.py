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

class Section2Scene(TeachingScene):
    def construct(self):
        # Script setup
        title_str = "Dirichlet’s Discovery: Infinite Tracks"
        lines_str = [
            'These sequences form two infinite tracks for primes.',
            'Primes like five and thirteen populate the gold track.',
            'Primes like three and seven populate the silver track.',
            'Dirichlet proved both tracks contain infinitely many primes.',
            'No matter how far, prime runners never stop appearing.'
        ]
        self.setup_layout(title_str, lines_str)

        # Colors
        COLOR_T1 = "#FFD700"  # Gold
        COLOR_T2 = "#C0C0C0"  # Silver
        HIGHLIGHT = YELLOW
        
        # Mapping values to the grid's X range (Col 1 to 6: 0.5 to 5.5)
        def n_to_x(n, n_max=20):
            return 0.5 + (n / n_max) * 5.0

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        # Track lines
        track1 = Line(self.grid["B1"], self.grid["B6"], color=COLOR_T1, stroke_width=4)
        track2 = Line(self.grid["D1"], self.grid["D6"], color=COLOR_T2, stroke_width=4)
        
        # Labels - Issues 33 & 34 fixed
        label1 = Text("Track 4k + 1", font_size=20, color=COLOR_T1)
        self.place_in_area(label1, 'A1', 'A2', scale_factor=0.8)
        
        label2 = Text("Track 4k + 3", font_size=20, color=COLOR_T2)
        self.place_in_area(label2, 'C1', 'C2', scale_factor=0.8)

        # Asset: Runner icons - Issue 27
        runner_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/runners.svg"
        runner1 = SVGMobject(runner_asset_path, height=0.4, fill_color=COLOR_T1, stroke_color=COLOR_T1)
        runner2 = SVGMobject(runner_asset_path, height=0.4, fill_color=COLOR_T2, stroke_color=COLOR_T2)
        
        self.place_at_grid(runner1, "B1")
        self.place_at_grid(runner2, "D1")

        self.play(
            Create(track1), 
            Create(track2), 
            Write(label1), 
            Write(label2), 
            FadeIn(runner1), 
            FadeIn(runner2)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )
        
        primes1 = [5, 13, 17]
        dots1 = VGroup(*[
            Dot(point=[n_to_x(p), self.grid["B1"][1], 0], color=COLOR_T1, radius=0.08)
            for p in primes1
        ])
        labels_p1 = VGroup(*[
            Text(str(p), font_size=16, color=COLOR_T1).next_to(dots1[i], UP, buff=0.1)
            for i, p in enumerate(primes1)
        ])
        
        self.play(LaggedStart(*[FadeIn(d) for d in dots1], lag_ratio=0.2))
        self.play(FadeIn(labels_p1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )
        
        primes2 = [3, 7, 11, 19]
        dots2 = VGroup(*[
            Dot(point=[n_to_x(p), self.grid["D1"][1], 0], color=COLOR_T2, radius=0.08)
            for p in primes2
        ])
        labels_p2 = VGroup(*[
            Text(str(p), font_size=16, color=COLOR_T2).next_to(dots2[i], UP, buff=0.1)
            for i, p in enumerate(primes2)
        ])
        
        self.play(LaggedStart(*[FadeIn(d) for d in dots2], lag_ratio=0.2))
        self.play(FadeIn(labels_p2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT)
        )
        
        # Zoom out simulation: re-map existing dots to a 0-100 range
        new_dots1 = VGroup(*[
            Dot(point=[n_to_x(p, 100), self.grid["B1"][1], 0], color=COLOR_T1, radius=0.04)
            for p in primes1
        ])
        new_dots2 = VGroup(*[
            Dot(point=[n_to_x(p, 100), self.grid["D1"][1], 0], color=COLOR_T2, radius=0.04)
            for p in primes2
        ])
        
        self.play(
            FadeOut(labels_p1), FadeOut(labels_p2),
            Transform(dots1, new_dots1),
            Transform(dots2, new_dots2),
            run_time=2
        )
        
        # Extend tracks with arrows
        arrow1 = Arrow(self.grid["B6"], self.grid["B6"] + RIGHT * 0.4, color=COLOR_T1, buff=0)
        arrow2 = Arrow(self.grid["D6"], self.grid["D6"] + RIGHT * 0.4, color=COLOR_T2, buff=0)
        self.play(Create(arrow1), Create(arrow2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT)
        )
        
        # More primes pop up in 0-100 range
        extra_primes1 = [29, 37, 41, 53, 61, 73, 89, 97]
        extra_primes2 = [23, 31, 43, 47, 59, 67, 71, 79, 83]
        
        extra_dots1 = VGroup(*[
            Dot(point=[n_to_x(p, 100), self.grid["B1"][1], 0], color=COLOR_T1, radius=0.04)
            for p in extra_primes1
        ])
        extra_dots2 = VGroup(*[
            Dot(point=[n_to_x(p, 100), self.grid["D1"][1], 0], color=COLOR_T2, radius=0.04)
            for p in extra_primes2
        ])
        
        # Runners advance - Issue 27
        self.play(
            LaggedStart(*[FadeIn(d) for d in extra_dots1], lag_ratio=0.1),
            LaggedStart(*[FadeIn(d) for d in extra_dots2], lag_ratio=0.1),
            runner1.animate.move_to([n_to_x(97, 100), self.grid["B1"][1], 0]),
            runner2.animate.move_to([n_to_x(83, 100), self.grid["D1"][1], 0]),
            run_time=3
        )
        self.wait(2)

        # Reset lecture colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
