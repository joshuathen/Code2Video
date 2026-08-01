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
        # Title and Lecture Lines (Updated to match MAS Stage-3 instructions)
        lines = [
            'Prime distribution directly dictates the value of pi.',
            'Algebraic rhythms transform into perfect geometric symmetry.',
            'Primes define the very fabric of our mathematical world.'
        ]
        self.setup_layout("Synthesis: Order from Prime Chaos", lines)

        # Colors
        COLOR_1 = YELLOW
        COLOR_2 = GREEN
        COLOR_PI = "#FF00FF" # Glowing Pi
        COLOR_TEXT = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        # Representing "Sorting buckets" (Issue 38: Move to B3, E3)
        bucket_1 = Rectangle(height=0.7, width=0.7, color=COLOR_1)
        bucket_2 = Rectangle(height=0.7, width=0.7, color=COLOR_1)
        self.place_at_grid(bucket_1, "B3", scale_factor=1.0)
        self.place_at_grid(bucket_2, "E3", scale_factor=1.0)
        
        # Representing "Tracks" (Issue 29: Use SVG asset)
        track_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/tracks.svg"
        track_1 = SVGMobject(track_asset).set_color(COLOR_2)
        track_2 = SVGMobject(track_asset).set_color(COLOR_2)
        self.place_at_grid(track_1, "B5", scale_factor=0.6)
        self.place_at_grid(track_2, "E5", scale_factor=0.6)
        
        self.play(
            FadeIn(bucket_1), FadeIn(bucket_2),
            DrawBorderThenFill(track_1), DrawBorderThenFill(track_2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_2)
        
        # Prepare Pi components
        # Issue 39: pi_top at area C3-C5
        pi_top = RoundedRectangle(corner_radius=0.1, height=0.2, width=3.0, color=COLOR_PI, fill_opacity=0.8)
        self.place_in_area(pi_top, "C3", "C5", scale_factor=1.0)
        
        # Legs
        pi_leg_l = RoundedRectangle(corner_radius=0.1, height=1.4, width=0.2, color=COLOR_PI, fill_opacity=0.8)
        self.place_at_grid(pi_leg_l, "D3", scale_factor=1.0)
        
        pi_leg_r = RoundedRectangle(corner_radius=0.1, height=1.4, width=0.2, color=COLOR_PI, fill_opacity=0.8)
        self.place_at_grid(pi_leg_r, "D5", scale_factor=1.0)

        pi_group = VGroup(pi_top, pi_leg_l, pi_leg_r)

        # Morph Animation (Tracks and Buckets -> Pi)
        self.play(
            ReplacementTransform(VGroup(track_1, track_2), pi_top),
            ReplacementTransform(bucket_1, pi_leg_l),
            ReplacementTransform(bucket_2, pi_leg_r),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_PI)
        
        # Final label (Issue 40: final_text F2-F6, scale 0.8)
        final_text = Text("The Hidden Rhythm of Primes", font_size=24, color=COLOR_TEXT)
        self.place_in_area(final_text, "F2", "F6", scale_factor=0.8)
        
        # Glow/Scale effect for the final synthesis
        self.play(
            Write(final_text),
            pi_group.animate.scale(1.1),
            run_time=2
        )
        self.play(
            pi_group.animate.scale(1/1.1),
            run_time=1
        )
        self.wait(3)
