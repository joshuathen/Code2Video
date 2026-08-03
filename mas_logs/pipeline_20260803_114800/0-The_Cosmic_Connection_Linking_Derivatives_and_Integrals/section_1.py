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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "- Calculus rests on two main pillars.",
            "- Derivatives measure instantaneous change, like velocity.",
            "- Integrals measure total accumulation, like distance."
        ]
        self.setup_layout("The Grand Paradox: Change vs. Accumulation", lecture_lines)
        
        # Colors
        COLOR_DASH = "#FFD700"
        COLOR_SPEED = "#00BFFF"
        COLOR_ACCUM = "#00FF00"
        
        # Asset path
        CHEETAH_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg"
        
        # === Animation for Lecture Line 1 ===
        # Dash the Cheetah icon (#FFD700) appears on a horizontal path.
        self.lecture[0].set_color(YELLOW)
        
        # Load Cheetah SVG asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg]
        dash = SVGMobject(CHEETAH_ASSET).set_color(COLOR_DASH)
        dash_label = Text("Dash", font_size=20, color=COLOR_DASH)
        dash_group = VGroup(dash, dash_label).arrange(UP, buff=0.1)
        
        # Path on row D
        path = Line(self.grid["D1"], self.grid["D6"], color=GRAY)
        
        # Place dash in area C1-D6 as per critic feedback
        self.place_in_area(dash_group, 'C1', 'D6', scale_factor=0.6)
        
        self.play(Create(path))
        self.play(FadeIn(dash_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Derivatives measure instantaneous change, like velocity.
        # An arrow labeled 'Speed' (#00BFFF) emerges from Dash, showing change.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_SPEED)
        
        speed_arrow = Arrow(start=LEFT*0.5, end=RIGHT*0.5, color=COLOR_SPEED)
        speed_label = Text("Speed", font_size=18, color=COLOR_SPEED)
        speed_group = VGroup(speed_arrow, speed_label).arrange(UP, buff=0.1)
        
        # 'Change' text at B2
        change_text = Text("Change", font_size=32, color=COLOR_SPEED)
        self.place_at_grid(change_text, "B2")
        
        # Link speed arrow to dash using an updater to track its movement
        speed_group.add_updater(lambda m: m.next_to(dash, RIGHT, buff=0.1))
        
        self.play(FadeIn(speed_group))
        # Move dash to grid position D3
        self.play(
            dash_group.animate.move_to(self.grid["D3"]),
            FadeIn(change_text),
            run_time=2
        )
        self.play(change_text.animate.scale(1.2), rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Integrals measure total accumulation, like distance.
        # The path behind Dash glows green (#00FF00) to show total accumulation.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ACCUM)
        
        accum_text = Text("Accumulation", font_size=32, color=COLOR_ACCUM)
        self.place_at_grid(accum_text, "B5")
        
        # Glowing path segment starts from D1 (start of movement)
        start_pos = self.grid["D1"]
        glow_path = Line(start_pos, dash.get_center(), color=COLOR_ACCUM, stroke_width=6)
        
        self.play(
            Create(glow_path),
            FadeIn(accum_text),
            dash_group.animate.move_to(self.grid["D6"]),
            UpdateFromFunc(glow_path, lambda m: m.put_start_and_end_on(start_pos, dash.get_center())),
            run_time=2
        )
        
        # Pulsing text in sync
        self.play(
            change_text.animate.scale(1.2),
            accum_text.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        # 'Derivative' and 'Integral' labels placed at E2 and E5 per critic feedback
        coin1 = Circle(radius=0.7, color=COLOR_DASH, fill_opacity=0.9)
        coin_t1 = Text("Derivative", font_size=16, color=BLACK)
        side1 = VGroup(coin1, coin_t1)
        
        coin2 = Circle(radius=0.7, color=COLOR_DASH, fill_opacity=0.9)
        coin_t2 = Text("Integral", font_size=16, color=BLACK)
        side2 = VGroup(coin2, coin_t2)
        
        self.place_at_grid(side1, "E2", scale_factor=0.6)
        self.place_at_grid(side2, "E5", scale_factor=0.6)
        
        # Show both coins with rotation
        self.play(FadeIn(side1))
        self.play(Rotate(side1, axis=UP, angle=2*PI), run_time=1)
        self.play(FadeIn(side2))
        self.play(Rotate(side2, axis=UP, angle=2*PI), run_time=1)
        
        self.wait(2)
