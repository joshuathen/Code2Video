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
        title = "Iteration and Measurement"
        lines = [
            "We loop the Oracle and Diffusion steps together.",
            "With each iteration, the target amplitude grows much taller.",
            "We repeat this process roughly square root of N times.",
            "Finally, we measure the system to collapse the state.",
            "The algorithm successfully finds the target with high probability."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        oracle_box = Rectangle(width=1.5, height=0.8, color=BLUE_B).set_fill(BLUE_E, opacity=0.3)
        oracle_text = Text("Oracle", color=WHITE, font_size=24)
        oracle_group = VGroup(oracle_box, oracle_text)
        self.place_at_grid(oracle_group, "B2", scale_factor=0.6)
        
        diffusion_box = Rectangle(width=1.5, height=0.8, color=ORANGE).set_fill(DARK_BROWN, opacity=0.3)
        diffusion_text = Text("Diffusion", color=WHITE, font_size=24)
        diffusion_group = VGroup(diffusion_box, diffusion_text)
        self.place_at_grid(diffusion_group, "B5", scale_factor=0.6)
        
        # Loop arrow
        # Create arrows manually relative to grid points
        loop_arrow = CurvedArrow(self.grid["B5"] + UP*0.4, self.grid["B2"] + UP*0.4, angle=-TAU/4, color="#00FF00")
        loop_arrow_back = CurvedArrow(self.grid["B2"] + DOWN*0.4, self.grid["B5"] + DOWN*0.4, angle=-TAU/4, color="#00FF00")
        loop_group = VGroup(loop_arrow, loop_arrow_back)
        
        self.play(FadeIn(oracle_group), FadeIn(diffusion_group))
        self.play(Create(loop_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        bar_colors = [GRAY, GRAY, GREEN, GRAY]
        bar_heights_start = [0.4, 0.4, 0.8, 0.4]
        bar_heights_end = [0.1, 0.1, 2.0, 0.1]
        
        bars = VGroup(*[
            Rectangle(width=0.4, height=h, color=c, fill_opacity=0.8) 
            for h, c in zip(bar_heights_start, bar_colors)
        ]).arrange(RIGHT, buff=0.4)
        self.place_in_area(bars, "C2", "D5", scale_factor=0.8)
        
        # Move symbols to Row A to make room
        new_oracle = VGroup(oracle_box.copy(), oracle_text.copy())
        self.place_at_grid(new_oracle, "A2", scale_factor=0.6) # Issue 37 fix
        
        new_diffusion = VGroup(diffusion_box.copy(), diffusion_text.copy())
        self.place_at_grid(new_diffusion, "A5", scale_factor=0.6) # Issue 37 fix
        
        new_loop = loop_group.copy()
        # Scale and move loop relative to new positions
        self.place_in_area(new_loop, "A2", "A5", scale_factor=0.8)

        self.play(
            Transform(oracle_group, new_oracle),
            Transform(diffusion_group, new_diffusion),
            Transform(loop_group, new_loop),
            FadeIn(bars)
        )
        
        # Animate bars
        # Scaling about the bottom edge to avoid move_to
        self.play(
            *[bars[i].animate.stretch_to_fit_height(bar_heights_end[i], about_edge=DOWN) for i in range(4)],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        repeat_text = Text("Repeat approximately √N times", color="#ADD8E6", font_size=24)
        self.place_in_area(repeat_text, "E2", "E5", scale_factor=0.7) # Issue 36 fix
        
        self.play(Write(repeat_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        noise_bars = VGroup(bars[0], bars[1], bars[3])
        target_bar = bars[2]
        
        self.play(
            FadeOut(noise_bars),
            FadeOut(oracle_group),
            FadeOut(diffusion_group),
            FadeOut(loop_group),
            FadeOut(repeat_text),
            target_bar.animate.set_color(YELLOW).scale(1.2)
        )
        
        # Collapse line at the base of the bar
        base_y = target_bar.get_bottom()[1]
        center_x = target_bar.get_center()[0]
        collapse_line = Line(np.array([center_x - 1.5, base_y, 0]), np.array([center_x + 1.5, base_y, 0]), color=WHITE)
        self.play(Create(collapse_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        found_text = Text("Target Found!", color="#00FF00", font_size=40)
        self.place_in_area(found_text, "A2", "B5", scale_factor=0.8) # Issue 35 fix
        
        self.play(Write(found_text))
        
        # Flashing effect
        for _ in range(3):
            self.play(found_text.animate.set_opacity(0.3), run_time=0.2)
            self.play(found_text.animate.set_opacity(1.0), run_time=0.2)
            
        self.wait(2)
