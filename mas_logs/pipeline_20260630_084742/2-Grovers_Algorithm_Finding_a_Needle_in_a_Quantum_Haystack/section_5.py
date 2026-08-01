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
        lecture_lines = [
            "Repeating these steps further increases the target's amplitude.",
            "Grover's algorithm finds the target in sqrt(N) steps.",
            "For a million items, we only need 1,000 queries.",
            "This provides a quadratic speedup over classical searching methods.",
            "Excessive iterations can eventually rotate away from the target."
        ]
        self.setup_layout("Iteration and the Speed Boost", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Circular arrow icon
        arrow_circle = Arc(radius=0.7, start_angle=0, angle=TAU*0.9, color="#00FF00")
        arrow_circle.add_tip(tip_length=0.2)
        repeat_text = Text("Repeat √N", color="#00FF00", font_size=28)
        arrow_group = VGroup(arrow_circle, repeat_text)
        # Fix: Issue 30
        self.place_in_area(arrow_group, "A4", "B6", scale_factor=0.8)
        
        self.play(Create(arrow_circle), Write(repeat_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"))
        
        # Bar Chart Setup
        bar_heights = [0.15, 0.15, 0.15, 0.2, 0.15, 0.15]
        target_idx = 3
        
        bars = VGroup()
        for i, h in enumerate(bar_heights):
            color = "#FFD700" if i == target_idx else "#555555"
            bar = Rectangle(width=0.4, height=h, fill_opacity=1, stroke_width=0, fill_color=color)
            bars.add(bar)
        
        bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        # Fix: Issue 31
        self.place_in_area(bars, "C2", "D5", scale_factor=0.9)
        
        self.play(FadeIn(bars))
        self.wait(0.5)
        
        # Scale the target bar to 1.0 (visually) while others vanish
        target_bar = bars[target_idx]
        other_bars = VGroup(*[b for i, b in enumerate(bars) if i != target_idx])
        
        self.play(
            target_bar.animate.stretch_to_fit_height(1.8, about_edge=DOWN),
            other_bars.animate.set_opacity(0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FF00"))
        
        # Comparison Table Text
        comp_text = Text(
            "1,000,000 items:\nClassical: 500,000\nQuantum: 1,000",
            color="#00FF00",
            font_size=22,
            line_spacing=0.8
        )
        # Fix: Issue 32
        self.place_in_area(comp_text, "E3", "F6", scale_factor=0.8)
        
        self.play(Write(comp_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color("#FFFF00"))
        
        # Detective icon [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/detect.svg]
        # Fix: Issue 22
        detective = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/detect.svg")
        detective.set_color("#FFFF00")
        # Position next to the tall gold bar in the grid
        self.place_at_grid(detective, "C6", scale_factor=0.4)
        
        self.play(FadeIn(detective))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#FF5555"))
        
        # Target bar height decreasing (over-rotation)
        self.play(
            target_bar.animate.stretch_to_fit_height(0.6, about_edge=DOWN),
            detective.animate.shift(DOWN * 0.6),
            run_time=2
        )
        self.wait(2)
