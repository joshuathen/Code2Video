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
        # Setup Layout
        title_str = "Prerequisite: Fermat's Principle"
        lines = [
            "Light follows the path of least time.",
            "Speed changes when light enters a new medium.",
            "The fastest path isn't always a straight line."
        ]
        self.setup_layout(title_str, lines)

        # Colors for matching elements
        SAND_COLOR = "#F4A460"
        WATER_COLOR = "#1E90FF"
        TEXT_HIGHLIGHT = "#FFFFFF"
        PATH_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(TEXT_HIGHLIGHT))
        
        # Principle Statement
        principle_text = Text("Fermat's Principle: Least Time", font_size=32, color=TEXT_HIGHLIGHT)
        self.place_in_area(principle_text, "A2", "B5", scale_factor=0.8) # Issue 32 fix
        self.play(Write(principle_text))
        self.wait(1)
        self.play(FadeOut(principle_text))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(SAND_COLOR))
        
        # Split screen backgrounds
        sand_rect = Rectangle(width=6.0, height=2.5, fill_color=SAND_COLOR, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(sand_rect, "A1", "C6")
        
        water_rect = Rectangle(width=6.0, height=2.5, fill_color=WATER_COLOR, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(water_rect, "D1", "F6")
        
        # Interface line between Row C and Row D
        mid_left = (self.grid["C1"] + self.grid["D1"]) / 2
        mid_right = (self.grid["C6"] + self.grid["D6"]) / 2
        interface_line = Line(start=mid_left, end=mid_right, color=WHITE, stroke_width=2)
        
        sand_label = Text("Sand (Fast)", font_size=20, color=SAND_COLOR)
        self.place_at_grid(sand_label, "B6", scale_factor=0.8) # Issue 34 fix
        
        water_label = Text("Water (Slow)", font_size=20, color=WATER_COLOR)
        self.place_at_grid(water_label, "E6", scale_factor=0.8) # Issue 33 fix

        self.play(
            FadeIn(sand_rect),
            FadeIn(water_rect),
            Create(interface_line),
            Write(sand_label),
            Write(water_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PATH_COLOR))

        # Positions using grid anchors
        start_pt = self.grid["A2"]
        end_pt = self.grid["F5"]
        interface_y = (self.grid["C1"][1] + self.grid["D1"][1]) / 2
        
        # Straight-line intersection
        straight_int = np.array([3.0, interface_y, 0])
        # Optimal path (least time) intersection: minimize travel in slow medium
        # Swimmer is at Col 5, so bend at Col 5 minimizes water distance.
        bent_int = np.array([self.grid["C5"][0], interface_y, 0]) 

        lifeguard_icon = Dot(start_pt, color=PATH_COLOR, radius=0.15)
        lifeguard_label = Text("Lifeguard", font_size=16, color=WHITE).next_to(lifeguard_icon, UP, buff=0.1)
        
        swimmer_icon = Dot(end_pt, color=WHITE, radius=0.1)
        swimmer_label = Text("Swimmer", font_size=16, color=WHITE).next_to(swimmer_icon, DOWN, buff=0.1)

        straight_path = DashedLine(start_pt, end_pt, color=GRAY, stroke_opacity=0.5)
        optimal_path = VGroup(
            Line(start_pt, bent_int, color=PATH_COLOR),
            Line(bent_int, end_pt, color=PATH_COLOR)
        )

        self.play(
            FadeIn(lifeguard_icon),
            Write(lifeguard_label),
            FadeIn(swimmer_icon),
            Write(swimmer_label)
        )
        self.play(Create(straight_path))
        self.wait(0.5)
        
        self.play(Create(optimal_path))
        
        # Animation of movement along optimal path
        # Running on sand (Fast)
        self.play(
            lifeguard_icon.animate.move_to(bent_int),
            lifeguard_label.animate.move_to(bent_int + UP * 0.3),
            run_time=1.5,
            rate_func=linear
        )
        # Swimming in water (Slow)
        self.play(
            lifeguard_icon.animate.move_to(end_pt),
            lifeguard_label.animate.move_to(end_pt + UP * 0.3),
            run_time=2.5,
            rate_func=linear
        )
        
        self.wait(2)
