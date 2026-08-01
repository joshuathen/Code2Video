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
        # Section Initialization
        title_text = "The Growth Paradox: A Rabbit Population"
        lecture_lines = [
            "In nature, growth often depends on current size.",
            "One cyber-rabbit splits into two, then four.",
            "More rabbits mean the population grows even faster."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Line 1: In nature, growth often depends on current size.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Initial rabbit circle
        rabbit = Circle(radius=0.25, color="#FFFFFF", fill_opacity=1)
        # Position in a larger area to better utilize space - Fixed per Issue 39
        self.place_in_area(rabbit, 'C1', 'E6', scale_factor=1.0)
        
        self.play(FadeIn(rabbit))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: One cyber-rabbit splits into two, then four.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Split 1 -> 2
        r1 = rabbit
        r2 = r1.copy()
        # Increased shift to utilize the full width of the right panel
        self.play(
            r1.animate.shift(LEFT * 1.2),
            r2.animate.shift(RIGHT * 1.2),
            run_time=0.8
        )
        group_2 = VGroup(r1, r2)
        self.wait(0.2)

        # Split 2 -> 4
        r3 = r1.copy()
        r4 = r2.copy()
        # Vertical dispersal
        self.play(
            group_2.animate.shift(UP * 0.7),
            VGroup(r3, r4).animate.shift(DOWN * 0.7),
            run_time=0.8
        )
        group_4 = VGroup(r1, r2, r3, r4)
        self.wait(0.2)

        # Split 4 -> 8 (Dispersing further horizontally)
        r_others = group_4.copy()
        self.play(
            group_4.animate.shift(LEFT * 0.8),
            r_others.animate.shift(RIGHT * 0.8),
            run_time=0.8
        )
        all_rabbits = VGroup(group_4, r_others)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: More rabbits mean the population grows even faster.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Display a green #00FF00 vector arrow above the group
        growth_arrow = Vector(RIGHT * 1.0, color="#00FF00")
        # Position above the group of rabbits
        growth_arrow.next_to(all_rabbits, UP, buff=0.5)
        
        self.play(Create(growth_arrow))
        
        # Scale the arrow significantly to visualize "faster" growth
        self.play(
            growth_arrow.animate.scale(4.0, about_edge=LEFT),
            run_time=2,
            rate_func=rush_into # Visualizes acceleration
        )
        
        self.wait(3)
