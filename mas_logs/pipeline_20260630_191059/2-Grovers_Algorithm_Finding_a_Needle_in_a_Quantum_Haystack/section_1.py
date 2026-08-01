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
        title = "The Classical Search Problem"
        lines = [
            "Searching an unsorted database is a fundamental challenge.",
            "We must check items one by one classically.",
            "Finding one item among N takes O(N) time."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_DEFAULT = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_TARGET = "#FFD700"
        COLOR_COMPLEXITY = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_DEFAULT))
        
        # Grid of 10 closed treasure chests labeled 'Search Space'
        search_space_label = Text("Search Space", font_size=24, color=COLOR_DEFAULT)
        # Resolved Issue 22: Centering search_space_label across columns
        self.place_in_area(search_space_label, "A2", "A6", scale_factor=0.8)
        
        chests = VGroup()
        positions = [
            "B1", "B2", "B3", "B4", "B5",
            "C1", "C2", "C3", "C4", "C5"
        ]
        
        for pos in positions:
            chest = RoundedRectangle(corner_radius=0.1, width=0.7, height=0.5, color=COLOR_DEFAULT)
            lid_line = Line(start=[-0.35, 0.1, 0], end=[0.35, 0.1, 0], color=COLOR_DEFAULT)
            chest_group = VGroup(chest, lid_line)
            self.place_at_grid(chest_group, pos, scale_factor=1.0)
            chests.add(chest_group)

        self.play(Write(search_space_label), FadeIn(chests))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_HIGHLIGHT))
        
        # Highlight chests one by one to show Alice checking them sequentially
        # We will highlight the first 6 chests sequentially
        for i in range(6):
            self.play(
                chests[i].animate.set_color(COLOR_HIGHLIGHT),
                run_time=0.4
            )
            self.play(
                chests[i].animate.set_color(COLOR_DEFAULT),
                run_time=0.2
            )
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_COMPLEXITY))
        
        # Change the 7th chest to #FFD700 (Gold)
        target_chest = chests[6]
        self.play(
            target_chest.animate.set_color(COLOR_TARGET),
            run_time=0.8
        )
        
        # Display 'O(N) Complexity'
        complexity_label = Text("O(N) Complexity", font_size=28, color=COLOR_COMPLEXITY)
        # Resolved Issue 23: Rebalancing complexity_label position and scale
        self.place_in_area(complexity_label, "E2", "F6", scale_factor=0.8)
        
        self.play(Write(complexity_label))
        self.wait(2)
