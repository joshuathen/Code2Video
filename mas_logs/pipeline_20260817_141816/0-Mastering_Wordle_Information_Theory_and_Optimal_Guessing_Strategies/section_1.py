from manim import *
import os

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
            "Wordle is a hidden information game.",
            "Maximize information gain to narrow search space.",
            "Twelve thousand guesses versus two thousand solutions."
        ]
        self.setup_layout("Introduction: The Goal and The Constraints", lecture_lines)
        
        # Placeholder for wordle_grid_initial
        wordle_grid = Rectangle(width=3, height=3, color=BLUE)
        wordle_grid_label = Text("Grid", font_size=20).move_to(wordle_grid.get_center())
        grid_group = VGroup(wordle_grid, wordle_grid_label)
        
        # Placeholder for wordle_bot_icon
        wordle_bot = Circle(radius=0.5, color=YELLOW).add(Dot().move_to(ORIGIN))
        wordle_bot_label = Text("Bot", font_size=20).next_to(wordle_bot, DOWN)
        bot_group = VGroup(wordle_bot, wordle_bot_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(grid_group, 'A1', 'B6', scale_factor=0.6)
        self.play(FadeIn(grid_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(bot_group, 'D2', scale_factor=0.7)
        self.play(FadeIn(bot_group))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        stats = VGroup(
            Text("Guesses: 12,972", font_size=24, color=BLUE),
            Text("Solutions: 2,315", font_size=24, color=GREEN)
        ).arrange(DOWN)
        self.place_in_area(stats, 'C3', 'D6', scale_factor=0.7)
        self.play(Write(stats))
        
        self.wait(2)
