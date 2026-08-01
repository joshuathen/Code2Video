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
        # Setup initial layout
        lecture_lines = [
            "Meet Lucky the Squirrel, our jump trial athlete.",
            "Every jump has exactly two possible outcomes.",
            "Success or failure; there is no middle ground."
        ]
        self.setup_layout("Prerequisite: The Bernoulli Trial", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create visual platforms
        ledge_left = Rectangle(width=1.8, height=0.4, color=GREY_B, fill_opacity=0.8)
        self.place_in_area(ledge_left, 'D1', 'D2')
        
        ledge_right = Rectangle(width=1.8, height=0.4, color=GREY_B, fill_opacity=0.8)
        self.place_in_area(ledge_right, 'D5', 'D6')
        
        # Lucky the Squirrel [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg]
        lucky = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg")
        lucky.set_color(WHITE)
        self.place_at_grid(lucky, 'C2', scale_factor=0.6)
        
        lucky_label = Text("Lucky", font_size=20, color=WHITE)
        lucky_label.next_to(lucky, UP, buff=0.1)
        
        lucky_group = VGroup(lucky, lucky_label)
        
        self.play(
            Create(ledge_left),
            Create(ledge_right),
            FadeIn(lucky_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        
        # Bernoulli Trial label at top
        bernoulli_text = Text("Bernoulli Trial: Two Outcomes Only", font_size=24, color="#FFFF00")
        # Fix Issue 31: Positioned at A2-A6
        self.place_in_area(bernoulli_text, 'A2', 'A6', scale_factor=0.7)
        
        self.play(Write(bernoulli_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Labels for outcomes
        success_label = Text("Success (p)", font_size=24, color="#00FF00")
        # Fix Issue 30: Positioned at C6
        self.place_at_grid(success_label, 'C6', scale_factor=0.8)
        
        failure_label = Text("Failure (1-p)", font_size=24, color="#FF0000")
        # Fix Issue 29: Positioned at E5
        self.place_at_grid(failure_label, 'E5', scale_factor=0.8)
        
        # Success Path
        arc_success = ArcBetweenPoints(
            start=self.grid['C2'],
            end=self.grid['C5'],
            angle=-TAU/4
        )
        
        # Arrows as requested by animation description
        arrow_success = Arrow(start=self.grid['C2'], end=self.grid['B5'], color="#00FF00", buff=0.2)
        arrow_failure = Arrow(start=self.grid['C2'], end=self.grid['E4'], color="#FF0000", buff=0.2)

        self.play(
            Create(arrow_success),
            MoveAlongPath(lucky_group, arc_success),
            FadeIn(success_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Return to start
        self.play(
            lucky_group.animate.move_to(self.grid['C2']),
            FadeOut(arrow_success),
            run_time=0.8
        )
        
        # Failure Path
        arc_failure = ArcBetweenPoints(
            start=self.grid['C2'],
            end=self.grid['E5'],
            angle=-TAU/8
        )
        
        self.play(
            Create(arrow_failure),
            MoveAlongPath(lucky_group, arc_failure),
            FadeIn(failure_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Final emphasis
        self.play(
            Indicate(success_label, color="#00FF00"),
            Indicate(failure_label, color="#FF0000"),
            run_time=2
        )
        
        self.wait(2)
