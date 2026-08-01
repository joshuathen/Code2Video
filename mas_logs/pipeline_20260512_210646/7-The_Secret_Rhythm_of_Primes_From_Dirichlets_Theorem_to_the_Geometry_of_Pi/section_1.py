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
        # Setup the scene layout with title and updated lecture lines
        title_str = "Prerequisite: The Modulo Sorting Machine"
        lines_str = [
            'Imagine odd numbers flowing into a sorting machine.',
            'We divide every number by four to find remainders.',
            'This creates two distinct groups of numbers.',
            'Gold bucket numbers leave a remainder of one.',
            'Silver bucket numbers leave a remainder of three.'
        ]
        self.setup_layout(title_str, lines_str)
        
        # Define colors as specified in animation description
        BOX_COLOR = "#00FF00"   # Modulo 4 Box
        GOLD_COLOR = "#FFD700"  # Gold Bucket
        SILVER_COLOR = "#C0C0C0" # Silver Bucket

        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Animation 1: A horizontal row of odd numbers (3, 5, 7, 9, 11, 13) enters from the left
        nums = [3, 5, 7, 9, 11, 13]
        num_mobs = VGroup(*[Text(str(n), font_size=36) for n in nums])
        
        # Initial positions: Position off-screen to the left relative to B1
        for mob in num_mobs:
            mob.move_to(self.grid["B1"] + LEFT * 4)
            
        self.play(
            AnimationGroup(*[
                mob.animate.move_to(self.grid[f"B{i+1}"])
                for i, mob in enumerate(num_mobs)
            ], lag_ratio=0.1),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight current lecture line
        self.play(self.lecture[1].animate.set_color(BOX_COLOR))
        
        # Animation 2: A central processing box labeled 'Modulo 4' appears in the middle
        # Fix: Issue 31 - place_in_area(box_group, 'B3', 'C4', scale_factor=0.9)
        box_rect = Rectangle(width=2.5, height=1.6, color=BOX_COLOR)
        box_label = Text("Modulo 4", font_size=22, color=BOX_COLOR)
        box_group = VGroup(box_rect, box_label)
        self.place_in_area(box_group, "B3", "C4", scale_factor=0.9)
        
        self.play(FadeIn(box_group))
        
        # Visual transition: Numbers move into the sorting box for "processing"
        box_center = box_group.get_center()
        self.play(
            num_mobs.animate.scale(0.5).move_to(box_center).set_opacity(0.4),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight current lecture line
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Animation 3: Two containers appear
        # Fix: Issue 30 - place_in_area(gold_bucket, 'E2', 'F3', scale_factor=0.8)
        gold_rect = Rectangle(width=2.3, height=1.6, color=GOLD_COLOR)
        gold_label = Text("Gold Bucket\n(4k+1)", font_size=16, color=GOLD_COLOR, line_spacing=0.8)
        gold_bucket = VGroup(gold_rect, gold_label)
        self.place_in_area(gold_bucket, "E2", "F3", scale_factor=0.8)
        
        # Fix: Issue 32 - place_in_area(silver_bucket, 'E5', 'F6', scale_factor=0.8)
        silver_rect = Rectangle(width=2.3, height=1.6, color=SILVER_COLOR)
        silver_label = Text("Silver Bucket\n(4k+3)", font_size=16, color=SILVER_COLOR, line_spacing=0.8)
        silver_bucket = VGroup(silver_rect, silver_label)
        self.place_in_area(silver_bucket, "E5", "F6", scale_factor=0.8)
        
        self.play(FadeIn(gold_bucket), FadeIn(silver_bucket))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Highlight current lecture line
        self.play(self.lecture[3].animate.set_color(GOLD_COLOR))
        
        # Animation 4: The numbers 5, 9, and 13 move from the center and drop into the Gold Bucket
        gold_indices = [1, 3, 5]
        gold_targets = VGroup(*[num_mobs[i] for i in gold_indices])
        
        self.play(
            gold_targets.animate.set_opacity(1).scale(1.4).arrange(RIGHT, buff=0.3).move_to(gold_rect.get_center()),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Highlight current lecture line
        self.play(self.lecture[4].animate.set_color(SILVER_COLOR))
        
        # Animation 5: The numbers 3, 7, and 11 move from the center and drop into the Silver Bucket
        silver_indices = [0, 2, 4]
        silver_targets = VGroup(*[num_mobs[i] for i in silver_indices])
        
        self.play(
            silver_targets.animate.set_opacity(1).scale(1.4).arrange(RIGHT, buff=0.3).move_to(silver_rect.get_center()),
            run_time=2
        )
        self.wait(2)
