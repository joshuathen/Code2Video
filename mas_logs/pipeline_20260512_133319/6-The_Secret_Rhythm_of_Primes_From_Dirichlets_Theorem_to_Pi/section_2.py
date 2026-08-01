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
        lecture_lines = [
            'Dirichlet proved these valid lanes contain infinitely many primes.',
            'No matter how far we go, primes never stop appearing.',
            'Amazingly, primes distribute almost equally among all valid lanes.',
            'For difference four, 4n plus 1 and 4n plus 3 match.',
            'This theorem guarantees an infinite supply for our journey.'
        ]
        self.setup_layout("Dirichlet’s Theorem: The Guarantee of Infinity", lecture_lines)
        
        # Colors
        LANE_1_COLOR = "#00FFFF" # 4n+1
        LANE_2_COLOR = "#FF00FF" # 4n+3
        SPHERE_COLOR = "#FFD700" # Gold

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Conveyor Belts (Represented as Rails)
        lane1_rail = Line(self.grid["C1"], self.grid["C6"], color=LANE_1_COLOR, stroke_width=4)
        lane2_rail = Line(self.grid["E1"], self.grid["E6"], color=LANE_2_COLOR, stroke_width=4)
        
        label1 = Text("4n + 1", font_size=20, color=LANE_1_COLOR)
        label2 = Text("4n + 3", font_size=20, color=LANE_2_COLOR)
        self.place_at_grid(label1, "C2", scale_factor=1.0)
        self.place_at_grid(label2, "E2", scale_factor=1.0)

        self.play(
            Create(lane1_rail),
            Create(lane2_rail),
            FadeIn(label1),
            FadeIn(label2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Counters
        count1 = 0
        count2 = 0
        counter1_text = Text(f"Count: {count1}", font_size=18, color=LANE_1_COLOR)
        counter2_text = Text(f"Count: {count2}", font_size=18, color=LANE_2_COLOR)
        self.place_at_grid(counter1_text, "C6", scale_factor=1.0).shift(RIGHT * 0.8)
        self.place_at_grid(counter2_text, "E6", scale_factor=1.0).shift(RIGHT * 0.8)
        self.add(counter1_text, counter2_text)

        primes_to_sort = [
            (5, 1), (7, 2), (11, 2), (13, 1), (17, 1), (19, 2)
        ]

        # Sorting logic loop
        for p_val, lane_num in primes_to_sort:
            # Create prime sphere
            sphere = VGroup(
                Circle(radius=0.25, color=SPHERE_COLOR, fill_opacity=0.8),
                Text(str(p_val), font_size=16, color=BLACK)
            )
            self.place_at_grid(sphere, "A3")
            
            # Animation sequence
            dest_row = "C" if lane_num == 1 else "E"
            
            self.play(FadeIn(sphere), run_time=0.3)
            # Fall to lane level
            self.play(sphere.animate.move_to(self.grid[f"{dest_row}3"]), run_time=0.4)
            # Move along lane
            self.play(sphere.animate.move_to(self.grid[f"{dest_row}6"]), run_time=0.6)
            
            # Update counters
            if lane_num == 1:
                count1 += 1
                new_counter = Text(f"Count: {count1}", font_size=18, color=LANE_1_COLOR)
                self.place_at_grid(new_counter, "C6", scale_factor=1.0).shift(RIGHT * 0.8)
            else:
                count2 += 1
                new_counter = Text(f"Count: {count2}", font_size=18, color=LANE_2_COLOR)
                self.place_at_grid(new_counter, "E6", scale_factor=1.0).shift(RIGHT * 0.8)
            
            self.play(
                FadeOut(sphere),
                Transform(counter1_text if lane_num == 1 else counter2_text, new_counter),
                run_time=0.2
            )

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Visualizing "Equal distribution"
        # We'll show a quick wave of many smaller dots to simulate a flow
        dots_1 = VGroup(*[Circle(radius=0.05, color=LANE_1_COLOR, fill_opacity=1) for _ in range(10)])
        dots_2 = VGroup(*[Circle(radius=0.05, color=LANE_2_COLOR, fill_opacity=1) for _ in range(10)])
        
        for i, d in enumerate(dots_1):
            d.move_to(self.grid["C1"] + RIGHT * (i * 0.4))
        for i, d in enumerate(dots_2):
            d.move_to(self.grid["E1"] + RIGHT * (i * 0.4))
            
        self.play(FadeIn(dots_1), FadeIn(dots_2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        highlight_rect = Rectangle(height=2.5, width=4, color=WHITE, stroke_opacity=0.5)
        self.place_in_area(highlight_rect, "B2", "E6")
        
        self.play(Create(highlight_rect), run_time=1)
        self.play(Indicate(label1), Indicate(label2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Camera pan simulation: move elements to the left to show "infinity"
        infinity_symbol = Text("...", font_size=40, color=WHITE)
        self.place_at_grid(infinity_symbol, "D6")
        
        self.play(
            VGroup(lane1_rail, lane2_rail, dots_1, dots_2, highlight_rect, label1, label2, counter1_text, counter2_text).animate.shift(LEFT * 1.5),
            FadeIn(infinity_symbol, shift=LEFT),
            run_time=2
        )
        self.wait(2)
