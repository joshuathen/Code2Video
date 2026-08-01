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

class Section3Scene(TeachingScene):
    def construct(self):
        # Updated lecture lines per prompt requirements
        lecture_lines = [
            'A subtle race exists between these two infinite teams.',
            'The silver team usually maintains a persistent lead.',
            'This mysterious preference is called Chebyshev’s Bias.'
        ]
        self.setup_layout("The Prime Race: Chebyshev’s Bias", lecture_lines)

        # Colors
        GOLD_COLOR = "#FFD700"
        SILVER_COLOR = "#C0C0C0"
        CYAN_COLOR = "#00FFFF"

        # Asset paths
        scoreboard_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/scoreboard.svg"
        counter_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/counter.svg"

        # Define scoreboard components
        self.scoreboard_bg = SVGMobject(scoreboard_path).set_color(WHITE)
        self.place_in_area(self.scoreboard_bg, "A2", "E6")
        
        self.label_4k1 = Text("Team 4k+1", color=GOLD_COLOR, font_size=24)
        self.place_at_grid(self.label_4k1, "C3")
        
        self.label_4k3 = Text("Team 4k+3", color=SILVER_COLOR, font_size=24)
        self.place_at_grid(self.label_4k3, "C5")

        self.counter_icon1 = SVGMobject(counter_path).set_color(GOLD_COLOR)
        self.place_at_grid(self.counter_icon1, "D3", scale_factor=0.8)
        
        self.counter_icon2 = SVGMobject(counter_path).set_color(SILVER_COLOR)
        self.place_at_grid(self.counter_icon2, "D5", scale_factor=0.8)

        self.count_4k1 = Text("0", color=GOLD_COLOR, font_size=28)
        self.place_at_grid(self.count_4k1, "D3")

        self.count_4k3 = Text("0", color=SILVER_COLOR, font_size=28)
        self.place_at_grid(self.count_4k3, "D5")

        self.bias_title = Text("Chebyshev's Bias", color=CYAN_COLOR, font_size=32)
        self.place_in_area(self.bias_title, "A3", "B5")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Animation 1: Digital scoreboard and labels appear
        self.play(
            FadeIn(self.scoreboard_bg),
            FadeIn(self.label_4k1),
            FadeIn(self.label_4k3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SILVER_COLOR)
        
        # Animation 2: Both counters rapidly increase
        self.play(
            FadeIn(self.counter_icon1), 
            FadeIn(self.counter_icon2), 
            FadeIn(self.count_4k1), 
            FadeIn(self.count_4k3)
        )
        
        # Simulated fast counting sequence to ~1 million
        intermediate_values = [
            ("124,532", "124,541"),
            ("489,201", "489,235"),
            ("812,045", "812,098"),
            ("999,958", "1,000,042")
        ]

        for val1, val3 in intermediate_values:
            new_count1 = Text(val1, color=GOLD_COLOR, font_size=28)
            new_count3 = Text(val3, color=SILVER_COLOR, font_size=28)
            self.place_at_grid(new_count1, "D3")
            self.place_at_grid(new_count3, "D5")
            
            self.play(
                Transform(self.count_4k1, new_count1),
                Transform(self.count_4k3, new_count3),
                run_time=0.4
            )

        # Highlight the persistent lead of Team 4k+3
        rect_lead = SurroundingRectangle(self.count_4k3, color=SILVER_COLOR, buff=0.1)
        self.play(Create(rect_lead))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CYAN_COLOR)
        
        # Animation 3: 'Chebyshev's Bias' appears in glowing text
        self.play(Write(self.bias_title))
        self.play(self.bias_title.animate.set_stroke(CYAN_COLOR, width=2))
        self.play(Indicate(self.bias_title, color=CYAN_COLOR))
        self.wait(2)
