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

class Section4Scene(TeachingScene):
    def construct(self):
        # 1. Setup the UI and layout
        self.setup_layout(
            "The Pi Collision Puzzle",
            [
                "Phase 4: Counting Bounces",
                "- System: Wall + Two Blocks",
                "- Mass Ratio M = 100^n * m",
                "- Elastic collisions only",
                "- Total bounces = Digits of Pi"
            ]
        )

        # 2. Create the environment (Wall and Floor)
        wall = Line(UP, DOWN, color=GREY, stroke_width=6).scale(2)
        self.place_at_grid(wall, "C1")
        wall.shift(LEFT * 0.4)

        floor = Line(LEFT, RIGHT, color=GREY, stroke_width=4).scale(3.5)
        self.place_at_grid(floor, "E3")
        floor.shift(DOWN * 0.6 + RIGHT * 0.5)

        # 3. Create the Blocks
        # Small block (m)
        block_m = Square(side_length=0.6, fill_opacity=0.8, color=BLUE)
        self.place_at_grid(block_m, "D2")
        
        # Large block (M)
        block_M = Square(side_length=1.2, fill_opacity=0.8, color=RED)
        self.place_at_grid(block_M, "D5")

        # Labels
        m_label = MathTex("m", font_size=24).next_to(block_m, UP)
        M_label = MathTex("M = 100m", font_size=24).next_to(block_M, UP)

        self.play(
            Create(wall), 
            Create(floor), 
            FadeIn(block_m, shift=UP), 
            FadeIn(block_M, shift=UP),
            Write(m_label),
            Write(M_label)
        )

        # 4. Counter mechanism
        collision_count = 0
        counter_label = Text("Bounces: ", font_size=32, color=WHITE)
        self.place_at_grid(counter_label, "A4")
        counter_label.shift(LEFT * 0.5)
        
        count_val = Integer(collision_count).next_to(counter_label, RIGHT).set_color(YELLOW)
        counter_group = VGroup(counter_label, count_val)
        self.add(counter_group)

        # 5. Animation Sequence (Demonstrating first 3 collisions)
        # Collision 1: M hits m
        self.play(block_M.animate.next_to(block_m, RIGHT, buff=0), run_time=1.2)
        collision_count += 1
        count_val.set_value(collision_count)
        self.play(Indicate(count_val, scale_factor=1.5, color=YELLOW))

        # Collision 2: m hits wall
        self.play(block_m.animate.next_to(wall, RIGHT, buff=0), run_time=0.6)
        collision_count += 1
        count_val.set_value(collision_count)
        self.play(Indicate(count_val, scale_factor=1.5, color=YELLOW))

        # Collision 3: m hits M
        self.play(
            block_m.animate.next_to(block_M, LEFT, buff=0),
            block_M.animate.shift(RIGHT * 0.2),
            run_time=0.6
        )
        collision_count += 1
        count_val.set_value(collision_count)
        self.play(Indicate(count_val, scale_factor=1.5, color=YELLOW))

        # 6. Conclusion
        conclusion = Text("Total: 31 bounces for n=1", font_size=26, color=GREEN_B)
        self.place_at_grid(conclusion, "F4")
        self.play(Write(conclusion))

        self.wait(3)
