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
        lecture_lines = [
            "Imagine two blocks sliding on a frictionless floor.",
            "A wall sits on the far left side.",
            "A large block collides with a stationary small block.",
            "We count every collision between blocks and the wall.",
            "Watch as the collision count reveals digits of Pi."
        ]
        self.setup_layout("The Setup: A Curious Experiment", lecture_lines)

        # Asset Paths
        wall_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        block_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Floor Line
        # We place it relative to the grid row E
        floor_y = self.grid["E1"][1] - 0.5
        floor = Line(
            np.array([self.grid["E1"][0] - 0.5, floor_y, 0]),
            np.array([self.grid["E6"][0] + 0.5, floor_y, 0]),
            color=WHITE
        )
        
        # Small block 'm' - Issue 34: scale_factor=0.7, pos 'E2'
        small_block = SVGMobject(block_asset).set_color("#00FF00")
        self.place_at_grid(small_block, "E2", scale_factor=0.7)
        small_label = MathTex("m", color="#00FF00").scale(0.8)
        small_label.next_to(small_block, UP, buff=0.1)
        
        # Large block 'M' - Issue 33: scale_factor=1.5, pos 'E6'
        large_block = SVGMobject(block_asset).set_color("#0000FF")
        self.place_at_grid(large_block, "E6", scale_factor=1.5)
        large_label = MathTex("M", color="#0000FF").scale(0.8)
        large_label.next_to(large_block, UP, buff=0.1)

        self.play(
            Create(floor), 
            FadeIn(small_block), 
            Write(small_label), 
            FadeIn(large_block), 
            Write(large_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Wall - Issue 27
        wall = SVGMobject(wall_asset).set_color("#FFFFFF")
        # Place wall in the first column area to the left of the blocks
        self.place_in_area(wall, "C1", "E1", scale_factor=1.2)
        # Adjust wall to sit on the floor
        wall.shift(DOWN * (wall.get_bottom()[1] - floor_y))
        
        self.play(FadeIn(wall))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Large block moves toward small block
        # Updaters for labels to follow blocks
        large_label.add_updater(lambda m: m.next_to(large_block, UP, buff=0.1))
        
        target_pos_x = self.grid["E3"][0]
        self.play(
            large_block.animate.set_x(target_pos_x),
            run_time=2,
            rate_func=linear
        )
        large_label.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Counter appears - Issue 32: place at 'A5'
        counter_label = Text("Collisions:", font_size=24, color="#FFFF00")
        self.place_at_grid(counter_label, "A5")
        counter_val = Integer(0, color="#FFFF00")
        counter_val.next_to(counter_label, RIGHT, buff=0.2)
        
        self.play(Write(counter_label), Write(counter_val))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Flash values 3, 31, 314
        for val in [3, 31, 314]:
            self.play(
                counter_val.animate.set_value(val),
                Flash(counter_val, color="#FFFF00", flash_radius=0.4),
                run_time=1
            )
            self.wait(0.5)

        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
