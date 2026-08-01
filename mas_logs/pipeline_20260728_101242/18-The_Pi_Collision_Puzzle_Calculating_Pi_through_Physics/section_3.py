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
        title = "The Pattern Emerges: Counting the Clacks"
        lines = [
            "Let's vary the mass of the large block.",
            "With equal masses, we count exactly three collisions.",
            "Increase the mass ratio to one hundred to one.",
            "Now we count thirty-one collisions.",
            "The digits of Pi begin to emerge clearly."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_M = "#3498DB" # Blue
        COLOR_m = "#F1C40F" # Yellow
        COLOR_COUNTER = "#FF4D4F" # Red
        COLOR_HIGHLIGHT = "#FF4D4F"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_M)
        
        # Physical Setup
        floor = Line(self.grid["E1"] + LEFT*0.4, self.grid["E6"] + RIGHT*0.4, color=GRAY)
        wall = Line(self.grid["B1"] + LEFT*0.4, self.grid["E1"] + LEFT*0.4, color=GRAY)
        
        # Blocks (Assets) - Issue 23
        block_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        block_m = SVGMobject(block_svg_path).set_color(COLOR_m)
        block_M = SVGMobject(block_svg_path).set_color(COLOR_M)
        
        # Place blocks on the grid
        self.place_at_grid(block_m, "E2", scale_factor=0.3)
        self.place_at_grid(block_M, "E4", scale_factor=0.45)
        
        # Labels
        label_m = Text("m", font_size=20).next_to(block_m, UP, buff=0.1)
        label_M_val = Text("M = m", font_size=20).next_to(block_M, UP, buff=0.1)
        
        # Counter
        counter_label = Text("Collisions:", font_size=24, color=WHITE)
        self.place_at_grid(counter_label, "B3")
        
        # Fixed: Issue 28 - move counter_val to B4 for better proximity to label
        counter_val = Text("0", font_size=36, color=COLOR_COUNTER)
        self.place_at_grid(counter_val, "B4")

        self.play(
            Create(floor),
            Create(wall),
            FadeIn(block_m),
            FadeIn(block_M),
            Write(label_m),
            Write(label_M_val),
            FadeIn(counter_label),
            FadeIn(counter_val)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_m)
        
        # M = m -> 3 collisions
        new_counter_val_3 = Text("3", font_size=36, color=COLOR_COUNTER)
        self.place_at_grid(new_counter_val_3, "B4") # Consistent with Issue 28
        
        self.play(
            Transform(counter_val, new_counter_val_3),
            Flash(new_counter_val_3, color=COLOR_COUNTER)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_M)
        
        # Increase ratio to 100:1
        new_label_M_100 = Text("M = 100m", font_size=20).next_to(block_M, UP, buff=0.1)
        
        self.play(
            Transform(label_M_val, new_label_M_100)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_COUNTER)
        
        # 31 collisions
        new_counter_val_31 = Text("31", font_size=36, color=COLOR_COUNTER)
        self.place_at_grid(new_counter_val_31, "B4") # Consistent with Issue 28
        
        self.play(
            Transform(counter_val, new_counter_val_31),
            Flash(new_counter_val_31, color=COLOR_COUNTER)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # M = 10,000m -> 314 collisions
        new_label_M_10k = Text("M = 10,000m", font_size=20).next_to(block_M, UP, buff=0.1)
        new_counter_val_314 = Text("314", font_size=36, color=COLOR_COUNTER)
        self.place_at_grid(new_counter_val_314, "B4") # Consistent with Issue 28
        
        self.play(
            Transform(label_M_val, new_label_M_10k),
            Transform(counter_val, new_counter_val_314),
            run_time=1.5
        )
        self.wait(0.5)
        
        # M = 1,000,000m -> 3141 collisions
        # Fixed: Issue 29 - Keep centered over E4 and counter at B4
        new_label_M_1M = Text("M = 1,000,000m", font_size=20).next_to(block_M, UP, buff=0.1)
        new_counter_val_3141 = Text("3141", font_size=36, color=COLOR_COUNTER)
        self.place_at_grid(new_counter_val_3141, "B4")
        
        self.play(
            Transform(label_M_val, new_label_M_1M),
            Transform(counter_val, new_counter_val_3141),
            run_time=1.5
        )
        self.wait(1)
        
        # Highlight pattern flash (Issue 23/Storyboard)
        self.play(
            Indicate(counter_val, color=COLOR_HIGHLIGHT, scale_factor=1.2)
        )
        self.wait(2)
