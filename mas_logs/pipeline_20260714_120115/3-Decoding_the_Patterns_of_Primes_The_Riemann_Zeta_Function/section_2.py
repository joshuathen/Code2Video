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
        # Section Title and Lecture Lines
        title_text = "Prerequisite: The Power of Infinite Sums"
        lecture_lines = [
            "Adding infinite fractions can lead to a precise result.",
            "The sum of reciprocal squares equals Pi squared over six.",
            "This beautiful discovery is the simplest Zeta function form."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Using Text instead of MathTex to avoid 'latex' dependency error
        sum_formula = Text(
            "1 + 1/4 + 1/9 + 1/16 + ...", 
            color=WHITE
        )
        # Using place_in_area for wide formula as per L015 and L010 (avoid Col 1)
        self.place_in_area(sum_formula, "B2", "B5", scale_factor=0.9)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Write(sum_formula),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stack colorful blocks [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/blocks.svg] 
        # with decreasing heights matching the terms.
        
        asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/blocks.svg"
        
        # Create blocks with heights proportional to 1/n^2
        n_terms = 5
        height_values = [1/(n**2) for n in range(1, n_terms + 1)]
        colors = [RED_A, ORANGE, YELLOW, GREEN_A, BLUE_A]
        
        blocks = VGroup()
        for i in range(n_terms):
            # Load asset and scale it proportionally
            block = SVGMobject(asset_path)
            # Base scale for the first block (n=1)
            block.scale(1.2 * height_values[i])
            block.set_color(colors[i])
            blocks.add(block)
        
        # Stack them vertically with the largest at the bottom
        blocks.arrange(UP, buff=0.05)
        
        # Place the stack in the grid area, avoiding Row A (L001)
        self.place_in_area(blocks, "C3", "E4", scale_factor=1.0)
        
        self.play(
            self.lecture[1].animate.set_color(YELLOW),
            FadeIn(blocks, shift=UP),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A dashed horizontal line appears at height (PI^2)/6 in #00FFFF.
        
        # Calculate the total limit height relative to the first block's height
        unit_height = blocks[0].get_height()
        total_limit_height = unit_height * (np.pi**2 / 6)
        
        # Find the base of the stack to position the line
        bottom_y = blocks.get_bottom()[1]
        line_y = bottom_y + total_limit_height
        
        # Create dashed line spanning columns 2 to 5
        dashed_line = DashedLine(
            start=[self.grid["C2"][0], line_y, 0],
            end=[self.grid["C5"][0], line_y, 0],
            color="#00FFFF",
            dash_length=0.1
        )
        
        # Using Text with Unicode instead of MathTex to avoid 'latex' dependency
        limit_label = Text("π²/6", color="#00FFFF")
        # Position label in the next column
        self.place_at_grid(limit_label, "C6", scale_factor=0.8)
        # Adjust Y to align precisely with the dashed line
        limit_label.set_y(line_y)
        
        self.play(
            self.lecture[2].animate.set_color("#00FFFF"),
            Create(dashed_line),
            Write(limit_label),
            run_time=2
        )
        self.wait(3)
