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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis: Solving the 3-Disk Puzzle", [
            "Increment the binary counter to identify each move.",
            "The rightmost bit flip determines which disk moves.",
            "Step four flips the third bit for the largest disk.",
            "Follow the sequence to reach the final state.",
            "Seven moves complete the three-disk puzzle."
        ])

        # Colors
        COLOR_HIGHLIGHT = "#00FFFF"
        COLOR_GOLD = "#FFD700"
        DISK_COLORS = [BLUE_B, GREEN_B, RED_B]

        # Asset Paths
        TOWER_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg"
        DISK_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)

        # Counter setup - Adjusted per Issues 42, 43
        counter_label = Text("Binary Counter", font_size=24, color=WHITE)
        self.place_in_area(counter_label, "A2", "A3", scale_factor=0.8)
        
        bits_val = "000"
        bits_vg = VGroup(*[Text(char, font_size=40) for char in bits_val]).arrange(RIGHT, buff=0.4)
        self.place_in_area(bits_vg, "B2", "B3", scale_factor=1.0)

        # Towers Header setup - Adjusted per Issue 41
        towers_label = Text("Towers", font_size=24, color=WHITE)
        tower_icon = SVGMobject(TOWER_ASSET).set_color(WHITE).set_height(0.5)
        tower_group = VGroup(towers_label, tower_icon).arrange(RIGHT, buff=0.3)
        # Using suggested area but manually shifting up to act as a header for the rods below
        self.place_in_area(tower_group, "B4", "F6", scale_factor=0.8)
        tower_group.shift(UP * 2.0)

        # Pegs and Disks setup
        # Use grid columns 4, 5, 6 for pegs A, B, C
        peg_a_x = self.grid["F4"][0]
        peg_b_x = self.grid["F5"][0]
        peg_c_x = self.grid["F6"][0]
        base_y = self.grid["F1"][1]
        top_y = self.grid["B1"][1]

        peg_a = Line([peg_a_x, base_y, 0], [peg_a_x, top_y, 0], color=GREY_C, stroke_width=4)
        peg_b = Line([peg_b_x, base_y, 0], [peg_b_x, top_y, 0], color=GREY_C, stroke_width=4)
        peg_c = Line([peg_c_x, base_y, 0], [peg_c_x, top_y, 0], color=GREY_C, stroke_width=4)
        base_line = Line([peg_a_x - 0.5, base_y, 0], [peg_c_x + 0.5, base_y, 0], color=GREY_C, stroke_width=6)

        # Create Disks using Assets
        d1 = SVGMobject(DISK_ASSET).set_color(DISK_COLORS[0]).set_width(0.6).set_height(0.3)
        d2 = SVGMobject(DISK_ASSET).set_color(DISK_COLORS[1]).set_width(0.9).set_height(0.3)
        d3 = SVGMobject(DISK_ASSET).set_color(DISK_COLORS[2]).set_width(1.2).set_height(0.3)
        disks = [d1, d2, d3]

        # Initial positioning on Peg A (stacked 3 at bottom, 1 at top)
        for i, d in enumerate(reversed(disks)):
            d.move_to([peg_a_x, base_y + 0.2 + i * 0.35, 0])

        self.add(counter_label, bits_vg, tower_group, peg_a, peg_b, peg_c, base_line, *disks)
        self.wait(1)

        # Move 1-3 to reach '011'
        moves_1_3 = [
            ("001", 0, peg_c_x, 0.2),   # D1 to C
            ("010", 1, peg_b_x, 0.2),   # D2 to B
            ("011", 0, peg_b_x, 0.55),  # D1 to B
        ]

        for b_str, d_idx, target_x, target_y_offset in moves_1_3:
            new_bits = VGroup(*[Text(char, font_size=40) for char in b_str]).arrange(RIGHT, buff=0.4)
            self.place_in_area(new_bits, "B2", "B3", scale_factor=1.0)
            self.play(
                Transform(bits_vg, new_bits),
                disks[d_idx].animate.move_to([target_x, base_y + target_y_offset, 0]),
                run_time=0.6
            )

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Binary increments from '011' to '100'; highlight the changed 3rd bit in cyan #00FFFF.
        # Step 4: 100. The 3rd bit (index 0) is highlighted.
        new_bits_4 = VGroup(*[Text(char, font_size=40) for char in "100"]).arrange(RIGHT, buff=0.4)
        new_bits_4[0].set_color(COLOR_HIGHLIGHT)
        self.place_in_area(new_bits_4, "B2", "B3", scale_factor=1.0)

        # Move Disk 3 from Peg A to Peg C (Storyboard says Peg B, but goal is Peg C for standard puzzle)
        self.play(
            Transform(bits_vg, new_bits_4),
            disks[2].animate.move_to([peg_c_x, base_y + 0.2, 0]),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Sequence through the remaining steps quickly until the counter reaches '111'.
        # Step 5: 101 -> Disk 1 to A
        # Step 6: 110 -> Disk 2 to C
        # Step 7: 111 -> Disk 1 to C
        moves_5_7 = [
            ("101", 0, peg_a_x, 0.2),   # D1 to A
            ("110", 1, peg_c_x, 0.55),  # D2 to C
            ("111", 0, peg_c_x, 0.9),   # D1 to C
        ]

        for b_str, d_idx, target_x, target_y_offset in moves_5_7:
            new_bits = VGroup(*[Text(char, font_size=40) for char in b_str]).arrange(RIGHT, buff=0.4)
            self.place_in_area(new_bits, "B2", "B3", scale_factor=1.0)
            self.play(
                Transform(bits_vg, new_bits),
                disks[d_idx].animate.move_to([target_x, base_y + target_y_offset, 0]),
                run_time=0.5
            )

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # All disks are now stacked on Peg C; the screen glows gold #FFD700.
        glow = Rectangle(
            width=self.camera.frame_width, 
            height=self.camera.frame_height, 
            fill_color=COLOR_GOLD, 
            fill_opacity=0.15, 
            stroke_width=0
        )
        self.play(FadeIn(glow))
        self.wait(2)
