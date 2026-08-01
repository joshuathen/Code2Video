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
        lecture_lines = [
            "Every disk configuration has a unique ternary address.",
            "Largest disks are the most significant digits.",
            "Smallest disks represent the least significant digits.",
            "State 'zero-two' means big on A, small on C.",
            "Three disks create twenty-seven unique state addresses."
        ]
        self.setup_layout("Mapping States to Ternary Coordinates", lecture_lines)

        # Colors for highlighting
        COLOR_1 = YELLOW
        COLOR_2 = BLUE_B
        COLOR_3 = GREEN_B
        COLOR_4 = ORANGE
        COLOR_5 = PINK

        # Common Mobjects: Pegs
        peg_width = 0.1
        peg_height = 2.5
        pegs = VGroup(
            Rectangle(width=peg_width, height=peg_height, fill_opacity=1, color=GRAY),
            Rectangle(width=peg_width, height=peg_height, fill_opacity=1, color=GRAY),
            Rectangle(width=peg_width, height=peg_height, fill_opacity=1, color=GRAY)
        ).arrange(RIGHT, buff=1.2)
        
        peg_labels = VGroup(
            Text("A (0)", font_size=18),
            Text("B (1)", font_size=18),
            Text("C (2)", font_size=18)
        )
        for i, label in enumerate(peg_labels):
            label.next_to(pegs[i], DOWN, buff=0.2)

        pegs_and_labels = VGroup(pegs, peg_labels)
        self.place_in_area(pegs_and_labels, "C1", "F6", scale_factor=0.8)

        # Disks
        disk_large = Rectangle(width=1.4, height=0.3, fill_opacity=1, color=COLOR_2)
        disk_small = Rectangle(width=0.8, height=0.3, fill_opacity=1, color=COLOR_3)

        # Function to position disks on pegs
        def get_disk_pos(disk, peg_index, height_index):
            base_pos = pegs[peg_index].get_bottom()
            return base_pos + UP * (0.3 * height_index + 0.15)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        self.play(Create(pegs), Write(peg_labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        disk_large.move_to(get_disk_pos(disk_large, 0, 0))
        addr_text_l2 = Text("Digit 1 (MSB)", font_size=24, color=COLOR_2)
        self.place_at_grid(addr_text_l2, "A2")
        self.play(FadeIn(disk_large), FadeIn(addr_text_l2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        disk_small.move_to(get_disk_pos(disk_small, 0, 1))
        addr_text_l3 = Text("Digit 2 (LSB)", font_size=24, color=COLOR_3)
        self.place_at_grid(addr_text_l3, "A5")
        self.play(FadeIn(disk_small), FadeIn(addr_text_l3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_4))
        
        # State '02': Large on A (0), Small on C (2)
        new_disk_large_pos = get_disk_pos(disk_large, 0, 0)
        new_disk_small_pos = get_disk_pos(disk_small, 2, 0)
        
        # Use t2c to safely color "0" and "2" without indexing issues
        address_display = Text("Address: 0 2", font_size=36, t2c={"0": COLOR_2, "2": COLOR_3})
        self.place_in_area(address_display, "B2", "B5")

        self.play(
            disk_large.animate.move_to(new_disk_large_pos),
            disk_small.animate.move_to(new_disk_small_pos),
            ReplacementTransform(VGroup(addr_text_l2, addr_text_l3), address_display)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_5))
        
        # Clear specific config to show grid
        self.play(FadeOut(pegs_and_labels), FadeOut(disk_large), FadeOut(disk_small), FadeOut(address_display))
        
        # Create a 3x3 grid of states
        states_3x3 = VGroup()
        for i in range(3):
            row_group = VGroup()
            for j in range(3):
                state_label = Text(f"{i}{j}", font_size=24, color=COLOR_5)
                row_group.add(state_label)
            row_group.arrange(RIGHT, buff=0.8)
            states_3x3.add(row_group)
        states_3x3.arrange(DOWN, buff=0.6)
        
        grid_title = Text("All 9 States (N=2)", font_size=24, color=WHITE)
        total_3x3 = VGroup(grid_title, states_3x3).arrange(DOWN, buff=0.4)
        self.place_in_area(total_3x3, "B2", "E5")
        
        n3_info = Text("For N=3: 3^3 = 27 states", font_size=20, color=COLOR_5)
        self.place_in_area(n3_info, 'F2', 'F5')
        
        self.play(FadeIn(total_3x3))
        self.play(Write(n3_info))
        self.wait(2)
