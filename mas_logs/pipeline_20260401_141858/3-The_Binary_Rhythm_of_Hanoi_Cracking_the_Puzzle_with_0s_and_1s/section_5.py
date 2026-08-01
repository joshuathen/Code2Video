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
        # Setup title and lecture lines
        lecture_lines = [
            "Let's synchronize the binary counter with the physical moves.",
            "Moves one through three follow the binary rhythm perfectly.",
            "At move four, the largest disk finally shifts position.",
            "The final three moves complete the secondary stack.",
            "Seven moves later, the binary sequence solves the puzzle."
        ]
        self.setup_layout("Execution: The 3-Disk Walkthrough", lecture_lines)

        # Helper to create disks
        def create_disk(label, width, color):
            rect = RoundedRectangle(corner_radius=0.1, width=width, height=0.4, 
                                   fill_opacity=1.0, fill_color=color, stroke_color=WHITE)
            text = Text(label, font_size=20, color=BLACK)
            return VGroup(rect, text)

        # Helper to get disk position on a peg
        def get_disk_pos(peg_idx, stack_height):
            # peg_idx: 0, 1, 2 -> columns 2, 4, 6
            # stack_height: 0, 1, 2 -> rows E, D, C
            col_map = ["2", "4", "6"]
            row_map = ["E", "D", "C"]
            return self.grid[f"{row_map[stack_height]}{col_map[peg_idx]}"]

        # === INITIAL SETUP ===
        # Create Pegs
        pegs = VGroup()
        for col in ["2", "4", "6"]:
            base = Line(self.grid[f"F{col}"] + LEFT*0.6, self.grid[f"F{col}"] + RIGHT*0.6, color=GRAY)
            rod = Line(self.grid[f"F{col}"], self.grid[f"B{col}"], color=GRAY)
            pegs.add(VGroup(base, rod))
        
        peg_labels = VGroup(
            Text("A", font_size=24).next_to(pegs[0], DOWN, buff=0.2),
            Text("B", font_size=24).next_to(pegs[1], DOWN, buff=0.2),
            Text("C", font_size=24).next_to(pegs[2], DOWN, buff=0.2)
        )
        
        # Create Disks
        d3 = create_disk("D3", 1.4, BLUE_D)
        d2 = create_disk("D2", 1.0, GREEN_D)
        d1 = create_disk("D1", 0.6, RED_D)
        disks = [d1, d2, d3] # smallest to largest
        
        # Initial positions on Peg A (idx 0)
        d3.move_to(get_disk_pos(0, 0))
        d2.move_to(get_disk_pos(0, 1))
        d1.move_to(get_disk_pos(0, 2))
        
        # Binary Counter
        counter_label = Text("Binary Step:", font_size=20, color=YELLOW)
        self.place_at_grid(counter_label, "A2", scale_factor=0.8) # Issue 39: Move to A2
        
        counter_val = Text("000", font_size=36, font="Courier", color=WHITE)
        self.place_at_grid(counter_val, "A3", scale_factor=1.0) # Issue 39: Move to A3
        
        stacks = [[d3, d2, d1], [], []]

        # === Animation for Lecture Line 1 ===
        # "Let's synchronize the binary counter with the physical moves."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(pegs), Write(peg_labels), FadeIn(d1, d2, d3), FadeIn(counter_label, counter_val))
        self.wait(1)

        # Helper for updating counter
        def update_counter(val_str):
            new_val = Text(val_str, font_size=36, font="Courier", color=WHITE)
            new_val.move_to(counter_val.get_center())
            return Transform(counter_val, new_val)

        # Helper for moving disk
        def move_disk_anim(disk_idx, from_peg, to_peg):
            # disk_idx: 0 for d1, 1 for d2, 2 for d3
            disk = disks[disk_idx]
            stacks[from_peg].remove(disk)
            target_pos = get_disk_pos(to_peg, len(stacks[to_peg]))
            stacks[to_peg].append(disk)
            return disk.animate.move_to(target_pos)

        # === Animation for Lecture Line 2 ===
        # "Moves one through three follow the binary rhythm perfectly."
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Step 1: 001 (D1 to C)
        self.play(update_counter("001"), move_disk_anim(0, 0, 2), run_time=0.8)
        self.wait(0.2)
        # Step 2: 010 (D2 to B)
        self.play(update_counter("010"), move_disk_anim(1, 0, 1), run_time=0.8)
        self.wait(0.2)
        # Step 3: 011 (D1 to B)
        self.play(update_counter("011"), move_disk_anim(0, 2, 1), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "At move four, the largest disk finally shifts position."
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        # Step 4: 100 (D3 to C)
        self.play(update_counter("100"), move_disk_anim(2, 0, 2), run_time=1.0)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "The final three moves complete the secondary stack."
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        # Step 5: 101 (D1 to A)
        self.play(update_counter("101"), move_disk_anim(0, 1, 0), run_time=0.8)
        self.wait(0.2)
        # Step 6: 110 (D2 to C)
        self.play(update_counter("110"), move_disk_anim(1, 1, 2), run_time=0.8)
        self.wait(0.2)
        # Step 7: 111 (D1 to C)
        self.play(update_counter("111"), move_disk_anim(0, 0, 2), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # "Seven moves later, the binary sequence solves the puzzle."
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # Issue 40/41: Use place_in_area at A4-A5 with scale 0.8
        flash_text = Text("2^3 - 1 = 7 Moves", font_size=32, color="#00FF00")
        self.place_in_area(flash_text, "A4", "A5", scale_factor=0.8)
        
        self.play(Write(flash_text))
        self.play(Indicate(flash_text, color="#00FF00", scale_factor=1.1))
        self.wait(2)
        
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
