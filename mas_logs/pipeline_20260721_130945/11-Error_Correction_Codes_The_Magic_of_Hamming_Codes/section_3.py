from manim import *

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
        # Define lecture lines from storyboard
        lecture_lines = [
            "Richard Hamming proposed using multiple parity bits for subsets.",
            "These subsets overlap, creating a unique signature for each bit.",
            "Parity bits occupy positions that are powers of two."
        ]
        
        self.setup_layout("The Hamming Strategy: Overlapping Checks", lecture_lines)
        
        # Colors
        WHITE_COLOR = "#FFFFFF"
        YELLOW_COLOR = "#FFFF00"
        CYAN_COLOR = "#00FFFF"
        GRAY_COLOR = "#888888"
        
        # Initial state: Dim all lecture lines except the first one
        for line in self.lecture[1:]:
            line.set_color(GRAY_COLOR)
        self.lecture[0].set_color(WHITE_COLOR)
        
        # Initialize containers for slots and labels
        slots = VGroup()
        pos_labels = VGroup()
        bit_labels = VGroup()
        
        # Helper to create a slot at position i
        def create_slot(i):
            square = Square(side_length=0.7, color=WHITE_COLOR)
            num_txt = Text(str(i), font_size=14, color=GRAY_COLOR).next_to(square, UP, buff=0.1)
            
            if i in [1, 2, 4]:
                if i == 1: content_txt = Text("P1", font_size=20, color=WHITE_COLOR)
                elif i == 2: content_txt = Text("P2", font_size=20, color=WHITE_COLOR)
                elif i == 4: content_txt = Text("P4", font_size=20, color=WHITE_COLOR)
            else:
                d_idx = {3: 1, 5: 2, 6: 3, 7: 4}[i]
                content_txt = Text(f"D{d_idx}", font_size=20, color=WHITE_COLOR)
            
            content_txt.move_to(square.get_center())
            return VGroup(square, num_txt, content_txt)

        # === Animation for Lecture Line 1 ===
        # Display seven slots labeled 1 through 7, with P1, P2, and P4 at positions 1, 2, and 4.
        # Line 1 is already colored WHITE in setup.
        
        all_slots_group = VGroup(*[create_slot(i) for i in range(1, 8)]).arrange(RIGHT, buff=0.2)
        # Accessing internal mobjects for later animations
        for slot_unit in all_slots_group:
            slots.add(slot_unit[0])
            pos_labels.add(slot_unit[1])
            bit_labels.add(slot_unit[2])
            
        # Fix for Issue 33: Adjust positioning to avoid overlap with lecture text (B3-E6, scale 0.7)
        self.place_in_area(all_slots_group, "B3", "E6", scale_factor=0.7)
        
        self.play(
            AnimationGroup(
                *[FadeIn(unit) for unit in all_slots_group],
                lag_ratio=0.1
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight slots 1, 3, 5, 7 in yellow (#FFFF00) to show the coverage of Parity Bit 1.
        self.play(
            self.lecture[0].animate.set_color(GRAY_COLOR),
            self.lecture[1].animate.set_color(YELLOW_COLOR)
        )
        
        p1_indices = [0, 2, 4, 6] # Positions 1, 3, 5, 7 (0-indexed)
        p1_rects = VGroup(*[SurroundingRectangle(slots[i], color=YELLOW_COLOR, buff=0.05) for i in p1_indices])
        
        self.play(
            *[Indicate(slots[i], color=YELLOW_COLOR) for i in p1_indices],
            Create(p1_rects),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight slots 2, 3, 6, 7 in cyan (#00FFFF) to show the overlapping coverage of Parity Bit 2.
        self.play(
            self.lecture[1].animate.set_color(GRAY_COLOR),
            self.lecture[2].animate.set_color(CYAN_COLOR)
        )
        
        p2_indices = [1, 2, 5, 6] # Positions 2, 3, 6, 7 (0-indexed)
        # Note: buff=0.12 prevents visual overlap with P1's rectangles (buff=0.05)
        p2_rects = VGroup(*[SurroundingRectangle(slots[i], color=CYAN_COLOR, buff=0.12) for i in p2_indices])
        
        self.play(
            *[Indicate(slots[i], color=CYAN_COLOR) for i in p2_indices],
            Create(p2_rects),
            run_time=1.5
        )
        self.wait(3)
