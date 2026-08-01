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

class Section4Scene(TeachingScene):
    def construct(self):
        # === Setup Layout ===
        title_text = "The Power of 2: Positioning Bits"
        lecture_lines = [
            "Parity bits occupy positions at powers of two.",
            "All other positions hold the actual message data.",
            "This specific layout enables the mathematical correction magic."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define indices for parity and data
        parity_indices = [0, 1, 3]  # Positions 1, 2, 4 (0-indexed)
        data_indices = [2, 4, 5, 6]    # Positions 3, 5, 6, 7 (0-indexed)

        # === Animation for Lecture Line 1 ===
        # Description: Display numbers 1-7; highlight 1, 2, and 4 in blue (#5555FF). Change Line 1 color to #FFFF00. self.wait(2).
        
        # Highlight lecture line first
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Create slots and numbers
        slots = VGroup(*[
            Circle(radius=0.4, color="#FFFFFF").set_stroke(width=2) 
            for _ in range(7)
        ]).arrange(RIGHT, buff=0.3)
        
        nums = VGroup(*[
            Text(str(i), font_size=24, color="#FFFFFF") 
            for i in range(1, 8)
        ])
        for n, s in zip(nums, slots):
            n.move_to(s.get_center())
            
        bit_positions = VGroup(slots, nums)
        # Using Area C1-C6 as suggested by VideoCritic (Issues 38 & 39)
        self.place_in_area(bit_positions, "C1", "C6", scale_factor=0.8)
        
        self.play(FadeIn(bit_positions))
        self.wait(0.5)

        # Highlight powers of 2 (positions 1, 2, 4)
        self.play(
            *[slots[i].animate.set_color("#5555FF") for i in parity_indices],
            *[nums[i].animate.set_color("#5555FF") for i in parity_indices],
            *[Indicate(nums[i], color="#5555FF") for i in parity_indices]
        )
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Description: Fill positions 3, 5, 6, 7 with white 'D' characters (#FFFFFF). Change Line 2 color to #FFFF00. self.wait(2).
        
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        d_labels = VGroup()
        for i in data_indices:
            d = Text("D", font_size=24, color="#FFFFFF")
            d.move_to(slots[i].get_center())
            d_labels.add(d)

        # Morph numbers at positions 3, 5, 6, 7 to 'D' labels
        self.play(
            *[Transform(nums[i], d_labels[idx]) for idx, i in enumerate(data_indices)]
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Description: Arrange the labels into the sequence [P1, P2, D3, P4, D5, D6, D7]. Change Line 3 color to #FFFF00. self.wait(2).
        
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Define final labels for the sequence
        # Index map: 0:P1, 1:P2, 2:D3, 3:P4, 4:D5, 5:D6, 6:D7
        final_label_texts = ["P1", "P2", "D3", "P4", "D5", "D6", "D7"]
        final_colors = ["#5555FF", "#5555FF", "#FFFFFF", "#5555FF", "#FFFFFF", "#FFFFFF", "#FFFFFF"]
        
        final_labels = VGroup()
        for i in range(7):
            lbl = Text(final_label_texts[i], font_size=22, color=final_colors[i])
            lbl.move_to(slots[i].get_center())
            final_labels.add(lbl)

        # Transform current elements into the full [P, D] sequence
        transforms = []
        # Transform parity numbers (already blue) to P1, P2, P4
        for i in parity_indices:
            transforms.append(Transform(nums[i], final_labels[i]))
        
        # Transform current 'D' characters to D3, D5, D6, D7
        for i in data_indices:
            transforms.append(Transform(nums[i], final_labels[i]))

        self.play(*transforms)
        self.wait(2.0)
