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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        title = "Prerequisite: Building Arithmetic Progressions"
        lines = [
            "We sort numbers into lanes based on their remainders.",
            "Compare the 4n plus 1 and 4n plus 3 tracks.",
            "Primes appear in both lanes following distinct arithmetic paths."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_BLUE = "#ADD8E6"
        COLOR_ORANGE = "#FFCC99"
        COLOR_GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_BLUE), run_time=0.5)

        # Display an + b formula (Issue 25 fix)
        formula = Text("an + b", font_size=90)
        self.place_in_area(formula, "A2", "A5", scale_factor=0.6)
        self.play(Write(formula))

        # Initial numbers 1-16
        nums = VGroup(*[Text(str(i), font_size=36) for i in range(1, 17)])
        # Position them in a temporary central grid (area C2:F5)
        for i, num in enumerate(nums):
            r_idx = i // 4
            c_idx = i % 4
            r = ["C", "D", "E", "F"][r_idx]
            c = ["2", "3", "4", "5"][c_idx]
            self.place_at_grid(num, f"{r}{c}")
        
        self.play(FadeIn(nums))
        self.wait(0.5)

        # Sort numbers into lanes (Indices 1..16 are 0..15)
        lane_a_indices = [0, 4, 8, 12]  # 1, 5, 9, 13
        lane_b_indices = [2, 6, 10, 14] # 3, 7, 11, 15
        other_indices = [i for i in range(16) if i not in (lane_a_indices + lane_b_indices)]

        move_animations = []
        for i, idx in enumerate(lane_a_indices):
            target_pos = self.grid[f"{['C', 'D', 'E', 'F'][i]}2"]
            move_animations.append(nums[idx].animate.move_to(target_pos))
        
        for i, idx in enumerate(lane_b_indices):
            # Column 4 for Lane B (to match Issue 26 label position)
            target_pos = self.grid[f"{['C', 'D', 'E', 'F'][i]}4"]
            move_animations.append(nums[idx].animate.move_to(target_pos))

        self.play(
            *move_animations,
            *[FadeOut(nums[idx]) for idx in other_indices],
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_ORANGE),
            run_time=0.5
        )

        # Define and place Labels (Issue 26 fix)
        lane_a_label = Text("4n + 1", font_size=28, color=COLOR_BLUE)
        lane_b_label = Text("4n + 3", font_size=28, color=COLOR_ORANGE)
        self.place_at_grid(lane_a_label, "B2", scale_factor=0.8)
        self.place_at_grid(lane_b_label, "B4", scale_factor=0.8)

        # Color the numbers in lanes
        color_animations = []
        for idx in lane_a_indices:
            color_animations.append(nums[idx].animate.set_color(COLOR_BLUE))
        for idx in lane_b_indices:
            color_animations.append(nums[idx].animate.set_color(COLOR_ORANGE))

        self.play(
            FadeIn(lane_a_label),
            FadeIn(lane_b_label),
            *color_animations,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_GOLD),
            run_time=0.5
        )

        # Primes highlight: 5, 13 (Lane A) and 3, 7, 11 (Lane B)
        # indices in nums: 4, 12 and 2, 6, 10
        prime_indices = [4, 12, 2, 6, 10]
        boxes = VGroup()
        for idx in prime_indices:
            box = SurroundingRectangle(nums[idx], color=COLOR_GOLD, buff=0.1)
            boxes.add(box)

        self.play(Create(boxes))
        self.wait(2)
