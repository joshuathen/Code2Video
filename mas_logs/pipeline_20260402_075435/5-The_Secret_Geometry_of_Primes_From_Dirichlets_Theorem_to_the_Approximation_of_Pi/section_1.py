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
        # Setup the layout with provided text
        title_text = "The Prime Atoms and the Number Line"
        lines = [
            "Imagine the number line as a digital highway.",
            "Our Prime Scout travels this road searching for atoms.",
            "Primes like 2, 3, and 5 ignite like gold.",
            "Non-primes fade away, leaving only the fundamental building blocks.",
            "Zooming out reveals a vast field of prime sparks."
        ]
        self.setup_layout(title_text, lines)
        
        # Colors
        GOLD = "#FFD700"
        GRAY = "#808080"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GOLD)
        
        # Create a white horizontal line with tick marks from 1 to 10.
        # Fixed positioning: Move higher (B2-B6) and stay right to avoid occlusion.
        number_line = NumberLine(
            x_range=[1, 10, 1],
            length=4.5,
            include_numbers=True,
            font_size=20,
            color=WHITE_COLOR,
            stroke_width=2,
            label_constructor=Text
        )
        self.place_in_area(number_line, "B2", "B6")
        
        self.play(Create(number_line), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GOLD)

        # A golden dot (#FFD700) representing the 'Prime Scout' appears at 1 and begins moving right.
        scout = Dot(color=GOLD, radius=0.1)
        scout.move_to(number_line.n2p(1))
        self.add(scout)
        
        numbers_dict = {}
        for decimal in number_line.numbers:
            val = int(round(decimal.get_value()))
            numbers_dict[val] = decimal

        primes = [2, 3, 5, 7]
        composites = [4, 6, 8, 9]
        
        tracker = ValueTracker(1)
        scout.add_updater(lambda m: m.move_to(number_line.n2p(tracker.get_value())))

        # Start movement
        self.play(tracker.animate.set_value(2), run_time=0.6, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GOLD)
        
        # Flash primes as we encounter them
        for target_val in [2, 3, 4, 5]:
            if target_val > 2:
                self.play(tracker.animate.set_value(target_val), run_time=0.4, rate_func=linear)
            
            if target_val in primes:
                self.play(
                    Flash(number_line.n2p(target_val), color=GOLD, flash_radius=0.4),
                    numbers_dict[target_val].animate.set_color(GOLD).scale(1.3),
                    run_time=0.4
                )
            elif target_val in composites:
                # Early transition to Line 4 logic for composite 4
                self.lecture[2].set_color(WHITE)
                self.lecture[3].set_color(GOLD)
                self.play(
                    numbers_dict[target_val].animate.set_color(GRAY),
                    run_time=0.3
                )

        # === Animation for Lecture Line 4 ===
        # Continue the traversal for the remaining numbers
        for target_val in range(6, 11):
            self.play(tracker.animate.set_value(target_val), run_time=0.4, rate_func=linear)
            
            if target_val in primes:
                self.play(
                    Flash(number_line.n2p(target_val), color=GOLD, flash_radius=0.4),
                    numbers_dict[target_val].animate.set_color(GOLD).scale(1.3),
                    run_time=0.4
                )
            elif target_val in composites:
                self.play(
                    numbers_dict[target_val].animate.set_color(GRAY),
                    run_time=0.3
                )
            else:
                self.wait(0.2)
        
        scout.remove_updater(scout.updaters[0])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD)

        # Reveal a dense trail by zooming out.
        extended_line = NumberLine(
            x_range=[1, 100, 1],
            length=25,
            include_numbers=False,
            color=WHITE_COLOR,
            stroke_width=1
        )
        
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True

        sparks = VGroup()
        for i in range(1, 101):
            if is_prime(i):
                spark = Dot(color=GOLD, radius=0.03).move_to(extended_line.n2p(i))
                sparks.add(spark)
        
        extended_line.move_to(number_line.n2p(1), aligned_edge=LEFT)
        reveal_group = VGroup(extended_line, sparks)
        
        # Remove old elements
        self.remove(number_line, scout)
        for val in numbers_dict:
            self.remove(numbers_dict[val])
            
        self.add(reveal_group)
        
        # Position reveal_group in the center of the right grid area (B2 to E6)
        # Scale significantly to avoid bleeding into lecture notes.
        self.play(
            reveal_group.animate.scale(0.15).move_to(self.grid["C4"]),
            run_time=3,
            rate_func=slow_into
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
