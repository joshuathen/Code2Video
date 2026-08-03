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
        # Initializing the layout
        self.setup_layout(
            "The Familiar Reality: Euclidean Distance",
            [
                "On a number line, distance usually depends on size.",
                "Larger jumps like 1, 2, 4 move toward infinity.",
                "Sums only converge if added terms get smaller."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create NumberLine
        # Fix for Issue 31: Use place_in_area and adjust range to fit.
        # Map 0 to C1 and 4 to C5. Length is 4.
        number_line = NumberLine(
            x_range=[0, 4, 1],
            length=4,
            include_numbers=False,
            color=WHITE
        )
        # Center of C1-C5 is at x=2.5.
        self.place_in_area(number_line, "C1", "C5")
        
        # Manual Labels for 0, 1, 2, 4 at Row D
        label0 = Text("0", font_size=20)
        self.place_at_grid(label0, "D1")
        label1 = Text("1", font_size=20)
        self.place_at_grid(label1, "D2")
        label2 = Text("2", font_size=20)
        self.place_at_grid(label2, "D3")
        label4 = Text("4", font_size=20)
        self.place_at_grid(label4, "D5")
        
        labels = VGroup(label0, label1, label2, label4)
        
        self.play(Create(number_line))
        self.play(FadeIn(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Rabbit represented by a Red Dot
        rabbit = Dot(color="#FF0000")
        rabbit_pos = ValueTracker(0)
        # Using updater for smooth movement relative to number line
        rabbit.add_updater(lambda m: m.move_to(number_line.n2p(rabbit_pos.get_value())))
        
        rabbit_label = Text("Rabbit", font_size=16, color="#FF0000")
        rabbit_label.add_updater(lambda m: m.next_to(rabbit, UP, buff=0.1))
        
        self.add(rabbit, rabbit_label)
        
        # Jumps: 0 -> 1, then to 3 (1+2), then to 7 (3+4)
        # Even though line visually ends at 4, n2p works for 7.
        for target in [1, 3, 7]:
            self.play(rabbit_pos.animate.set_value(target), run_time=1)
            self.wait(0.2)
        
        # Vanish off-screen right
        self.play(rabbit_pos.animate.set_value(12), run_time=1.5)
        self.remove(rabbit, rabbit_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Reset rabbit and show convergence
        rabbit_pos.set_value(0)
        self.add(rabbit, rabbit_label)
        
        # Vertical dashed limit line at 2.0 (Grid Col 3)
        limit_line = DashedLine(
            start=self.grid["B3"],
            end=self.grid["D3"],
            color="#00FF00"
        )
        limit_text = Text("Limit: 2.0", font_size=16, color="#00FF00")
        # Fix for Issue 32: Move limit_text to B3
        self.place_at_grid(limit_text, "B3", scale_factor=0.9)
        
        self.play(Create(limit_line), FadeIn(limit_text))
        
        # Rabbit jumps 1, then 0.5, then 0.25
        # Positions: 1, 1.5, 1.75
        for target in [1, 1.5, 1.75]:
            self.play(rabbit_pos.animate.set_value(target), run_time=1)
            self.wait(0.2)
            
        self.wait(2)
