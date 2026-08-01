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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        lecture_lines = [
            "This binary pattern scales to any number of disks.",
            "Even complex towers follow the same simple bit flips.",
            "Witness the mathematical beauty in this rhythmic solution."
        ]
        self.setup_layout("Conclusion: The Power of Patterns", lecture_lines)

        # Colors for lines
        colors = [BLUE_C, GREEN_C, YELLOW_C]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Create a simple 3-disk stack
        base_3 = Line(LEFT, RIGHT, color=GREY).scale(1.5)
        disks_3 = VGroup(*[
            Rectangle(width=1.5 - i*0.3, height=0.2, fill_opacity=1, color=BLUE_B, stroke_width=1)
            for i in range(3)
        ]).arrange(UP, buff=0.05).next_to(base_3, UP, buff=0)
        stack_3 = VGroup(base_3, disks_3)
        self.place_in_area(stack_3, "B2", "D4", scale_factor=1.0)
        
        self.play(FadeIn(stack_3))
        self.wait(1)
        
        # Scale down 3-disk stack and replace with 10-disk stack
        base_10 = Line(LEFT, RIGHT, color=GREY).scale(1.5)
        disks_10 = VGroup(*[
            Rectangle(width=1.8 - i*0.15, height=0.1, fill_opacity=1, color=BLUE_E, stroke_width=0.5)
            for i in range(10)
        ]).arrange(UP, buff=0.02).next_to(base_10, UP, buff=0)
        stack_10 = VGroup(base_10, disks_10)
        
        # Reposition 10-disk stack (Issue 44, 50)
        self.place_in_area(stack_10, "B3", "E5", scale_factor=0.7)

        self.play(
            stack_3.animate.scale(0.5).move_to(stack_10.get_center()),
            run_time=1
        )
        self.play(ReplacementTransform(stack_3, stack_10))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # Binary counter scrolling rapidly
        counter_tracker = ValueTracker(0)
        
        def get_binary_text():
            val = int(counter_tracker.get_value())
            # Show binary representation with 10 bits
            bin_str = format(val, '010b')
            return Text(bin_str, font="Monospace", font_size=36, color=GREEN_B)

        binary_display = always_redraw(get_binary_text)
        # Fix binary_display position (Issue 42, 50)
        self.place_at_grid(binary_display, "D4", scale_factor=0.9)
        
        counter_label = Text("Steps (Binary):", font_size=20)
        # Position label relative to binary_display within grid
        self.place_at_grid(counter_label, "C4", scale_factor=1.0)
        
        self.play(FadeIn(counter_label), FadeIn(binary_display))
        
        # Scroll rapidly from 0 to 1023
        self.play(counter_tracker.animate.set_value(1023), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.wait(1)
        
        # Fade everything out
        self.play(
            FadeOut(stack_10),
            FadeOut(binary_display),
            FadeOut(counter_label),
            FadeOut(self.title),
            FadeOut(self.lecture)
        )
        
        # Final Title
        final_title = Text("The Binary Rhythm: Pattern in Chaos", font_size=40, color=WHITE, weight=BOLD)
        # Fix final_title size and positioning (Issue 43, 50)
        self.place_in_area(final_title, "A2", "F5", scale_factor=0.75)
        
        self.play(FadeIn(final_title))
        self.play(Indicate(final_title, color=BLUE_A))
        self.wait(3)
