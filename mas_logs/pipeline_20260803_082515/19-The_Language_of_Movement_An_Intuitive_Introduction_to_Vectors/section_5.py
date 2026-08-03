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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Vector Addition: The Tip-to-Tail Rule",
            [
                "Combining movements is called vector addition.",
                "Place the second vector's tail at the first tip.",
                "The result connects the start to the end.",
                "This new arrow is the resultant vector.",
                "It shows the total net movement."
            ]
        )

        # Define Colors
        COLOR_A = "#00BFFF"  # Deep Sky Blue
        COLOR_B = "#FFFF00"  # Yellow
        COLOR_RESULTANT = "#FF00FF"  # Magenta

        # === Animation for Lecture Line 1 ===
        # "Combining movements is called vector addition."
        self.play(self.lecture[0].animate.set_color(COLOR_A))
        
        # Draw Vector A pointing up from D2 to B2
        vec_a = Arrow(
            start=self.grid["D2"],
            end=self.grid["B2"],
            buff=0,
            color=COLOR_A,
            stroke_width=6
        )
        label_a = MathTex(r"\vec{A}", color=COLOR_A, font_size=32)
        self.place_at_grid(label_a, "C2") # Position label near Vector A (Issue 39 fix)

        self.play(Create(vec_a), Write(label_a))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Place the second vector's tail at the first tip."
        self.play(self.lecture[1].animate.set_color(COLOR_B))
        
        # Draw Vector B initially pointing right from D2 to D4
        vec_b = Arrow(
            start=self.grid["D2"],
            end=self.grid["D4"],
            buff=0,
            color=COLOR_B,
            stroke_width=6
        )
        label_b = MathTex(r"\vec{B}", color=COLOR_B, font_size=32)
        # Position label near Vector B's initial location (Issue 40 fix requested A4, but initial was E3)
        # Note: If we use A4 initially, the animation logic below needs to move it to its final position correctly.
        self.place_at_grid(label_b, "A4") 

        # Load Asset for Issue 25
        movement_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/movement.svg")
        movement_icon.set_color(WHITE).scale(0.3)
        # Place icon near the tail of Vector B initially
        movement_icon.move_to(self.grid["D2"] + DOWN * 0.3)

        self.play(Create(vec_b), Write(label_b), FadeIn(movement_icon))
        self.wait(0.5)

        # Shift Vector B so its tail is at Vector A's tip (B2)
        # We'll use a ValueTracker to animate the shift smoothly
        shift_tracker = ValueTracker(0)
        
        # Initial positions (adjusted to match current placement)
        b_start_init = self.grid["D2"]
        b_end_init = self.grid["D4"]
        label_b_init = self.grid["A4"] 
        icon_init = movement_icon.get_center()
        
        # Final positions
        b_start_final = self.grid["B2"]
        b_end_final = self.grid["B4"]
        label_b_final = self.grid["A3"] # Keep final label B near the new position
        icon_final = self.grid["B2"] + DOWN * 0.3

        def update_vec_b(mob):
            alpha = shift_tracker.get_value()
            new_start = (1 - alpha) * b_start_init + alpha * b_start_final
            new_end = (1 - alpha) * b_end_init + alpha * b_end_final
            mob.put_start_and_end_on(new_start, new_end)

        def update_label_b(mob):
            alpha = shift_tracker.get_value()
            new_pos = (1 - alpha) * label_b_init + alpha * label_b_final
            mob.move_to(new_pos)
            
        def update_icon(mob):
            alpha = shift_tracker.get_value()
            new_pos = (1 - alpha) * icon_init + alpha * icon_final
            mob.move_to(new_pos)

        vec_b.add_updater(update_vec_b)
        label_b.add_updater(update_label_b)
        movement_icon.add_updater(update_icon)
        
        self.play(shift_tracker.animate.set_value(1), run_time=2)
        
        vec_b.remove_updater(update_vec_b)
        label_b.remove_updater(update_label_b)
        movement_icon.remove_updater(update_icon)
        
        self.play(FadeOut(movement_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The result connects the start to the end."
        self.play(self.lecture[2].animate.set_color(COLOR_RESULTANT))
        
        # Resultant vector from start of A (D2) to end of B (B4)
        vec_res = Arrow(
            start=self.grid["D2"],
            end=self.grid["B4"],
            buff=0,
            color=COLOR_RESULTANT,
            stroke_width=8
        )
        self.play(Create(vec_res))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This new arrow is the resultant vector."
        self.play(self.lecture[3].animate.set_color(COLOR_RESULTANT))
        
        label_res = Text("Resultant", color=COLOR_RESULTANT, font_size=24)
        self.place_at_grid(label_res, "C6", scale_factor=0.8) # Issue 41 fix: moved from C5 to C6
        
        self.play(Write(label_res))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "It shows the total net movement."
        self.play(self.lecture[4].animate.set_color(COLOR_RESULTANT))
        
        # Flash the resultant vector to highlight it
        self.play(Flash(vec_res, color=COLOR_RESULTANT, flash_radius=0.5))
        self.play(Indicate(vec_res, color=COLOR_RESULTANT))
        self.wait(2)
