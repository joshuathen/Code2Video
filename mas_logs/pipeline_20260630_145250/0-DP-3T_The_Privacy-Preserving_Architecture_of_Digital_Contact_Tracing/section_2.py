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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: Cryptographic Hash Functions & BLE",
            [
                "Hash functions create unique, irreversible digital fingerprints.",
                "Bluetooth Low Energy enables short-range device discovery.",
                "These tools form the foundation of privacy-preserving tracing."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Hash Function box
        hash_box = RoundedRectangle(corner_radius=0.1, height=1.5, width=2.5, color="#A569BD", fill_opacity=0.2)
        hash_label = Text("Hash Function", font_size=24, color="#A569BD")
        hash_group = VGroup(hash_box, hash_label)
        # Fix: Issue 36 - Change area to B3-C5
        self.place_in_area(hash_group, "B3", "C5")
        
        # Input Text
        input_text = Text("Secret ID", font_size=20, color=WHITE)
        # Fix: Issue 35 - Place at B2
        self.place_at_grid(input_text, "B2", scale_factor=0.8)
        
        # Output Hex
        output_hex = Text("8f3c...2a", font_size=20, color="#A569BD")
        self.place_at_grid(output_hex, "C6")
        
        self.play(FadeIn(hash_group))
        self.wait(0.5)
        self.play(input_text.animate.move_to(hash_box.get_center()), run_time=1)
        self.play(FadeOut(input_text, scale=0.5), run_time=0.5)
        self.play(FadeIn(output_hex, shift=RIGHT), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Transition
        self.play(FadeOut(hash_group), FadeOut(output_hex))
        
        # Phone 1 using Asset
        # Fix: Issue 31 - Asset integration
        phone_1 = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg")
        phone_1.set_color("#3498DB")
        self.place_at_grid(phone_1, "D2", scale_factor=0.6)
        
        # Pulses
        pulse_color = "#5DADE2"
        pulses = VGroup(*[
            Circle(radius=r, stroke_color=pulse_color, stroke_opacity=1 - (r/3)) 
            for r in [0.5, 1.0, 1.5]
        ]).move_to(phone_1.get_center())
        
        self.play(FadeIn(phone_1))
        
        pulse_tracker = ValueTracker(0)
        def update_pulses(p_group):
            val = pulse_tracker.get_value()
            for i, p in enumerate(p_group):
                offset_val = (val + i * 0.33) % 1.0
                new_radius = 0.2 + offset_val * 2.5
                p.set_width(new_radius * 2)
                p.set_stroke(opacity=1 - offset_val)

        pulses.add_updater(update_pulses)
        self.add(pulses)
        self.play(pulse_tracker.animate.set_value(2), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Phone 2 using Asset
        # Fix: Issue 31 - Asset integration
        phone_2 = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg")
        phone_2.set_color("#2ECC71")
        # Fix: Issue 37 - Start at E5
        self.place_at_grid(phone_2, "E5", scale_factor=0.6)
        
        # Check mark label
        check_mark = Text("✓", color="#2ECC71", font_size=40)
        # Position check mark near phone 2 destination (D5)
        self.place_at_grid(check_mark, "D6", scale_factor=0.8) # Adjusted to be side-by-side or slightly offset
        
        # Animate phone 2 moving into BLE range
        self.play(phone_2.animate.move_to(self.grid["D5"]), run_time=2)
        self.play(Write(check_mark))
        self.play(pulse_tracker.animate.set_value(4), run_time=2, rate_func=linear)
        self.wait(2)

        # Cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(2)
