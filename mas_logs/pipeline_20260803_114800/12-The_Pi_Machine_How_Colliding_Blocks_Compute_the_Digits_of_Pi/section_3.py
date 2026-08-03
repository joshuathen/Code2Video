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
            "- For equal masses, we count three total collisions.",
            "- Increase the large mass by a factor of 100.",
            "- Now we count thirty-one collisions between the blocks.",
            "- Larger mass ratios reveal the digits of Pi.",
            "- The collision count mirrors three point one four one."
        ]
        self.setup_layout("The Patterns Emerge", lecture_lines)

        # === Setup Objects ===
        # Counter
        counter_tracker = ValueTracker(0)
        counter_label = Text("Collisions:", font_size=24, color="#FF00FF")
        counter_num = DecimalNumber(0, num_decimal_places=0, color="#FF00FF")
        counter_num.add_updater(lambda d: d.set_value(counter_tracker.get_value()))
        counter_group = VGroup(counter_label, counter_num).arrange(RIGHT, buff=0.2)
        
        # Mass Ratio
        mass_label = Text("Mass Ratio:", font_size=24, color=WHITE)
        mass_val = Text("1 : 1", font_size=24, color=WHITE)
        mass_group = VGroup(mass_label, mass_val).arrange(RIGHT, buff=0.2)

        # Positioning (Resolving Issue 25 & 26)
        # Fix: Line 64 -> place_at_grid(counter_group, 'A4', scale_factor=0.8)
        self.place_at_grid(counter_group, 'A4', scale_factor=0.8)
        # Fix: Line 69 -> place_at_grid(mass_group, 'A5', scale_factor=0.8)
        self.place_at_grid(mass_group, 'A5', scale_factor=0.8)

        # Results (Resolving Issue 27)
        res_3 = Text("3", font_size=40, color="#FFFF00")
        res_31 = Text("31", font_size=40, color="#FFFF00")
        res_314 = Text("314", font_size=40, color="#FFFF00")
        pi_sym = MathTex(r"\pi", font_size=60, color="#FFFFFF")

        # Fix: Positioning result values
        self.place_at_grid(res_3, 'D3')
        self.place_at_grid(res_31, 'D4')
        self.place_at_grid(res_314, 'D5')
        self.place_at_grid(pi_sym, 'D6')

        # Environment
        # Wall at B1-C1, Floor along C row
        wall = Line(self.grid["B1"] + UP*0.5, self.grid["C1"] + DOWN*0.5, color=GREY)
        floor = Line(self.grid["C1"] + LEFT*0.5, self.grid["C6"] + RIGHT*0.5, color=GREY)
        
        block_s = Square(side_length=0.4, color=BLUE, fill_opacity=0.8)
        block_l = Square(side_length=0.7, color=RED, fill_opacity=0.8)
        
        self.place_at_grid(block_s, "C2")
        self.place_at_grid(block_l, "C5")

        # === Animation for Lecture Line 1 ===
        self.play(
            self.lecture[0].animate.set_color("#FF00FF"),
            FadeIn(VGroup(wall, floor, block_s, block_l, counter_group, mass_group))
        )
        
        # Collision 1: L hits S
        target_l_pos = self.grid["C2"] + RIGHT * 0.55
        self.play(block_l.animate.move_to(target_l_pos), run_time=0.4)
        counter_tracker.set_value(1)
        
        # Collision 2: S hits Wall
        target_s_pos = self.grid["C1"] + RIGHT * 0.2
        self.play(block_s.animate.move_to(target_s_pos), run_time=0.3)
        counter_tracker.set_value(2)
        
        # Collision 3: S hits L
        target_s_pos_2 = target_l_pos + LEFT * 0.55
        self.play(block_s.animate.move_to(target_s_pos_2), run_time=0.3)
        counter_tracker.set_value(3)
        
        # Exit
        self.play(
            block_l.animate.move_to(self.grid["C6"]), 
            block_s.animate.move_to(self.grid["C5"]), 
            run_time=0.5
        )
        self.play(Write(res_3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color("#00FFFF")
        )
        new_mass_val = Text("100 : 1", font_size=24, color="#00FFFF")
        new_mass_val.move_to(mass_val.get_center())
        
        self.play(Transform(mass_val, new_mass_val))
        
        self.play(
            block_s.animate.move_to(self.grid["C2"]),
            block_l.animate.move_to(self.grid["C5"]),
            counter_tracker.animate.set_value(0)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.play(counter_tracker.animate.set_value(31), run_time=2, rate_func=linear)
        self.play(Write(res_31))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE), 
            self.lecture[3].animate.set_color("#FFFF00")
        )
        new_mass_val_2 = Text("10,000 : 1", font_size=24, color="#FFFF00")
        new_mass_val_2.move_to(mass_val.get_center())
        
        self.play(Transform(mass_val, new_mass_val_2))
        
        self.play(counter_tracker.animate.set_value(314), run_time=2, rate_func=linear)
        self.play(Write(res_314))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE), 
            self.lecture[4].animate.set_color("#FFFFFF")
        )
        self.play(FadeIn(pi_sym))
        self.play(Indicate(res_3), Indicate(res_31), Indicate(res_314), color="#FFFF00")
        self.wait(2)
