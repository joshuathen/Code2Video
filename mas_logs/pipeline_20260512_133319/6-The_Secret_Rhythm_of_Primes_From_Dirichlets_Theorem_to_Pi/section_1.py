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
        self.setup_layout(
            "The Prerequisites: Arithmetic Progressions and Primes", 
            [
                "Arithmetic progressions follow a simple pattern: a, a+d, a+2d.", 
                "Focus on the sequences 4n plus 1 and 4n plus 3.", 
                "Co-primality requires the starting number and difference share no factors.", 
                "Without co-primality, a sequence can only hold one prime.", 
                "Valid prime lanes show a pattern of infinite potential."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # [a, a+d, a+2d...] in #00FF00 drifting slowly
        pattern_text = Text("a, a + d, a + 2d, ...", font_size=30, color="#00FF00")
        self.place_at_grid(pattern_text, "A3")
        
        self.play(FadeIn(pattern_text, shift=RIGHT))
        self.play(pattern_text.animate.shift(RIGHT * 0.5), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Labels for sequences
        label1 = Text("4n + 1", font_size=22, color="#00FFFF")
        self.place_at_grid(label1, "B1", scale_factor=0.8)
        
        label2 = Text("4n + 3", font_size=22, color="#FF00FF")
        self.place_at_grid(label2, "C1", scale_factor=0.8)
        
        # 4n+1 (1, 5, 9, 13, 17)
        train1_boxes = VGroup(*[
            VGroup(
                Square(side_length=0.7, color="#00FFFF"),
                Text(str(1 + 4*i), font_size=20, color=WHITE)
            ) for i in range(5)
        ]).arrange(RIGHT, buff=0.1)
        # Issue 34 fix: Use B2 to B6
        self.place_in_area(train1_boxes, "B2", "B6")
        
        # 4n+3 (3, 7, 11, 15, 19)
        train2_boxes = VGroup(*[
            VGroup(
                Square(side_length=0.7, color="#FF00FF"),
                Text(str(3 + 4*i), font_size=20, color=WHITE)
            ) for i in range(5)
        ]).arrange(RIGHT, buff=0.1)
        # Issue 34 fix: Use C2 to C6
        self.place_in_area(train2_boxes, "C2", "C6")
        
        self.play(FadeIn(label1), FadeIn(train1_boxes, shift=RIGHT))
        self.play(FadeIn(label2), FadeIn(train2_boxes, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Highlight a=1 and d=4 with circles
        # Issue 36 fix: Move a_label to A2
        a_label = Text("a = 1", font_size=18, color="#FFFF00")
        self.place_at_grid(a_label, "A2", scale_factor=0.8)
        
        d_label = Text("d = 4", font_size=18, color="#FFFF00")
        self.place_at_grid(d_label, "A5", scale_factor=0.8)
        
        circle_a = Circle(radius=0.4, color="#FFFF00").move_to(train1_boxes[0])
        circle_d = Circle(radius=0.4, color="#FFFF00").move_to(d_label)
        
        gcd_formula = Text("gcd(1, 4) = 1", font_size=32, color="#FFFF00")
        self.place_at_grid(gcd_formula, "D3")
        
        self.play(
            Create(circle_a), 
            Create(circle_d), 
            Write(a_label), 
            Write(d_label)
        )
        self.play(Write(gcd_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF5555"))
        
        # Non-coprime sequence 2n+2 (2, 4, 6, 8, 10...)
        fail_label = Text("2n + 2", font_size=20, color="#FF5555")
        self.place_at_grid(fail_label, "E1", scale_factor=0.8)
        
        bad_seq = VGroup(*[
            VGroup(
                Square(side_length=0.6, color="#FF5555"),
                Text(str(2*(i+1)), font_size=18, color=WHITE)
            ) for i in range(5)
        ]).arrange(RIGHT, buff=0.1)
        # Issue 35 fix: Use E2 to E5
        self.place_in_area(bad_seq, "E2", "E5")
        
        fail_gcd = Text("gcd(2, 2) = 2", font_size=24, color="#FF5555")
        self.place_at_grid(fail_gcd, "E6")

        self.play(Write(fail_label), FadeIn(bad_seq))
        self.play(Write(fail_gcd))
        self.wait(0.5)
        
        # 2 glows gold while all other numbers turn #444444 and fade
        self.play(
            Indicate(bad_seq[0], color="#FFD700"), # Gold
            bad_seq[1:].animate.set_opacity(0.3).set_color("#444444"),
            fail_gcd.animate.set_color("#444444")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        # Clean up highlights and distractions
        self.play(
            FadeOut(bad_seq), FadeOut(fail_label), FadeOut(fail_gcd),
            FadeOut(circle_a), FadeOut(circle_d), FadeOut(a_label), FadeOut(d_label),
            FadeOut(gcd_formula)
        )
        
        # Final glow and extension
        self.play(
            train1_boxes.animate.set_color("#00FFFF").scale(1.2),
            train2_boxes.animate.set_color("#FF00FF").scale(1.2),
            label1.animate.set_color("#00FFFF"),
            label2.animate.set_color("#FF00FF")
        )
        
        lane_text = Text("VALID PRIME LANES", font_size=36, color="#00FF00")
        self.place_in_area(lane_text, "E1", "F6")
        self.play(Write(lane_text))
        
        self.wait(2)
