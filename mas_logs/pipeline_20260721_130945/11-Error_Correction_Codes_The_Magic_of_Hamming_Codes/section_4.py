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
        title_text = "Graphical Logic: The Venn Diagram Method"
        lecture_lines = [
            "A Hamming (7,4) code uses three overlapping circles.",
            "Data bits occupy the intersections of these circles.",
            "Each circle's parity bit monitors its specific bit group.",
            "If a data bit flips, multiple parity checks will fail.",
            "The failure pattern reveals the exact error location.",
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        color_a = "#FF5555"
        color_b = "#55FF55"
        color_c = "#5555FF"
        color_fail = "#FF0000"
        color_success = "#00FF00"
        color_text = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Create three overlapping circles
        # Radius 1.8 scaled by 0.8 is 1.44. Distance between C4 and C6 is 2.0. 
        # Overlap = 2.88 - 2.0 = 0.88.
        circle_a = Circle(radius=1.8, color=color_a, stroke_width=4)
        circle_b = Circle(radius=1.8, color=color_b, stroke_width=4)
        circle_c = Circle(radius=1.8, color=color_c, stroke_width=4)
        
        # Apply VideoCritic grid positions (Issues 34, 35)
        self.place_at_grid(circle_a, "C4", scale_factor=0.8)
        self.place_at_grid(circle_b, "C6", scale_factor=0.8)
        self.place_at_grid(circle_c, "E5", scale_factor=0.8)
        
        self.play(
            Create(circle_a), Create(circle_b), Create(circle_c),
            self.lecture[0].animate.set_color(color_a),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place bits in regions
        def create_bit(name, value, color=WHITE):
            name_txt = Text(name, font_size=18, color=color)
            val_txt = Text(value, font_size=22, color=color)
            group = VGroup(name_txt, val_txt).arrange(DOWN, buff=0.1)
            return group, val_txt

        p1_grp, p1_val = create_bit("P1", "0")
        p2_grp, p2_val = create_bit("P2", "0")
        p4_grp, p4_val = create_bit("P4", "0")
        d3_grp, d3_val = create_bit("D3", "0")
        d5_grp, d5_val = create_bit("D5", "0")
        d6_grp, d6_val = create_bit("D6", "0")
        d7_grp, d7_val = create_bit("D7", "0")

        # Map to grid based on VideoCritic suggestions (Issues 34, 35, 36)
        # Parity bits in outer areas
        self.place_at_grid(p1_grp, "B4", scale_factor=0.7)
        self.place_at_grid(p2_grp, "B6", scale_factor=0.7)
        self.place_at_grid(p4_grp, "F5", scale_factor=0.7)
        
        # Data bits in intersections
        self.place_at_grid(d3_grp, "C5", scale_factor=0.7) # A ∩ B
        self.place_at_grid(d5_grp, "D4", scale_factor=0.7) # A ∩ C
        self.place_at_grid(d6_grp, "D6", scale_factor=0.7) # B ∩ C
        self.place_at_grid(d7_grp, "D5", scale_factor=0.7) # A ∩ B ∩ C

        all_bits = VGroup(p1_grp, p2_grp, p4_grp, d3_grp, d5_grp, d6_grp, d7_grp)

        self.play(
            FadeIn(all_bits),
            self.lecture[1].animate.set_color(color_text),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pulse Circle A and its bit group {P1, D3, D5, D7}
        circle_a_group = VGroup(p1_grp, d3_grp, d5_grp, d7_grp)
        self.play(
            Indicate(circle_a, color=WHITE),
            Indicate(circle_a_group, color=WHITE),
            self.lecture[2].animate.set_color(color_text),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Flip D3 and show failures in A and B
        flipped_d3_val = Text("1", font_size=22, color=WHITE).move_to(d3_val.get_center())
        
        self.play(
            Transform(d3_val, flipped_d3_val),
            circle_a.animate.set_color(color_fail),
            circle_b.animate.set_color(color_fail),
            self.lecture[3].animate.set_color(color_fail),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight D3 as the unique error location (Issue 36)
        location_highlight = Circle(radius=0.6, color=color_success, stroke_width=6)
        self.place_at_grid(location_highlight, "C5", scale_factor=1.0)
        
        self.play(
            Create(location_highlight),
            self.lecture[4].animate.set_color(color_success),
            run_time=1.5
        )
        self.wait(3)
