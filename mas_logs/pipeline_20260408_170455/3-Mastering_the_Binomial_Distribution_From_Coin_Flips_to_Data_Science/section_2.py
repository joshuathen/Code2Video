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
        # Setup the layout
        self.setup_layout(
            "Defining the Binomial Distribution (The BINS Test)", 
            [
                'For a Binomial Distribution, we use the BINS test.', 
                'Outcomes must be Binary: a simple success or failure.', 
                'Each trial must be Independent from the previous ones.', 
                'We need a fixed Number of trials, like ten.', 
                'The probability of success must remain the Same throughout.'
            ]
        )
        
        # Define colors for the acronym letters
        COLOR_B = "#58C4DD"
        COLOR_I = "#83C167"
        COLOR_N = "#FFFF00"
        COLOR_S = "#FF8080"
        
        # === Animation for Lecture Line 1 ===
        # Display the acronym 'BINS' vertically in large letters
        b_letter = Text("B", font_size=60, color=COLOR_B)
        i_letter = Text("I", font_size=60, color=COLOR_I)
        n_letter = Text("N", font_size=60, color=COLOR_N)
        s_letter = Text("S", font_size=60, color=COLOR_S)

        # Issue 25: Shift the acronym letters down one row to center the stack
        self.place_at_grid(b_letter, "B3", scale_factor=0.8)
        self.place_at_grid(i_letter, "C3", scale_factor=0.8)
        self.place_at_grid(n_letter, "D3", scale_factor=0.8)
        self.place_at_grid(s_letter, "E3", scale_factor=0.8)

        self.lecture[0].set_color(WHITE)
        self.play(
            FadeIn(b_letter),
            FadeIn(i_letter),
            FadeIn(n_letter),
            FadeIn(s_letter),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight 'B' (Binary), add checkmark and 'Binary: 2 outcomes'
        self.lecture[1].set_color(COLOR_B)
        check_b = Text("✓", color=COLOR_B, font_size=40)
        # Issue 26: Reposition checkmarks to align with the new letter rows
        self.place_at_grid(check_b, "B2", scale_factor=0.7)
        
        desc_b = Text("Binary: 2 outcomes", color=COLOR_B, font_size=24)
        # Issue 27: Use area positioning to provide more horizontal breathing room
        self.place_in_area(desc_b, "B4", "B5", scale_factor=0.6)

        self.play(
            b_letter.animate.scale(1.2),
            Write(check_b),
            FadeIn(desc_b),
            run_time=1
        )
        self.play(b_letter.animate.scale(1/1.2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight 'I' (Independent), add checkmark and 'Independent: No influence'
        self.lecture[2].set_color(COLOR_I)
        check_i = Text("✓", color=COLOR_I, font_size=40)
        # Issue 26: Reposition checkmarks to align with the new letter rows
        self.place_at_grid(check_i, "C2", scale_factor=0.7)
        
        desc_i = Text("Independent: No influence", color=COLOR_I, font_size=24)
        # Issue 27: Use area positioning to provide more horizontal breathing room
        self.place_in_area(desc_i, "C4", "C5", scale_factor=0.6)

        self.play(
            i_letter.animate.scale(1.2),
            Write(check_i),
            FadeIn(desc_i),
            run_time=1
        )
        self.play(i_letter.animate.scale(1/1.2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight 'N' (Number), add checkmark and 'Number: Fixed n=10'
        self.lecture[3].set_color(COLOR_N)
        check_n = Text("✓", color=COLOR_N, font_size=40)
        # Issue 26: Reposition checkmarks to align with the new letter rows
        self.place_at_grid(check_n, "D2", scale_factor=0.7)
        
        desc_n = Text("Number: Fixed n=10", color=COLOR_N, font_size=24)
        # Issue 27: Use area positioning to provide more horizontal breathing room
        self.place_in_area(desc_n, "D4", "D5", scale_factor=0.6)

        self.play(
            n_letter.animate.scale(1.2),
            Write(check_n),
            FadeIn(desc_n),
            run_time=1
        )
        self.play(n_letter.animate.scale(1/1.2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight 'S' (Same probability), add checkmark and 'Same prob: Constant p'
        self.lecture[4].set_color(COLOR_S)
        check_s = Text("✓", color=COLOR_S, font_size=40)
        # Issue 26: Reposition checkmarks to align with the new letter rows
        self.place_at_grid(check_s, "E2", scale_factor=0.7)
        
        desc_s = Text("Same prob: Constant p", color=COLOR_S, font_size=24)
        # Issue 27: Use area positioning to provide more horizontal breathing room
        self.place_in_area(desc_s, "E4", "E5", scale_factor=0.6)

        self.play(
            s_letter.animate.scale(1.2),
            Write(check_s),
            FadeIn(desc_s),
            run_time=1
        )
        self.play(s_letter.animate.scale(1/1.2), run_time=0.5)
        self.wait(2)
