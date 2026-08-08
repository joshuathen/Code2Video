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
        # Section data
        title = "Prerequisite: The Binary Heartbeat"
        lines = [
            "Binary counting acts like a rhythmic visual odometer.",
            "Each move flips specific bits in a repeating cycle.",
            "The rightmost bit changes every single step."
        ]
        
        # Setup the layout
        self.setup_layout(title, lines)
        
        # Colors
        color_white = WHITE
        color_highlight = "#FFFF00" # Yellow for bits that change

        # === Animation for Lecture Line 1 ===
        # Display the binary sequence '0 0 0' in #FFFFFF at screen center accompanied by [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/od.svg].
        self.lecture[0].set_color(color_white)
        
        # Creating digits separately to allow individual animation
        d1 = MathTex("0", color=color_white)
        d2 = MathTex("0", color=color_white)
        d3 = MathTex("0", color=color_white)
        binary_digits = VGroup(d1, d2, d3).arrange(RIGHT, buff=0.8)
        
        # Issue 23 Fix: Positioning and scaling
        self.place_in_area(binary_digits, "C2", "D6", scale_factor=2.2)
        
        # Issue 17 Fix: Odometer icon
        odometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/od.svg", color=WHITE)
        self.place_at_grid(odometer, "B4", scale_factor=0.6)
        
        self.play(
            Write(binary_digits),
            FadeIn(odometer, shift=DOWN*0.2)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Increment to '0 0 1' and flash the rightmost digit in #FFFF00.
        self.lecture[1].set_color(color_highlight)
        
        # Use scale_factor=2.2 as per binary_digits group scale
        d3_new = MathTex("1", color=color_highlight).scale(2.2).move_to(d3)
        
        self.play(
            FadeOut(d3, shift=UP*0.3),
            FadeIn(d3_new, shift=UP*0.3),
            run_time=0.8
        )
        self.play(Flash(d3_new, color=color_highlight, line_length=0.4))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Increment to '0 1 0' and flash the middle digit in #FFFF00.
        self.lecture[2].set_color(color_highlight)
        
        # Update d3 back to 0 and d2 to 1
        d3_final = MathTex("0", color=color_white).scale(2.2).move_to(d3)
        d2_new = MathTex("1", color=color_highlight).scale(2.2).move_to(d2)
        
        self.play(
            # d3 goes from 1 back to 0
            FadeOut(d3_new, shift=UP*0.3),
            FadeIn(d3_final, shift=UP*0.3),
            # d2 goes from 0 to 1
            FadeOut(d2, shift=UP*0.3),
            FadeIn(d2_new, shift=UP*0.3),
            run_time=0.8
        )
        self.play(Flash(d2_new, color=color_highlight, line_length=0.4))
        self.wait(2)
