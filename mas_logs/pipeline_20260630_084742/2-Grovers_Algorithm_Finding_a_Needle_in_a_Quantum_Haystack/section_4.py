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
        # Configuration
        title = "Step 2: The Diffusion Operator (Reflection about the Mean)"
        lines = [
            "The Diffusion Operator reflects amplitudes about their mean.",
            "First, we calculate the average of all current amplitudes.",
            "Each amplitude moves to the opposite side of average.",
            "The negative target amplitude leaps high above the rest.",
            "This process effectively amplifies the correct answer's probability."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        COLOR_TARGET = "#FFD700"  # Gold
        COLOR_OTHER = "#00FFFF"   # Cyan
        COLOR_MEAN = "#888888"    # Grey
        COLOR_FLASH = "#FFFFFF"   # White

        # Initialize Bar Chart Elements
        # We need 8 bars. 3rd bar is target.
        # Starting state from Section 3: 7 bars at +1 height, 1 bar at -1 height.
        base_h = 0.8
        target_index = 2
        
        bars = VGroup()
        x_axis = Line(LEFT * 2, RIGHT * 2, color=WHITE)
        
        for i in range(8):
            h = -base_h if i == target_index else base_h
            color = COLOR_TARGET if i == target_index else COLOR_OTHER
            bar = Rectangle(
                width=0.3, 
                height=abs(h), 
                fill_color=color, 
                fill_opacity=0.8, 
                stroke_width=1
            )
            # Anchor at the x-axis
            if h > 0:
                bar.next_to(x_axis.point_from_proportion(i/7), UP, buff=0)
            else:
                bar.next_to(x_axis.point_from_proportion(i/7), DOWN, buff=0)
            bars.add(bar)

        chart = VGroup(x_axis, bars)
        # Resolved Issue 28 & 29: Positioned lower (C-F) and restricted to col 5 to avoid label clipping.
        self.place_in_area(chart, "C1", "F5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(x_axis), FadeIn(bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Average calculation: (7*1 + 1*(-1)) / 8 = 6/8 = 0.75
        # In our scale, average is 0.75 * base_h
        mean_y_val = 0.75 * base_h
        # Position of mean line relative to x_axis
        mean_line_start = x_axis.get_start() + UP * mean_y_val
        mean_line_end = x_axis.get_end() + UP * mean_y_val
        mean_line = DashedLine(mean_line_start, mean_line_end, color=COLOR_MEAN)
        mean_label = Text("Mean", font_size=16, color=COLOR_MEAN).next_to(mean_line, RIGHT, buff=0.1)

        self.play(Create(mean_line), FadeIn(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # New heights based on prompt requirements (Target 3x, Others 0.2x)
        # Note: These are relative to the original base_h magnitude.
        target_new_h = 3.0 * base_h
        others_new_h = 0.2 * base_h
        
        animations = []
        for i, bar in enumerate(bars):
            if i == target_index:
                # Reflect target: from -base_h to target_new_h
                new_bar = Rectangle(
                    width=0.3, height=target_new_h, 
                    fill_color=COLOR_TARGET, fill_opacity=0.8, stroke_width=1
                ).next_to(x_axis.point_from_proportion(i/7), UP, buff=0)
            else:
                # Reflect others: from base_h to others_new_h
                new_bar = Rectangle(
                    width=0.3, height=others_new_h, 
                    fill_color=COLOR_OTHER, fill_opacity=0.8, stroke_width=1
                ).next_to(x_axis.point_from_proportion(i/7), UP, buff=0)
            animations.append(Transform(bar, new_bar))

        self.play(*animations, run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # The growth/shrink is already visual from previous transform, 
        # but let's emphasize the target leaping.
        target_bar = bars[target_index]
        self.play(
            target_bar.animate.set_fill(opacity=1.0).scale(1.1, about_edge=DOWN),
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        # Highlight flash
        flash_rect = target_bar.copy().set_fill(COLOR_FLASH, opacity=1).set_stroke(COLOR_FLASH, width=2)
        self.add(flash_rect)
        self.play(FadeOut(flash_rect, scale=1.5), run_time=1)
        
        self.wait(2)
